import struct
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

B1_0_BLOCK_SIZE = 32
B1_0_BLOCK_BYTES = 6

def dequantize_b1_0(raw_bytes: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    num_blocks = in_dim // B1_0_BLOCK_SIZE
    raw = raw_bytes.reshape(out_dim, num_blocks, B1_0_BLOCK_BYTES)
    # fp16 scale: pack 2 bytes into uint16, reinterpret as float16, convert to f32
    lo = raw[:, :, 0].astype(np.uint16)
    hi = raw[:, :, 1].astype(np.uint16)
    scales = (lo | (hi << 8)).view(np.float16).astype(np.float32)
    # bits: unpack 4 uint8 → 32 bits
    bits = np.unpackbits(raw[:, :, 2:6], axis=-1, bitorder='little').astype(np.float32)
    signs = 2.0 * bits - 1.0
    return (signs * scales[:, :, np.newaxis]).reshape(out_dim, in_dim)


def read_gguf_tensors(path: str):
    tensors = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def read_str():
            slen = struct.unpack("<Q", f.read(8))[0]
            return f.read(slen).decode("utf-8")

        def read_val(typ_char):
            if typ_char == 8:  # string
                return read_str()
            elif typ_char == 4:  # int32
                return struct.unpack("<i", f.read(4))[0]
            elif typ_char == 5:  # float32
                return struct.unpack("<f", f.read(4))[0]
            elif typ_char == 6:  # bool
                return struct.unpack("?", f.read(1))[0]
            elif typ_char == 2:  # uint32
                return struct.unpack("<I", f.read(4))[0]
            elif typ_char == 9:  # array
                arr_typ = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                return [read_val(arr_typ) for _ in range(arr_len)]
            elif typ_char == 12:  # float64
                return struct.unpack("<d", f.read(8))[0]
            else:
                raise ValueError(f"Unknown KV type: {typ_char}")

        for _ in range(n_kv):
            key = read_str()
            typ = struct.unpack("<I", f.read(4))[0]
            val = read_val(typ)

        for _ in range(n_tensors):
            name = read_str()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = {"shape": tuple(dims), "dtype": dtype, "offset": offset}

        f.seek(0, 2)
        file_size = f.tell()

        tensor_list = list(tensors.items())
        for idx, (name, info) in enumerate(tensor_list):
            dims = info["shape"]
            if info["dtype"] == 0:
                bsize = int(np.prod(dims)) * 4
            elif info["dtype"] == 100:
                in_dim, out_dim = dims[0], dims[1]
                bsize = out_dim * (in_dim // B1_0_BLOCK_SIZE) * B1_0_BLOCK_BYTES
            else:
                raise ValueError(f"Unknown dtype {info['dtype']} for {name}")
            if idx + 1 < len(tensor_list):
                next_off = tensor_list[idx + 1][1]["offset"]
            else:
                next_off = file_size
            info["byte_size"] = min(bsize, next_off - info["offset"])

        for name, info in tensors.items():
            f.seek(info["offset"])
            raw = np.frombuffer(f.read(file_size - info["offset"]), dtype=np.uint8)
            info["data_raw"] = raw[:info["byte_size"]].copy()

    return tensors


def map_safetensors_to_gguf(pt_name: str) -> str:
    name = pt_name
    name = name.replace("single_transformer_blocks.", "blk.")
    name = name.replace("double_stream_modulation", "double_stream")
    return name


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm) if norm > 0 else 1.0


@pytest.mark.weight_parity
def test_weight_parity_safetensors_vs_gguf(
    diffuser_model: Path,
    diffuser_gguf: Path,
):
    print("Loading safetensors...", flush=True)

    gguf_tensors = read_gguf_tensors(str(diffuser_gguf))
    print(f"  GGUF: {len(gguf_tensors)} tensors", flush=True)

    with safe_open(str(diffuser_model), framework="pt", device="cpu") as f:
        pt_keys = f.keys()
        print(f"  Safetensors: {len(pt_keys)} tensors", flush=True)

        results = []
        total = len(pt_keys)
        t_start = time.time()
        for idx, pt_name in enumerate(pt_keys):
            gguf_name = map_safetensors_to_gguf(pt_name)
            assert gguf_name in gguf_tensors, f"Missing {gguf_name} in GGUF"

            pt_tensor = f.get_tensor(pt_name).to(torch.float32).numpy()
            gguf_info = gguf_tensors[gguf_name]
            gguf_raw = gguf_info["data_raw"]
            gguf_shape = gguf_info["shape"]
            gguf_dtype = gguf_info["dtype"]

            if gguf_dtype == 0:
                logical_shape = tuple(reversed(gguf_shape))
                dequant = gguf_raw.view(np.float32).reshape(logical_shape)
                sim = cosine_similarity(pt_tensor, dequant)
                results.append((gguf_name, "F32", sim, None, pt_tensor.shape))
                print(f"  [{idx+1}/{total}] {gguf_name:<45} F32   cos={sim:.6f}  {str(pt_tensor.shape):<20}", flush=True)
            elif gguf_dtype == 100:
                in_dim, out_dim = gguf_shape[0], gguf_shape[1]
                dequant = dequantize_b1_0(gguf_raw, out_dim, in_dim)
                sim = cosine_similarity(pt_tensor, dequant)
                pt_sign = (pt_tensor >= 0).ravel()
                dq_sign = (dequant >= 0).ravel()
                sign_acc = float((pt_sign == dq_sign).mean()) * 100.0
                results.append((gguf_name, "B1_0", sim, sign_acc, pt_tensor.shape))
                print(f"  [{idx+1}/{total}] {gguf_name:<45} B1_0  cos={sim:.6f} sign={sign_acc:6.2f}%  {str(pt_tensor.shape):<20}", flush=True)
            else:
                results.append((gguf_name, f"dtype={gguf_dtype}", 0.0, None, pt_tensor.shape))
                print(f"  [{idx+1}/{total}] {gguf_name:<45} ?     dtype={gguf_dtype}", flush=True)

        elapsed = time.time() - t_start

        # Print summary table
        print(f"\n{'Tensor Name':<45} {'Type':<5} {'Cos Sim':<10} {'Sign%':<8} {'Shape':<20}")
        print("-" * 95)
        for name, dtype, sim, sign_acc, shape in results:
            sim_str = f"{sim:.6f}" if dtype == "F32" or dtype == "B1_0" else "N/A"
            sign_str = f"{sign_acc:.1f}%" if sign_acc is not None else "N/A"
            print(f"{name:<45} {dtype:<5} {sim_str:<10} {sign_str:<8} {str(shape):<20}")

        # Aggregate metrics inside the with block so `results` is in scope
        f32_sims = [r[2] for r in results if r[1] == "F32"]
        b1_sims = [r[2] for r in results if r[1] == "B1_0"]
        b1_signs = [r[3] for r in results if r[3] is not None]

        print(f"\n--- Summary (in {elapsed:.1f}s) ---")
        print(f"F32 tensors:  n={len(f32_sims)}  mean cos={np.mean(f32_sims):.6f}  min cos={np.min(f32_sims):.6f}")
        print(f"B1_0 tensors: n={len(b1_sims)}  mean cos={np.mean(b1_sims):.6f}  min cos={np.min(b1_sims):.6f}")
        if b1_signs:
            print(f"B1_0 sign accuracy: mean={np.mean(b1_signs):.2f}%  min={np.min(b1_signs):.2f}%")

        cos_all = [r[2] for r in results if r[1] in ("F32", "B1_0")]
        assert np.min(cos_all) >= 0.99, \
            f"Minimum cosine similarity {np.min(cos_all):.6f} < 0.99"
