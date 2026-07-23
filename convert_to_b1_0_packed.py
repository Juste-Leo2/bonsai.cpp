#!/usr/bin/env python3
"""Convert diffusion_pytorch_model.safetensors to a B1_0-packed safetensors.

Transformer block linear weights → B1_0 quantized (U8, 6 bytes per 32-weight block)
Edge projections, norms, time embeddings → raw float32

Output: models/flux2_4b_1bit.safetensors (~1.3 GB instead of ~7 GB)
"""

import os
import sys
import time
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

BLOCK_SIZE = 32
B1_0_BLOCK_BYTES = 6


def quantize_b1_0(tensor: np.ndarray) -> np.ndarray:
    """B1_0 quantize a 2D weight [out_dim, in_dim] → uint8 [out_dim, in_dim//32*6].

    Block format (6 bytes per block of 32 weights):
      bytes 0-1:  scale  (float16, little-endian)
      bytes 2-5:  signs  (32 bits, packed little-endian)
    """
    out_dim, in_dim = tensor.shape
    assert in_dim % BLOCK_SIZE == 0, f"in_dim {in_dim} not divisible by {BLOCK_SIZE}"

    num_blocks = in_dim // BLOCK_SIZE
    reshaped = tensor.reshape(out_dim, num_blocks, BLOCK_SIZE)

    scales = np.mean(np.abs(reshaped), axis=-1).astype(np.float16)
    bits = (reshaped >= 0).astype(np.uint8)
    packed_bits = np.packbits(bits, axis=-1, bitorder="little")

    raw = np.zeros((out_dim, num_blocks, B1_0_BLOCK_BYTES), dtype=np.uint8)
    scales_bytes = scales.view(np.uint8).reshape(out_dim, num_blocks, 2)
    raw[:, :, 0:2] = scales_bytes
    raw[:, :, 2:6] = packed_bits

    return raw.reshape(out_dim, -1)


def should_quantize(name: str, shape: tuple) -> bool:
    """Return True if this tensor should be B1_0 quantized."""
    if not (name.startswith("transformer_blocks.") or
            name.startswith("single_transformer_blocks.")):
        return False
    # 1D tensors are norms/biases — keep F32
    if len(shape) == 1:
        return False
    return True


def main():
    src = "diffusion_pytorch_model.safetensors"
    dst = "models/flux2_4b_1bit.safetensors"

    if not os.path.exists(src):
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs("models", exist_ok=True)

    print(f"Loading {src} ...")
    t0 = time.time()

    output_tensors = {}
    quantized_count = 0
    f32_count = 0
    orig_bytes = 0
    packed_bytes = 0

    with safe_open(src, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total = len(keys)

        for i, key in enumerate(keys):
            tensor_pt = f.get_tensor(key)
            shape = tuple(tensor_pt.shape)

            if should_quantize(key, shape):
                tensor_np = tensor_pt.float().numpy()
                packed = quantize_b1_0(tensor_np)
                output_tensors[key] = packed
                quantized_count += 1
                orig_elems = int(np.prod(shape))
                pack_elems = packed.size
                orig_bytes += orig_elems * 4
                packed_bytes += pack_elems
                pct = (1 - pack_elems / (orig_elems * 4)) * 100
                print(f"  [{i+1:3d}/{total}] B1_0  {key}  "
                      f"{shape} → {packed.shape}  ({pct:.0f}% compression)")
            else:
                tensor_np = tensor_pt.float().numpy()
                output_tensors[key] = tensor_np
                f32_count += 1
                nelems = int(np.prod(shape))
                orig_bytes += nelems * 4
                packed_bytes += nelems * 4
                print(f"  [{i+1:3d}/{total}] F32   {key}  {shape}")

    print(f"\nConverting {total} tensors ({quantized_count} B1_0 + {f32_count} F32) ...")
    print(f"  Original fp32 equivalent:  {orig_bytes / 1024**3:.2f} GB")
    print(f"  Packed size:               {packed_bytes / 1024**3:.2f} GB")
    print(f"  Compression:               {(1 - packed_bytes / orig_bytes) * 100:.1f}%")

    print(f"\nWriting {dst} ({packed_bytes / 1024**2:.1f} MB) ...")
    save_file(output_tensors, dst)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s → {dst} ({os.path.getsize(dst) / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
