import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import safe_open as np_safe_open
from diffusers import Flux2Transformer2DModel

from tests.run_diffuser import (
    DiffuserParams,
    generate_synthetic_input,
    noise_pred_c_format,
)


BLOCK_SIZE = 32
BLOCK_BYTES = 6


def dequantize_b1_0(raw: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """Dequantize a B1_0 packed uint8 tensor back to fp32 (out_dim, in_dim)."""
    num_blocks = in_dim // BLOCK_SIZE
    raw = raw.reshape(out_dim, num_blocks, BLOCK_BYTES)
    lo = raw[:, :, 0].astype(np.uint16)
    hi = raw[:, :, 1].astype(np.uint16)
    scales = (lo | (hi << 8)).view(np.float16).astype(np.float32)
    bits = np.unpackbits(raw[:, :, 2:6], axis=-1, bitorder="little").astype(np.float32)
    signs = 2.0 * bits - 1.0
    return (signs * scales[:, :, np.newaxis]).reshape(out_dim, in_dim)


def load_hf_model_from_packed(safetensors_path: str, params: DiffuserParams, device="cpu"):
    """Load a Flux2Transformer2DModel from a B1_0-packed safetensors file."""
    print(f"  Loading model from {safetensors_path} ...", flush=True)
    model = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=params.in_channels,
        out_channels=params.in_channels,
        num_layers=params.depth,
        num_single_layers=params.depth_single_blocks,
        attention_head_dim=params.head_dim,
        num_attention_heads=params.num_heads,
        joint_attention_dim=params.context_in_dim,
        mlp_ratio=params.mlp_ratio,
        axes_dims_rope=params.axes_dim,
        rope_theta=params.theta,
        guidance_embeds=False,
    )

    state_dict = {}
    with np_safe_open(safetensors_path, framework="np") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == np.uint8:
                out_dim = tensor.shape[0]
                packed_cols = tensor.shape[1]
                in_dim = (packed_cols // BLOCK_BYTES) * BLOCK_SIZE
                state_dict[key] = torch.from_numpy(dequantize_b1_0(tensor, out_dim, in_dim))
            else:
                state_dict[key] = torch.from_numpy(tensor.astype(np.float32))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)

    return model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm) if norm > 0 else 0.0


def run_c_binary(diffuser_binary: Path, diffuser_model: Path, emb_path: Path, workdir: Path):
    """Lance le binaire C bonsai_diffuser et retourne le vecteur noise_pred."""
    result = subprocess.run(
        [
            str(diffuser_binary),
            "--model", str(diffuser_model),
            "--embedding", str(emb_path),
            "-h", "64", "-w", "64",
            "--steps", "1", "--threads", "8",
        ],
        cwd=workdir,
    )
    assert result.returncode == 0, \
        f"bonsai_diffuser exited with code {result.returncode}"

    lat_final = np.fromfile(workdir / "latents_final.bin", dtype=np.float32)
    lat_initial = np.fromfile(workdir / "tmp" / "x_input.bin", dtype=np.float32)
    assert lat_final.size == lat_initial.size
    return (lat_final - lat_initial).reshape(4096, 128)


@pytest.mark.diffuser
def test_diffuser_matches_hf(
    diffuser_packed: Path,
    diffuser_binary: Path,
):
    params = DiffuserParams()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── 1. Generate synthetic inputs ──────────────────────────────
        print("Generating synthetic inputs...", flush=True)
        inputs = generate_synthetic_input(params, seed=42)

        c_tmp = tmp / "tmp"
        c_tmp.mkdir(exist_ok=True)
        x_input_path = c_tmp / "x_input.bin"
        lat_c = inputs["latents"][0].contiguous().float()
        lat_c.numpy().tofile(x_input_path)
        print(f"  x_input.bin: {list(lat_c.shape)}", flush=True)

        emb_path = tmp / "embeddings.bin"
        emb_c = inputs["embeddings"][0].contiguous().float()
        emb_c.numpy().tofile(emb_path)
        print(f"  embeddings.bin: {list(emb_c.shape)}", flush=True)

        # ── 2. Run C binary (B1_0 packed safetensors) ─────────────────
        print("Running C binary (B1_0 packed model)...", flush=True)
        np_cpp = run_c_binary(diffuser_binary, diffuser_packed, emb_path, tmp)
        print(f"  C noise_pred: μ={np_cpp.mean():.4f} σ={np_cpp.std():.4f} "
              f"min={np_cpp.min():.4f} max={np_cpp.max():.4f}", flush=True)

        # ── 3. HF reference from packed safetensors ───────────────────
        t0 = time.time()
        print("\n--- HF reference (dequantized B1_0 from packed safetensors) ---", flush=True)
        model = load_hf_model_from_packed(str(diffuser_packed), params, device)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        np_hf = noise_pred_c_format(model, inputs, device).cpu().numpy()
        print(f"  HF forward: {time.time() - t1:.1f}s", flush=True)
        print(f"  stats: μ={np_hf.mean():.6f} σ={np_hf.std():.6f} "
              f"min={np_hf.min():.6f} max={np_hf.max():.6f}", flush=True)

        # ── 4. Multi-metric comparison ─────────────────────────────
        print(f"\n{'='*60}", flush=True)

        # Cosine similarity
        sim = cosine_similarity(np_hf, np_cpp)
        print(f"  Cosine similarity:       {sim:.6f}", flush=True)

        # MSE / PSNR
        mse = float(np.mean((np_hf - np_cpp) ** 2))
        max_val = float(max(np_hf.max(), np_cpp.max()) - min(np_hf.min(), np_cpp.min()))
        psnr = float(20.0 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")
        print(f"  MSE:                     {mse:.6f}", flush=True)
        print(f"  PSNR (dynamic range):    {psnr:.2f} dB", flush=True)

        # Pearson correlation (per-channel averaged)
        ch_corrs = []
        for c in range(np_hf.shape[1]):
            hf_ch = np_hf[:, c]
            cpp_ch = np_cpp[:, c]
            hf_c = hf_ch - hf_ch.mean()
            cpp_c = cpp_ch - cpp_ch.mean()
            denom = np.sqrt(np.dot(hf_c, hf_c) * np.dot(cpp_c, cpp_c))
            ch_corrs.append(float(np.dot(hf_c, cpp_c) / denom) if denom > 0 else 0.0)
        pearson = float(np.mean(ch_corrs))
        print(f"  Mean Pearson r (128 ch): {pearson:.6f}", flush=True)

        # Directional agreement: sign of the noise prediction across tokens
        sign_agree = float(np.mean(np.sign(np_hf) == np.sign(np_cpp)))
        print(f"  Sign agreement:          {sign_agree*100:.2f}%", flush=True)

        # Per-position std: how much does each pixel deviate in noise space?
        diff = np_hf - np_cpp
        print(f"  Diff μ={diff.mean():.6f} σ={diff.std():.6f} "
              f"|σ_diff|/|σ_hf|={diff.std()/np_hf.std():.3f}", flush=True)

        print(f"{'='*60}", flush=True)

        assert sim >= 0.70, \
            f"C vs HF B1_0 cosine {sim:.6f} < 0.70 — probable bug de graphe"


@pytest.mark.diffuser
def test_diffuser_gpu_vs_hf(
    diffuser_packed: Path,
    diffuser_webgpu_binary: Path,
):
    """GPU B1_0 vs HF reference. Cos > 0.70 attendu."""
    if diffuser_webgpu_binary is None:
        pytest.skip("bonsai_diffuser_webgpu not found. Build with WebGPU first.")

    params = DiffuserParams()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using HF device: {device}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── 1. Generate synthetic inputs ──────────────────────────────
        print("Generating synthetic inputs...", flush=True)
        inputs = generate_synthetic_input(params, seed=42)

        c_tmp = tmp / "tmp"
        c_tmp.mkdir(exist_ok=True)
        x_input_path = c_tmp / "x_input.bin"
        lat_c = inputs["latents"][0].contiguous().float()
        lat_c.numpy().tofile(x_input_path)
        print(f"  x_input.bin: {list(lat_c.shape)}", flush=True)

        emb_path = tmp / "embeddings.bin"
        emb_c = inputs["embeddings"][0].contiguous().float()
        emb_c.numpy().tofile(emb_path)
        print(f"  embeddings.bin: {list(emb_c.shape)}", flush=True)

        # ── 2. Run GPU binary (B1_0 packed safetensors) ───────────────
        print("Running GPU binary (B1_0 packed model)...", flush=True)
        np_gpu = run_c_binary(diffuser_webgpu_binary, diffuser_packed, emb_path, tmp)
        print(f"  GPU noise_pred: μ={np_gpu.mean():.4f} σ={np_gpu.std():.4f} "
              f"min={np_gpu.min():.4f} max={np_gpu.max():.4f}", flush=True)

        # ── 3. HF reference from packed safetensors ───────────────────
        t0 = time.time()
        print("\n--- HF reference (dequantized B1_0 from packed safetensors) ---", flush=True)
        model = load_hf_model_from_packed(str(diffuser_packed), params, device)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        np_hf = noise_pred_c_format(model, inputs, device).cpu().numpy()
        print(f"  HF forward: {time.time() - t1:.1f}s", flush=True)
        print(f"  stats: μ={np_hf.mean():.6f} σ={np_hf.std():.6f} "
              f"min={np_hf.min():.6f} max={np_hf.max():.6f}", flush=True)

        # ── 4. Comparison ─────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        sim = cosine_similarity(np_hf, np_gpu)
        print(f"  Cosine similarity:       {sim:.6f}", flush=True)

        mse = float(np.mean((np_hf - np_gpu) ** 2))
        max_val = float(max(np_hf.max(), np_gpu.max()) - min(np_hf.min(), np_gpu.min()))
        psnr = float(20.0 * np.log10(max_val / np.sqrt(mse))) if mse > 0 else float("inf")
        print(f"  MSE:                     {mse:.6f}", flush=True)
        print(f"  PSNR (dynamic range):    {psnr:.2f} dB", flush=True)

        sign_agree = float(np.mean(np.sign(np_hf) == np.sign(np_gpu)))
        print(f"  Sign agreement:          {sign_agree*100:.2f}%", flush=True)

        diff = np_hf - np_gpu
        print(f"  Diff μ={diff.mean():.6f} σ={diff.std():.6f} "
              f"|σ_diff|/|σ_hf|={diff.std()/np_hf.std():.3f}", flush=True)
        print(f"{'='*60}", flush=True)

        assert sim >= 0.70, \
            f"GPU vs HF B1_0 cosine {sim:.6f} < 0.70 — probable bug de graphe"
