import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.run_diffuser import (
    DiffuserParams,
    generate_synthetic_input,
    load_hf_model,
    noise_pred_c_format,
)

COSINE_THRESHOLD = 0.90


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm)


def run_c_binary(diffuser_binary: Path, diffuser_gguf: Path, emb_path: Path, workdir: Path):
    """Lance le binaire C bonsai_diffuser et retourne le vecteur noise_pred."""
    result = subprocess.run(
        [
            str(diffuser_binary),
            "--model", str(diffuser_gguf),
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
    return (lat_final - lat_initial).reshape(128, 4096)


@pytest.mark.diffuser
def test_diffuser_matches_hf(
    diffuser_model: Path,
    diffuser_gguf: Path,
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

        # C binary looks for tmp/x_input.bin (hardcoded path)
        c_tmp = tmp / "tmp"
        c_tmp.mkdir(exist_ok=True)
        x_input_path = c_tmp / "x_input.bin"
        lat_c = inputs["latents"][0].t().contiguous().float()
        lat_c.numpy().tofile(x_input_path)
        print(f"  x_input.bin: {list(lat_c.shape)}", flush=True)

        emb_path = tmp / "embeddings.bin"
        emb_c = inputs["embeddings"][0].t().contiguous().float()
        emb_c.numpy().tofile(emb_path)
        print(f"  embeddings.bin: {list(emb_c.shape)}", flush=True)

        # ── 2. Run C binary (once, reuse across comparisons) ──────────
        print("Running C binary...", flush=True)
        np_cpp = run_c_binary(diffuser_binary, diffuser_gguf, emb_path, tmp)
        print(f"  C noise_pred: μ={np_cpp.mean():.4f} σ={np_cpp.std():.4f} "
              f"min={np_cpp.min():.4f} max={np_cpp.max():.4f}", flush=True)

        # ── 3. HF reference (fp32 weights) ────────────────────────────
        t0 = time.time()
        print("\n--- HF reference (fp32 weights) ---", flush=True)
        model_fp = load_hf_model(str(diffuser_model), params, device, quantize_b1_0=False)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)
        t1 = time.time()
        np_fp = noise_pred_c_format(model_fp, inputs, device).cpu().numpy()
        print(f"  HF forward: {time.time() - t1:.1f}s", flush=True)
        print(f"  stats: μ={np_fp.mean():.6f} σ={np_fp.std():.6f} "
              f"min={np_fp.min():.6f} max={np_fp.max():.6f}", flush=True)
        sim_fp = cosine_similarity(np_fp, np_cpp)
        print(f"  Cosine similarity (C vs HF fp32): {sim_fp:.6f}", flush=True)

        # ── 4. HF reference (B1_0 quantized weights) ──────────────────
        del model_fp
        if device == "cuda":
            torch.cuda.empty_cache()

        t0 = time.time()
        print("\n--- HF reference (B1_0 quantized weights) ---", flush=True)
        model_b1 = load_hf_model(str(diffuser_model), params, device, quantize_b1_0=True)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)
        t1 = time.time()
        np_b1 = noise_pred_c_format(model_b1, inputs, device).cpu().numpy()
        print(f"  HF forward: {time.time() - t1:.1f}s", flush=True)
        print(f"  stats: μ={np_b1.mean():.6f} σ={np_b1.std():.6f} "
              f"min={np_b1.min():.6f} max={np_b1.max():.6f}", flush=True)
        sim_b1 = cosine_similarity(np_b1, np_cpp)
        print(f"  Cosine similarity (C vs HF B1_0 quantized): {sim_b1:.6f}", flush=True)

        # ── 5. Conclusions ────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print(f"  C vs HF fp32:      {sim_fp:.6f}", flush=True)
        print(f"  C vs HF B1_0:      {sim_b1:.6f}", flush=True)
        print(f"  HF fp32 vs B1_0:   {cosine_similarity(np_fp, np_b1):.6f}", flush=True)
        print(f"{'='*60}", flush=True)

        # Le test valide si la version B1_0 quantifiée de PyTorch
        # corrèle fortement avec le binaire C (graphe identique).
        # Le seuil est plus bas car la quantification 1-bit est brutale.
        assert sim_b1 >= 0.95, \
            f"C vs B1_0 quantized cosine {sim_b1:.6f} < 0.95 — probable bug de graphe"