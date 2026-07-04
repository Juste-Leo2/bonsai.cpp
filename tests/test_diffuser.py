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

        # Save embeddings.bin for C
        emb_path = tmp / "embeddings.bin"
        emb_c = inputs["embeddings"][0].t().contiguous().float()
        emb_c.numpy().tofile(emb_path)
        print(f"  embeddings.bin: {list(emb_c.shape)}", flush=True)

        # ── 2. Run HF reference forward ───────────────────────────────
        t0 = time.time()
        print("Loading HF model...", flush=True)
        model = load_hf_model(str(diffuser_model), params, device)
        print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        print("Running HF forward...", flush=True)
        np_ref = noise_pred_c_format(model, inputs, device)  # (128, 4096)
        print(f"  HF forward done in {time.time() - t1:.1f}s", flush=True)
        np_ref_np = np_ref.cpu().numpy()
        print(f"  HF noise_pred: μ={np_ref_np.mean():.6f} σ={np_ref_np.std():.6f} "
              f"min={np_ref_np.min():.6f} max={np_ref_np.max():.6f}", flush=True)

        # ── 3. Run C binary ───────────────────────────────────────────
        # C expects x_input.bin in tmp/ subdir, embeddings from --embedding
        print(f"Running: {diffuser_binary} --model {diffuser_gguf.name} "
              f"--embedding embeddings.bin -h 64 -w 64 --steps 1 --threads 8", flush=True)
        result = subprocess.run(
            [
                str(diffuser_binary),
                "--model", str(diffuser_gguf),
                "--embedding", str(emb_path),
                "-h", "64",
                "-w", "64",
                "--steps", "1",
                "--threads", "8",
            ],
            cwd=tmpdir,
        )

        assert result.returncode == 0, \
            f"bonsai_diffuser exited with code {result.returncode}"

        # ── 4. Read C output and compute noise_pred ───────────────────
        latents_final_path = tmp / "latents_final.bin"
        assert latents_final_path.exists(), "Missing latents_final.bin"

        lat_final = np.fromfile(latents_final_path, dtype=np.float32)
        lat_initial = lat_c.numpy().ravel()
        assert lat_final.size == lat_initial.size, \
            f"Size mismatch: {lat_final.size} vs {lat_initial.size}"

        # noise_pred = latents_final - latents_initial (Euler, dt=1.0)
        np_cpp = lat_final - lat_initial
        np_cpp = np_cpp.reshape(np_ref_np.shape)
        print(f"  C noise_pred: μ={np_cpp.mean():.4f} σ={np_cpp.std():.4f} "
              f"min={np_cpp.min():.4f} max={np_cpp.max():.4f}", flush=True)

        # ── 5. Cosine similarity ──────────────────────────────────────
        sim = cosine_similarity(np_ref_np, np_cpp)
        print(f"  Cosine similarity: {sim:.6f}", flush=True)

        assert sim >= COSINE_THRESHOLD, \
            f"Cosine similarity {sim:.6f} < threshold {COSINE_THRESHOLD}"