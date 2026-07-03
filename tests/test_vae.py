import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.run_vae import (
    generate_synthetic_latent,
    load_decoder_weights,
    load_bn_stats,
    preprocess_latent,
    save_tensor_as_png,
    compute_metrics,
)

MSE_THRESHOLD = 5e-5  # PSNR ~43 dB


def test_vae_decoder_matches_cpp(vae_model: Path, vae_binary: Path):
    latent_h = 16
    latent_w = 16

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── 1. Write synthetic latent .bin for C++ ────────────────────
        latent_np = generate_synthetic_latent(latent_h, latent_w)
        latent_bin = tmp / "latent.bin"
        latent_np.tofile(latent_bin)

        # ── 2. Python reference PNG ───────────────────────────────────
        device = "cpu"
        decoder = load_decoder_weights(str(vae_model), device=device)
        running_mean, running_var = load_bn_stats(str(vae_model))
        running_mean = running_mean.to(device)
        running_var = running_var.to(device)

        latent = torch.from_numpy(latent_np).to(device)
        x = preprocess_latent(latent, running_mean, running_var)
        with torch.no_grad():
            out = decoder(x)

        py_png = tmp / "py_out.png"
        save_tensor_as_png(out, py_png)

        # ── 3. C++ bonsai_vae PNG ─────────────────────────────────────
        cpp_png = tmp / "cpp_out.png"
        result = subprocess.run(
            [
                str(vae_binary),
                "--model", str(vae_model),
                "--latent", str(latent_bin),
                "--output", str(cpp_png),
                "--H", str(latent_h),
                "--W", str(latent_w),
                "--threads", "4",
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, \
            f"bonsai_vae exited with code {result.returncode}\n{result.stderr}"
        assert cpp_png.exists(), "C++ did not produce output PNG"

        # ── 4. Compare ────────────────────────────────────────────────
        mse, psnr = compute_metrics(str(py_png), str(cpp_png))
        print(f"  MSE  = {mse:.8f}")
        print(f"  PSNR = {psnr:.2f} dB")

        assert mse < MSE_THRESHOLD, \
            f"MSE={mse:.8f} exceeds threshold {MSE_THRESHOLD:.8f} (PSNR={psnr:.2f} dB)"
