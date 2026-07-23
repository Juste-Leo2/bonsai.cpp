"""Test driver Python pour test_b1_linear.c

Génère les fichiers binaires d'entrée, exécute le test C, et compare le résultat
avec la référence déquantifiée B1_0 (pas fp32, car la quantification 1-bit est
intrinsèquement lossy).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROJECT_ROOT

B1_0_BLOCK_SIZE = 32


def dequantize_b1_0(weight: np.ndarray) -> np.ndarray:
    """Dequantize fp32 weights using the B1_0 scheme (same as C binary)."""
    out_dim, in_dim = weight.shape
    n_blocks = in_dim // B1_0_BLOCK_SIZE
    rec = np.zeros_like(weight)
    for r in range(out_dim):
        for blk in range(n_blocks):
            block = weight[r, blk * B1_0_BLOCK_SIZE:(blk + 1) * B1_0_BLOCK_SIZE]
            scale = np.mean(np.abs(block))
            for i in range(B1_0_BLOCK_SIZE):
                rec[r, blk * B1_0_BLOCK_SIZE + i] = (
                    scale if block[i] >= 0 else -scale
                )
    return rec


def test_b1_linear_via_c_binary():
    """Test end-to-end: génère données, lance test_b1_linear.c, compare."""
    test_dir = PROJECT_ROOT / "tests" / "c_ops" / "_test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    batch = 1
    in_dim = 64
    out_dim = 32

    act = torch.randn(batch, in_dim)
    weight = torch.randn(out_dim, in_dim)

    # Reference: full fp32 (for info only)
    expected_fp32 = F.linear(act, weight).numpy()

    # Reference: B1_0 dequantized (the C kernel should match this exactly)
    w_deq = dequantize_b1_0(weight.numpy())
    expected_b1 = (act.numpy() @ w_deq.T)

    act_path = test_dir / "act.bin"
    weight_path = test_dir / "weight.bin"
    out_path = test_dir / "out.bin"

    act.numpy().astype(np.float32).tofile(act_path)
    weight.numpy().astype(np.float32).tofile(weight_path)

    c_binary = PROJECT_ROOT / "build" / "test_b1_linear"
    if not c_binary.exists():
        pytest.skip("C binary not found. Build it first.")

    result = subprocess.run(
        [str(c_binary), str(act_path), str(weight_path), str(out_path),
         str(batch), str(in_dim), str(out_dim)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail(f"C binary failed with code {result.returncode}")

    out_data = np.fromfile(out_path, dtype=np.float32).reshape(batch, out_dim)

    cos_vs_fp32 = np.dot(expected_fp32.ravel(), out_data.ravel()) / (
        np.linalg.norm(expected_fp32.ravel()) * np.linalg.norm(out_data.ravel()))
    cos_vs_b1 = np.dot(expected_b1.ravel(), out_data.ravel()) / (
        np.linalg.norm(expected_b1.ravel()) * np.linalg.norm(out_data.ravel()))

    print(f"  Cosine vs fp32:  {cos_vs_fp32:.6f}")
    print(f"  Cosine vs B1_0:  {cos_vs_b1:.10f}")
    print(f"  Expected (B1_0) μ={expected_b1.mean():.6f} σ={expected_b1.std():.6f}")
    print(f"  C output         μ={out_data.mean():.6f} σ={out_data.std():.6f}")

    assert cos_vs_b1 > 0.9999, f"b1_linear kernel mismatch! cos vs B1_0 ref = {cos_vs_b1}"
    assert not np.isnan(out_data).any()
    assert not np.isinf(out_data).any()
