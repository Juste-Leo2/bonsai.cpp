"""Test driver Python pour test_b1_linear.c

Génère les fichiers binaires d'entrée, exécute le test C, et compare le résultat
avec la référence PyTorch.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROJECT_ROOT


def test_b1_linear_via_c_binary():
    """Test end-to-end: génère données, lance test_b1_linear.c, compare."""
    test_dir = PROJECT_ROOT / "tests" / "c_ops" / "_test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Générer les données de test
    torch.manual_seed(42)
    batch = 1
    in_dim = 64
    out_dim = 32

    act = torch.randn(batch, in_dim)
    weight = torch.randn(out_dim, in_dim)

    # Référence PyTorch
    expected = F.linear(act, weight).numpy()

    # 2. Sauvegarder en fichiers binaires
    act_path = test_dir / "act.bin"
    weight_path = test_dir / "weight.bin"
    out_path = test_dir / "out.bin"

    act.numpy().astype(np.float32).tofile(act_path)
    weight.numpy().astype(np.float32).tofile(weight_path)

    # 3. Exécuter le test C
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

    # 4. Lire le résultat
    out_data = np.fromfile(out_path, dtype=np.float32).reshape(batch, out_dim)

    # 5. Comparer avec tolérance (la quantification B1_0 n'est pas lossless)
    # L'erreur relative est généralement faible (< 1-2%)
    cos_sim = np.dot(expected.ravel(), out_data.ravel()) / (
        np.linalg.norm(expected.ravel()) * np.linalg.norm(out_data.ravel())
    )

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Expected shape: {expected.shape}, got: {out_data.shape}")
    print(f"  Expected mean: {expected.mean():.6f}, got: {out_data.mean():.6f}")
    print(f"  Expected std: {expected.std():.6f}, got: {out_data.std():.6f}")

    # Tolérance: le B1_0 est une approximation, donc pas d'égalité parfaite
    assert cos_sim > 0.999, f"Cosine similarity too low: {cos_sim}"

    # Tolérance plus strict sur la forme et l'absence de NaN
    assert not np.isnan(out_data).any()
    assert not np.isinf(out_data).any()
