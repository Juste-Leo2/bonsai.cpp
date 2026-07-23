#!/usr/bin/env python3
"""Génère des données de test binaires pour les tests C d'opérations du diffuseur.

Usage:
    python tests/c_ops/generate_test_data.py <output_dir>
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def write_float_bin(path, tensor):
    """Écrit un tenseur PyTorch en fichier binaire float32 natif."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    # Aplatir en row-major (C-style)
    arr_flat = arr.ravel(order='C')
    with open(path, 'wb') as f:
        f.write(arr_flat.tobytes())
    return arr_flat


def generate_b1_linear_data(output_dir, batch=1, in_dim=64, out_dim=32, seed=42):
    """Génère des données pour le test b1_linear."""
    torch.manual_seed(seed)

    act = torch.randn(batch, in_dim)
    weight = torch.randn(out_dim, in_dim)

    # Référence PyTorch
    expected = F.linear(act, weight)

    # Sauvegarder
    act_path = output_dir / f"b1_act_{batch}x{in_dim}.bin"
    weight_path = output_dir / f"b1_weight_{out_dim}x{in_dim}.bin"
    expected_path = output_dir / f"b1_expected_{batch}x{out_dim}.bin"

    write_float_bin(act_path, act)
    write_float_bin(weight_path, weight)
    write_float_bin(expected_path, expected)

    # Info
    info = {
        "batch": batch,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "act_path": str(act_path),
        "weight_path": str(weight_path),
        "expected_path": str(expected_path),
    }

    return info, expected


def generate_rope_data(output_dir, head_dim=128, n_heads=24, seq=4096, seed=42):
    """Génère des données pour le test RoPE 2D."""
    # Pour l'instant, c'est un placeholder
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default="tests/c_ops/test_data")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Génération des données de test dans {args.output}...")

    # Test b1_linear avec petites dimensions
    info, ref = generate_b1_linear_data(args.output, batch=1, in_dim=64, out_dim=32)
    print(f"  b1_linear (petit): {info}")
    print(f"    Référence PyTorch: shape={ref.shape}, mean={ref.mean():.6f}, std={ref.std():.6f}")

    # Test b1_linear avec dimensions réelles
    info, ref = generate_b1_linear_data(args.output, batch=1, in_dim=3072, out_dim=3072)
    print(f"  b1_linear (réel): {info}")
    print(f"    Référence PyTorch: shape={ref.shape}, mean={ref.mean():.6f}, std={ref.std():.6f}")

    print("Done!")


if __name__ == "__main__":
    main()
