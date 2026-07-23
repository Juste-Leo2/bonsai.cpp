#!/usr/bin/env python3
"""Test driver Python pour test_rope_2d.cpp

Génère les entrées, exécute le C, compare avec la référence PyTorch.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tests.conftest import PROJECT_ROOT


def build_rope_freqs_table_py(axes_dim, theta):
    """Version Python de build_rope_freqs_table (identique à diffuser_graph.cpp)."""
    max_half = max(ax // 2 for ax in axes_dim)
    n_axes = len(axes_dim)

    freqs = np.zeros((n_axes, max_half), dtype=np.float32)
    for a in range(n_axes):
        ax_half = axes_dim[a] // 2
        for j in range(ax_half):
            scale = (j * 2) / axes_dim[a]
            freqs[a, j] = 1.0 / (theta ** scale)
    return freqs


def build_rope_cos_sin_py(ids, freqs_table, axes_dim):
    """Version Python de build_rope_cos_sin (identique à diffuser_graph.cpp).

    ids: [n_axes, seq] float32
    """
    n_axes = len(axes_dim)
    seq = ids.shape[1]

    all_angles = []
    for a in range(n_axes):
        ax_half = axes_dim[a] // 2
        axis_ids = ids[a:a+1, :]  # [1, seq]
        axis_freqs = freqs_table[a:a+1, :ax_half]  # [1, ax_half]

        angles = axis_freqs.T @ axis_ids  # [ax_half, seq]
        all_angles.append(angles)

    all_angles = np.concatenate(all_angles, axis=0)  # [total_half, seq]

    cos_out = np.cos(all_angles)
    sin_out = np.sin(all_angles)

    return cos_out, sin_out


def apply_rope_2d_py(q, k, cos_t, sin_t):
    """Version Python de rope_2d_fwd_f32.

    q, k: [head_dim, n_heads, seq]
    cos_t, sin_t: [head_dim/2, seq]
    """
    head_dim, n_heads, seq = q.shape
    half = head_dim // 2

    q_out = np.zeros_like(q)
    k_out = np.zeros_like(k)

    for h in range(n_heads):
        for s in range(seq):
            base = h * head_dim + s * (head_dim * n_heads)
            angle_base = s * half

            for d_pair in range(half):
                even_idx = d_pair * 2 + base
                odd_idx = even_idx + 1
                angle_idx = d_pair + angle_base

                cos_v = cos_t.ravel()[angle_idx]
                sin_v = sin_t.ravel()[angle_idx]

                a_q = q.ravel()[even_idx]
                b_q = q.ravel()[odd_idx]
                q_out.ravel()[even_idx] = a_q * cos_v - b_q * sin_v
                q_out.ravel()[odd_idx] = a_q * sin_v + b_q * cos_v

                a_k = k.ravel()[even_idx]
                b_k = k.ravel()[odd_idx]
                k_out.ravel()[even_idx] = a_k * cos_v - b_k * sin_v
                k_out.ravel()[odd_idx] = a_k * sin_v + b_k * cos_v

    return q_out, k_out


def test_rope_2d_small():
    """Test RoPE 2D avec petites dimensions."""
    test_dir = PROJECT_ROOT / "tests" / "c_ops" / "_test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Petites dimensions pour un test rapide
    head_dim = 8
    n_heads = 2
    seq = 4

    q = np.random.randn(head_dim, n_heads, seq).astype(np.float32)
    k = np.random.randn(head_dim, n_heads, seq).astype(np.float32)

    # Générer cos/sin
    axes_dim = [32, 32, 32, 32]
    theta = 2000.0
    n_axes = 4

    ids = np.zeros((n_axes, seq), dtype=np.float32)
    for s in range(seq):
        ids[2, s] = float(s // 2)
        ids[3, s] = float(s % 2)

    freqs_table = build_rope_freqs_table_py(axes_dim, theta)
    cos_t, sin_t = build_rope_cos_sin_py(ids, freqs_table, axes_dim)

    # Appliquer (référence Python)
    q_ref, k_ref = apply_rope_2d_py(q, k, cos_t, sin_t)

    # Écrire les entrées
    q.tofile(test_dir / "rope_q.bin")
    k.tofile(test_dir / "rope_k.bin")
    cos_t.tofile(test_dir / "rope_cos.bin")
    sin_t.tofile(test_dir / "rope_sin.bin")

    # Exécuter le C
    c_binary = PROJECT_ROOT / "build" / "test_rope_2d"
    if not c_binary.exists():
        print(f"  SKIP: {c_binary} not found")
        return

    result = subprocess.run([
        str(c_binary),
        str(test_dir / "rope_q.bin"),
        str(test_dir / "rope_k.bin"),
        str(test_dir / "rope_cos.bin"),
        str(test_dir / "rope_sin.bin"),
        str(test_dir / "rope_q_out.bin"),
        str(test_dir / "rope_k_out.bin"),
        str(head_dim), str(n_heads), str(seq),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert False, f"C binary failed with code {result.returncode}"

    # Lire les sorties
    q_c = np.fromfile(test_dir / "rope_q_out.bin", dtype=np.float32).reshape(head_dim, n_heads, seq)
    k_c = np.fromfile(test_dir / "rope_k_out.bin", dtype=np.float32).reshape(head_dim, n_heads, seq)

    # Comparer
    max_diff_q = np.max(np.abs(q_c - q_ref))
    max_diff_k = np.max(np.abs(k_c - k_ref))

    print(f"  Max diff Q: {max_diff_q:.10f}")
    print(f"  Max diff K: {max_diff_k:.10f}")
    print(f"  Reference Q first 5: {q_ref.ravel()[:5]}")
    print(f"  C Q first 5:         {q_c.ravel()[:5]}")

    assert max_diff_q < 1e-5, f"Max diff Q too large: {max_diff_q}"
    assert max_diff_k < 1e-5, f"Max diff K too large: {max_diff_k}"
    assert not np.isnan(q_c).any()
    assert not np.isinf(q_c).any()
    print("  PASS!")
    print(result.stdout)


def test_rope_2d_full():
    """Test RoPE 2D avec dimensions réelles du diffuseur (head_dim=128, n_heads=24, seq=64)."""
    test_dir = PROJECT_ROOT / "tests" / "c_ops" / "_test_outputs"
    test_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    head_dim = 128
    n_heads = 24
    seq = 64

    q = np.random.randn(head_dim, n_heads, seq).astype(np.float32)
    k = np.random.randn(head_dim, n_heads, seq).astype(np.float32)

    # Générer cos/sin
    axes_dim = [32, 32, 32, 32]
    theta = 2000.0
    n_axes = 4

    ids = np.zeros((n_axes, seq), dtype=np.float32)
    for s in range(seq):
        ids[2, s] = float(s // 8)
        ids[3, s] = float(s % 8)

    freqs_table = build_rope_freqs_table_py(axes_dim, theta)
    cos_t, sin_t = build_rope_cos_sin_py(ids, freqs_table, axes_dim)
    print(f"  cos_t shape: {cos_t.shape}, sin_t shape: {sin_t.shape}")

    q_ref, k_ref = apply_rope_2d_py(q, k, cos_t, sin_t)

    q.tofile(test_dir / "rope_q_full.bin")
    k.tofile(test_dir / "rope_k_full.bin")
    cos_t.tofile(test_dir / "rope_cos_full.bin")
    sin_t.tofile(test_dir / "rope_sin_full.bin")

    c_binary = PROJECT_ROOT / "build" / "test_rope_2d"
    if not c_binary.exists():
        print(f"  SKIP: {c_binary} not found")
        return

    result = subprocess.run([
        str(c_binary),
        str(test_dir / "rope_q_full.bin"),
        str(test_dir / "rope_k_full.bin"),
        str(test_dir / "rope_cos_full.bin"),
        str(test_dir / "rope_sin_full.bin"),
        str(test_dir / "rope_q_out_full.bin"),
        str(test_dir / "rope_k_out_full.bin"),
        str(head_dim), str(n_heads), str(seq),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert False, f"C binary failed with code {result.returncode}"

    q_c = np.fromfile(test_dir / "rope_q_out_full.bin", dtype=np.float32).reshape(head_dim, n_heads, seq)
    k_c = np.fromfile(test_dir / "rope_k_out_full.bin", dtype=np.float32).reshape(head_dim, n_heads, seq)

    max_diff_q = np.max(np.abs(q_c - q_ref))
    max_diff_k = np.max(np.abs(k_c - k_ref))

    print(f"  Max diff Q: {max_diff_q:.10f}")
    print(f"  Max diff K: {max_diff_k:.10f}")

    assert max_diff_q < 1e-5, f"Max diff Q too large: {max_diff_q}"
    assert max_diff_k < 1e-5, f"Max diff K too large: {max_diff_k}"
    print("  PASS!")
    print(result.stdout)


if __name__ == "__main__":
    test_rope_2d_small()
    test_rope_2d_full()
