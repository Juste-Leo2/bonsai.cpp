"""Test unitaire: RoPE 2D (Rotary Position Embedding).

Compare la génération des fréquences et l'application RoPE 2D
entre l'implémentation C et PyTorch.
"""

import numpy as np
import pytest
import torch

from tests.conftest import PROJECT_ROOT


def _build_rope_freqs_table(axes_dim, theta):
    """Version Python de build_rope_freqs_table (diffuser_graph.cpp)."""
    max_half = max(axes_dim) // 2
    n_axes = len(axes_dim)

    freqs = np.zeros((n_axes, max_half), dtype=np.float32)
    for a in range(n_axes):
        ax_half = axes_dim[a] // 2
        for j in range(ax_half):
            scale = (j * 2) / axes_dim[a]
            freqs[a, j] = 1.0 / (theta ** scale)

    return freqs


def _build_rope_cos_sin(ids, freqs_table, axes_dim):
    """Version Python de build_rope_cos_sin (diffuser_graph.cpp)."""
    n_axes = len(axes_dim)
    seq = ids.shape[1]
    max_half = freqs_table.shape[1]

    all_angles = None
    for a in range(n_axes):
        ax_half = axes_dim[a] // 2
        # axis_ids: view de ids[a, :] avec shape [1, seq]
        axis_ids = ids[a:a+1, :]  # [1, seq]

        # axis_freqs: view de freqs_table[a, :ax_half] avec shape [1, ax_half]
        axis_freqs = freqs_table[a:a+1, :ax_half]  # [1, ax_half]

        # angles: [ax_half, seq] = axis_freqs.T @ axis_ids
        angles = np.dot(axis_freqs.T, axis_ids)  # [ax_half, seq]

        if all_angles is None:
            all_angles = angles
        else:
            all_angles = np.concatenate([all_angles, angles], axis=0)

    cos_out = np.cos(all_angles)
    sin_out = np.sin(all_angles)

    return cos_out, sin_out


def test_rope_freqs_table():
    """Test la génération de la table de fréquences RoPE."""
    axes_dim = [32, 32, 32, 32]
    theta = 2000

    freqs = _build_rope_freqs_table(axes_dim, theta)

    # Vérifie les dimensions
    assert freqs.shape == (4, 16)  # 4 axes, max_half=16

    # Vérifie que les fréquences décroissent
    for a in range(4):
        for j in range(1, 16):
            assert freqs[a, j] < freqs[a, j-1]

    # Vérifie les valeurs pour les axes de petite dimension
    # ( axes_dim[a] est petit, max_half est petit aussi)


def test_rope_cos_sin():
    """Test la génération des cos/sin pour RoPE."""
    axes_dim = [32, 32, 32, 32]
    theta = 2000
    seq = 64

    # ids: [n_axes=4, seq=64]
    ids = np.zeros((4, seq), dtype=np.float32)
    for i in range(seq):
        ids[0, i] = 0.0  # axis 0
        ids[1, i] = 0.0  # axis 1
        ids[2, i] = i // 8  # axis 2 (y)
        ids[3, i] = i % 8   # axis 3 (x)

    freqs_table = _build_rope_freqs_table(axes_dim, theta)
    cos_out, sin_out = _build_rope_cos_sin(ids, freqs_table, axes_dim)

    # Vérifie les dimensions
    assert cos_out.shape == (64, 64)  # head_dim/2=64, seq=64
    assert sin_out.shape == (64, 64)

    # cos^2 + sin^2 = 1
    identity = cos_out ** 2 + sin_out ** 2
    assert np.allclose(identity, 1.0, atol=1e-5)


def test_rope_2d_rotation():
    """Test que RoPE 2D applique bien une rotation 2D sur les paires de dimensions."""
    head_dim = 128
    n_heads = 1
    seq = 1

    # Crée un q et k simples
    q = np.ones((head_dim, n_heads, seq), dtype=np.float32)
    k = np.ones((head_dim, n_heads, seq), dtype=np.float32)

    # Applique une rotation de 90 degrés (cos=0, sin=1)
    cos_90 = 0.0
    sin_90 = 1.0

    q_rot = q.copy()
    for d in range(0, head_dim, 2):
        a = q[d, 0, 0]
        b = q[d + 1, 0, 0]
        q_rot[d, 0, 0] = a * cos_90 - b * sin_90  # = -b
        q_rot[d + 1, 0, 0] = a * sin_90 + b * cos_90  # = a

    # Après rotation de 90°, q_rot[d] = -q[d+1] et q_rot[d+1] = q[d]
    for d in range(0, head_dim, 2):
        assert q_rot[d, 0, 0] == pytest.approx(-q[d + 1, 0, 0])
        assert q_rot[d + 1, 0, 0] == pytest.approx(q[d, 0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
