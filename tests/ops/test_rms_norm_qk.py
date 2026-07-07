"""Test unitaire: RMS Norm avec scaling par head.

L'implémentation C utilise ggml_rms_norm avec un facteur epsilon=1e-6,
puis multiplie par un vecteur de poids par head.
"""

import numpy as np
import pytest
import torch

from tests.conftest import PROJECT_ROOT


def rms_norm_manual(x, weight, eps=1e-6):
    """Version Python de rms_norm_qk (diffuser_graph.cpp).

    x: [head_dim, n_heads, seq]
    weight: [head_dim]
    """
    # RMS Norm: x / sqrt(mean(x^2) + eps)
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=0, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)

    # Scale par head: [head_dim, 1, 1]
    out = x_norm * weight.view(-1, 1, 1)
    return out


def test_rms_norm_basic():
    """Test RMS Norm basique avec des valeurs simples."""
    torch.manual_seed(42)

    head_dim = 128
    n_heads = 24
    seq = 4096

    x = torch.randn(head_dim, n_heads, seq)
    weight = torch.ones(head_dim)

    out = rms_norm_manual(x, weight)

    # La norme RMS de l'output doit être proche de 1 pour chaque élément
    # (après normalisation mais avant le scale de 1)
    # Vérifie juste que le résultat n'a pas de NaN/Inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Test avec un weight non-unitaire
    weight2 = torch.randn(head_dim) * 2.0 + 1.0
    out2 = rms_norm_manual(x, weight2)
    assert not torch.isnan(out2).any()
    assert not torch.isinf(out2).any()


def test_rms_norm_zero_input():
    """Test RMS Norm avec input = 0 (doit donner 0)."""
    head_dim = 128
    n_heads = 24
    seq = 4096

    x = torch.zeros(head_dim, n_heads, seq)
    weight = torch.ones(head_dim)

    out = rms_norm_manual(x, weight)

    # Avec input=0, la variance est 0, donc sqrt(0 + eps) = sqrt(eps)
    # Le résultat est 0 / sqrt(eps) = 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_rms_norm_very_small_values():
    """Test la stabilité numérique avec des valeurs très petites."""
    torch.manual_seed(43)

    head_dim = 128
    n_heads = 24
    seq = 4096

    x = torch.randn(head_dim, n_heads, seq) * 1e-8
    weight = torch.ones(head_dim)

    out = rms_norm_manual(x, weight)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
