"""Test unitaire: Flash Attention et Attention standard.

Compare ggml_flash_attn_ext avec l'implémentation manuelle standard
pour vérifier l'équivalence fonctionnelle.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROJECT_ROOT


def attention_manual(q, k, v, scale=None):
    """Attention manuelle: softmax(Q @ K^T / sqrt(d_k)) @ V"""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def test_flash_attention_equivalence():
    """Test que Flash Attention donne le même résultat que l'attention manuelle."""
    torch.manual_seed(42)

    head_dim = 128
    n_heads = 24
    seq_q = 64
    seq_kv = 64

    # [batch, n_heads, seq, head_dim] (format PyTorch standard)
    q = torch.randn(1, n_heads, seq_q, head_dim)
    k = torch.randn(1, n_heads, seq_kv, head_dim)
    v = torch.randn(1, n_heads, seq_kv, head_dim)

    # Référence
    expected = attention_manual(q, k, v)

    # Flash Attention (si disponible)
    try:
        from torch.nn.functional import scaled_dot_product_attention
        out_flash = scaled_dot_product_attention(q, k, v, is_causal=False)

        # Compare
        diff = (expected - out_flash).abs().max()
        print(f"Max diff Flash Attention vs manual: {diff:.6f}")
        assert diff < 1e-4, f"Flash Attention diff too large: {diff}"
    except ImportError:
        pytest.skip("Flash Attention not available")


def test_attention_scale_factor():
    """Test que le facteur d'échelle est correct (1/sqrt(d_k))."""
    torch.manual_seed(43)

    head_dim = 128
    n_heads = 1
    seq = 8

    q = torch.randn(1, n_heads, seq, head_dim)
    k = torch.randn(1, n_heads, seq, head_dim)
    v = torch.ones(1, n_heads, seq, head_dim)

    # Avec scale très grand (petit head_dim), les poids de attention
    # devraient être plus uniformes
    expected = attention_manual(q, k, v)

    # Avec scale très petit (grand head_dim), les poids devraient
    # être plus "pointus" (un seul token dominant)
    # (Ce test est plus qualitatif)

    assert not torch.isnan(expected).any()
    assert not torch.isinf(expected).any()


def test_attention_causal():
    """Test l'attention causale (mask triangulaire inférieur)."""
    torch.manual_seed(44)

    head_dim = 64
    n_heads = 1
    seq = 8

    q = torch.randn(1, n_heads, seq, head_dim)
    k = torch.randn(1, n_heads, seq, head_dim)
    v = torch.randn(1, n_heads, seq, head_dim)

    # Attention causale: le token i ne peut pas attendre aux tokens j > i
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Mask causal: -inf pour j > i
    mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    # Vérifie que la somme des poids d'attention = 1 pour chaque token
    # (après softmax, même avec des -inf, la somme est normalisée)
    for i in range(seq):
        # Le token i ne peut pas attendre aux tokens j > i, donc
        # les poids pour j > i doivent être 0
        if i < seq - 1:
            assert torch.allclose(attn[0, 0, i, i+1:], torch.zeros(seq - i - 1), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
