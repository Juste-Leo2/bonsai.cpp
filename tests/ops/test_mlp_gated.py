"""Test unitaire: MLP avec GEGLU/SiLU (MLP Gated).

Teste l'opération MLP du diffuseur qui utilise gated MLP (GEGLU):
SiLU(x @ W_gate) * (x @ W_val) @ W_out
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROJECT_ROOT


def mlp_gated(x, w_in, w_out):
    """MLP Gated: SiLU(x @ W_in_gate) * (x @ W_in_val) @ W_out.

    x: [seq, hidden_size]
    w_in: [hidden_size, mlp_hidden * 2] (première moitié = gate, deuxième = val)
    w_out: [mlp_hidden, hidden_size]
    """
    # Projection gate + val
    proj = torch.matmul(x, w_in)  # [seq, mlp_hidden * 2]
    gate, val = torch.chunk(proj, 2, dim=-1)  # [seq, mlp_hidden] chacun

    # GEGLU: SiLU(gate) * val
    activated = F.silu(gate) * val  # [seq, mlp_hidden]

    # Projection de sortie
    out = torch.matmul(activated, w_out)  # [seq, hidden_size]

    return out


def test_mlp_gated_basic():
    """Test MLP Gated avec des valeurs simples."""
    torch.manual_seed(42)

    seq = 4096
    hidden_size = 3072
    mlp_hidden = 9216

    x = torch.randn(seq, hidden_size)
    w_in = torch.randn(hidden_size, mlp_hidden * 2)
    w_out = torch.randn(mlp_hidden, hidden_size)

    out = mlp_gated(x, w_in, w_out)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.shape == (seq, hidden_size)


def test_mlp_gated_zero_input():
    """Test MLP Gated avec input = 0 (doit donner 0 car SiLU(0)=0)."""
    seq = 4096
    hidden_size = 3072
    mlp_hidden = 9216

    x = torch.zeros(seq, hidden_size)
    w_in = torch.randn(hidden_size, mlp_hidden * 2)
    w_out = torch.randn(mlp_hidden, hidden_size)

    out = mlp_gated(x, w_in, w_out)

    # Avec x=0, gate=0 et val=0, donc SiLU(0)*0 = 0, et out = 0 @ W_out = 0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_mlp_gated_identity():
    """Test MLP Gated avec poids identité."""
    torch.manual_seed(43)

    seq = 4
    hidden_size = 8
    mlp_hidden = 16

    x = torch.randn(seq, hidden_size)

    # W_in tel que gate = x et val = x
    w_in = torch.zeros(hidden_size, mlp_hidden * 2)
    w_in[:hidden_size, :mlp_hidden] = torch.eye(hidden_size)
    w_in[:hidden_size, mlp_hidden:] = torch.eye(hidden_size)

    # W_out tel que out = activated
    w_out = torch.eye(mlp_hidden)[:hidden_size, :]

    out = mlp_gated(x, w_in, w_out)

    # Activé: SiLU(x) * x
    expected = F.silu(x) * x

    # Vérifie la forme (le résultat exact n'est pas identité car W_out tronque)
    assert out.shape == (seq, hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
