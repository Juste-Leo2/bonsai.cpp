"""Test unitaire: Modulation (scale, shift, gate).

Teste l'opération de modulation utilisée dans le double stream et single stream
du diffuseur:
- x' = x * (1 + scale) + shift
- out = gate * x'
"""

import numpy as np
import pytest
import torch

from tests.conftest import PROJECT_ROOT


def apply_modulation(x, scale, shift, gate):
    """Application de la modulation.

    x: [seq, hidden_size]
    scale, shift, gate: [hidden_size]
    """
    modulated = x * (1 + scale) + shift
    out = gate * modulated
    return out


def test_modulation_basic():
    """Test modulation basique."""
    torch.manual_seed(42)

    seq = 4096
    hidden_size = 3072

    x = torch.randn(seq, hidden_size)
    scale = torch.ones(hidden_size)
    shift = torch.zeros(hidden_size)
    gate = torch.ones(hidden_size)

    out = apply_modulation(x, scale, shift, gate)

    # Avec scale=1, shift=0, gate=1:
    # modulated = x * (1 + 1) = 2x
    # out = 1 * 2x = 2x
    assert torch.allclose(out, x * 2, atol=1e-5)


def test_modulation_identity():
    """Test modulation identité (scale=0, shift=0, gate=1)."""
    torch.manual_seed(43)

    seq = 4096
    hidden_size = 3072

    x = torch.randn(seq, hidden_size)
    scale = torch.zeros(hidden_size)
    shift = torch.zeros(hidden_size)
    gate = torch.ones(hidden_size)

    out = apply_modulation(x, scale, shift, gate)

    # Avec scale=0, shift=0, gate=1:
    # modulated = x * (1 + 0) + 0 = x
    # out = 1 * x = x
    assert torch.allclose(out, x, atol=1e-5)


def test_modulation_scale_only():
    """Test modulation avec建校 seule le scale."""
    torch.manual_seed(44)

    seq = 4096
    hidden_size = 3072

    x = torch.randn(seq, hidden_size)
    scale = torch.randn(hidden_size) * 0.5
    shift = torch.zeros(hidden_size)
    gate = torch.ones(hidden_size)

    out = apply_modulation(x, scale, shift, gate)

    # modulated = x * (1 + scale) + 0 = x * (1 + scale)
    expected = x * (1 + scale)
    assert torch.allclose(out, expected, atol=1e-5)


def test_modulation_gate_zero():
    """Test modulation avec gate = 0 (bloque complètement l'information)."""
    torch.manual_seed(45)

    seq = 4096
    hidden_size = 3072

    x = torch.randn(seq, hidden_size)
    scale = torch.randn(hidden_size)
    shift = torch.randn(hidden_size)
    gate = torch.zeros(hidden_size)

    out = apply_modulation(x, scale, shift, gate)

    # Avec gate = 0, out = 0 quel que soit x
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_modulation_combined():
    """Test modulation avec scale, shift et gate non-triviaux."""
    torch.manual_seed(46)

    seq = 4096
    hidden_size = 3072

    x = torch.randn(seq, hidden_size)
    scale = torch.randn(hidden_size) * 0.3
    shift = torch.randn(hidden_size) * 0.5
    gate = torch.randn(hidden_size) * 0.7 + 1.0

    out = apply_modulation(x, scale, shift, gate)

    # Calcul manuel
    modulated = x * (1 + scale) + shift
    expected = gate * modulated

    assert torch.allclose(out, expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
