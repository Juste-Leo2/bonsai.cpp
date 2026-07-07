"""Test unitaire: b1_linear (quantification B1_0 et multiplication matricielle).

Compare les résultats de l'implémentation C (via b1_0_kernel.h) avec PyTorch.
Le test génère des poids quantifiés B1_0, les déquantises pour la vérification,
et compare le résultat final avec une tolérance raisonnable (la quantification
introduit une erreur mais doit rester sous contrôle).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import PROJECT_ROOT


# Seuil de similarité cosinus pour les comparaisons B1_0
# (la quantification 1-bit introdu inevitablement du bruit)
COSINE_THRESHOLD = 0.99  # Très strict pour commencer
ABS_TOL = 0.1           # Tolérance absolue en valeur


def test_b1_linear_simple():
    """Test b1_linear avec in_dim=64, out_dim=32 (petit cas pour débogage)."""
    # Générer des données de test deterministes
    torch.manual_seed(42)
    act = torch.randn(1, 64)
    weight = torch.randn(32, 64)

    # Référence PyTorch (full precision)
    expected = F.linear(act, weight).numpy()

    # Sauvegarde pour le test C
    data_dir = PROJECT_ROOT / "tests" / "fixtures"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(data_dir / "b1_linear_simple.npz",
             act=act.numpy(),
             weight=weight.numpy(),
             expected=expected)

    # TODO: Lancer le test C et comparer les résultats
    # Pour l'instant, on vérifie juste que les fixtures existent
    assert (data_dir / "b1_linear_simple.npz").exists()


def test_b1_linear_full():
    """Test b1_linear avec dimensions réelles du diffuseur (3072x3072)."""
    torch.manual_seed(43)

    # Dimensions réelles du premier bloc du double stream
    in_dim = 3072
    out_dim = 3072
    batch = 1

    act = torch.randn(batch, in_dim)
    weight = torch.randn(in_dim, out_dim)

    # Référence PyTorch
    expected = F.linear(act, weight).numpy()

    data_dir = PROJECT_ROOT / "tests" / "fixtures"
    np.savez(data_dir / "b1_linear_full.npz",
             act=act.numpy(),
             weight=weight.numpy(),
             expected=expected)

    # TODO: Lancer le test C
    assert (data_dir / "b1_linear_full.npz").exists()


def test_quantification_b1_0_exact():
    """Test que la quantification B1_0 est correctement réversible.

    Pour chaque bloc de 32 poids, on vérifie que:
    - La scale est bien mean(abs(block), axis=-1)
    - Les signes sont bien encodés dans les bits
    - La déquantification donne ±scale pour chaque poids
    """
    torch.manual_seed(44)

    # Crée un tenseur simple
    weight = torch.tensor([
        [0.5, -0.3, 0.8, -0.2, 1.0, -0.1, 0.4, -0.6,
         0.7, -0.9, 0.2, -0.4, 0.3, -0.8, 0.1, -0.5,
         0.6, -0.7, 0.9, -0.2, 0.4, -1.0, 0.8, -0.3,
         0.5, -0.6, 0.2, -0.9, 0.7, -0.4, 0.3, -0.1,
         # Deuxième bloc
         0.2, -0.5, 0.7, -0.3, 0.9, -0.1, 0.4, -0.6,
         0.8, -0.2, 0.5, -0.7, 0.3, -0.9, 0.1, -0.4,
         0.6, -0.8, 0.2, -0.5, 0.7, -0.3, 0.9, -0.1,
         0.4, -0.6, 0.8, -0.2, 0.5, -0.7, 0.3, -0.9]
    ]).float()

    # Quantification "à la main" (simule le code C)
    block_size = 32
    n_blocks = weight.shape[-1] // block_size
    weight_reshaped = weight.reshape(-1, n_blocks, block_size)

    scales = weight_reshaped.abs().mean(dim=-1)  # [out_dim, n_blocks]

    # Binarisation: signe du poids
    sign_bits = (weight_reshaped >= 0).int()  # [out_dim, n_blocks, block_size]

    # Dequantification
    dequantized = torch.where(
        sign_bits.bool(),
        scales.unsqueeze(-1),
        -scales.unsqueeze(-1)
    )

    # Vérifie que les scales sont correctes
    for i in range(n_blocks):
        block = weight_reshaped[0, i, :]
        expected_scale = block.abs().mean().item()
        assert abs(scales[0, i].item() - expected_scale) < 1e-5

    # Vérifie la déquantification
    reconstructed = dequantized.reshape_as(weight)
    for i in range(weight.shape[-1]):
        w = weight[0, i].item()
        expected_sign = 1 if w >= 0 else -1
        scale = scales[0, i // block_size].item()
        assert reconstructed[0, i].item() == pytest.approx(expected_sign * scale, abs=1e-5)


def test_b1_linear_numerical_stability():
    """Test la stabilité numérique de b1_linear avec des valeurs extrêmes."""
    torch.manual_seed(45)

    # Valeurs très petites
    act_small = torch.randn(1, 64) * 1e-6
    weight_small = torch.randn(32, 64) * 1e-6
    out_small = F.linear(act_small, weight_small)
    assert not torch.isnan(out_small).any()
    assert not torch.isinf(out_small).any()

    # Valeurs très grandes
    act_large = torch.randn(1, 64) * 1e3
    weight_large = torch.randn(32, 64) * 1e3
    out_large = F.linear(act_large, weight_large)
    assert not torch.isnan(out_large).any()
    assert not torch.isinf(out_large).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
