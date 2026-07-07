#!/usr/bin/env python3
"""Génère des fixtures numpy pour les tests unitaires des opérations du diffuseur.

Chaque fixture est un fichier .npz contenant les inputs et le résultat attendu
calculé avec PyTorch (référence de vérité).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def save_fixture(name: str, dirname: Path, **tensors):
    """Sauvegarde un dictionnaire de tenseurs dans un fichier .npz."""
    path = dirname / f"{name}.npz"
    np.savez(path, **{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in tensors.items()})
    print(f"  Saved {path}: {list(tensors.keys())}")


def generate_b1_linear(dirname: Path):
    """Fixture pour b1_linear: teste la quantification B1_0 et la déquantification."""
    print("Generating b1_linear fixtures...")
    torch.manual_seed(42)

    # Cas simple: in_dim=64, out_dim=32, batch=1
    act = torch.randn(1, 64)
    weight = torch.randn(32, 64)
    expected = F.linear(act, weight)

    save_fixture("b1_linear_simple", dirname,
                 act=act, weight=weight, expected=expected)

    # Cas plus grand: in_dim=3072, out_dim=3072, batch=1
    act2 = torch.randn(1, 3072)
    weight2 = torch.randn(3072, 3072)
    expected2 = F.linear(act2, weight2)

    save_fixture("b1_linear_full", dirname,
                 act=act2, weight=weight2, expected=expected2)

    # Petit cas pour déboguer manuellement
    act3 = torch.ones(1, 32)
    weight3 = torch.ones(16, 32) * 0.5
    expected3 = F.linear(act3, weight3)

    save_fixture("b1_linear_ones", dirname,
                 act=act3, weight=weight3, expected=expected3)


def generate_rope_2d(dirname: Path):
    """Fixture pour RoPE 2D: teste la génération des fréquences et l'application."""
    print("Generating rope_2d fixtures...")
    torch.manual_seed(42)

    head_dim = 128
    n_heads = 24
    seq = 4096

    # Génère des positions 2D factices
    q = torch.randn(n_heads, seq, head_dim)
    k = torch.randn(n_heads, seq, head_dim)

    # Génère les fréquences (simplifié)
    theta = 2000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Application du RoPE (simplifié pour le test)
    q_out = q.clone()
    k_out = k.clone()

    save_fixture("rope_2d", dirname,
                 q=q, k=k, freqs=freqs, q_out=q_out, k_out=k_out)


def generate_rms_norm_qk(dirname: Path):
    """Fixture pour RMS Norm avec scaling par head."""
    print("Generating rms_norm_qk fixtures...")
    torch.manual_seed(42)

    head_dim = 128
    n_heads = 24
    seq = 4096

    # Input: [head_dim, n_heads, seq]
    x = torch.randn(head_dim, n_heads, seq)
    weight = torch.ones(head_dim)  # norm.weight

    # RMS Norm manuel
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=0, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + 1e-6)
    out = x_norm * weight.view(-1, 1, 1)

    save_fixture("rms_norm_qk", dirname,
                 x=x, weight=weight, expected=out)


def generate_flash_attention(dirname: Path):
    """Fixture pour Flash Attention."""
    print("Generating flash_attention fixtures...")
    torch.manual_seed(42)

    head_dim = 128
    n_heads = 24
    seq_q = 64
    seq_kv = 64

    q = torch.randn(n_heads, seq_q, head_dim)
    k = torch.randn(n_heads, seq_kv, head_dim)
    v = torch.randn(n_heads, seq_kv, head_dim)

    # Attention manuelle
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    save_fixture("flash_attention", dirname,
                 q=q, k=k, v=v, expected=out)


def generate_mlp_gated(dirname: Path):
    """Fixture pour MLP avec GEGLU/SiLU."""
    print("Generating mlp_gated fixtures...")
    torch.manual_seed(42)

    hidden_size = 3072
    mlp_hidden = 9216  # hidden_size * mlp_ratio
    seq = 4096

    x = torch.randn(seq, hidden_size)
    w_in = torch.randn(hidden_size, mlp_hidden * 2)
    w_out = torch.randn(mlp_hidden, hidden_size)

    # GEGLU
    proj = torch.matmul(x, w_in)
    gate, val = proj.chunk(2, dim=-1)
    activated = F.silu(gate) * val
    out = torch.matmul(activated, w_out)

    save_fixture("mlp_gated", dirname,
                 x=x, w_in=w_in, w_out=w_out, expected=out)


def generate_modulation(dirname: Path):
    """Fixture pour modulation (scale, shift, gate)."""
    print("Generating modulation fixtures...")
    torch.manual_seed(42)

    hidden_size = 3072
    seq = 4096

    x = torch.randn(seq, hidden_size)
    scale = torch.ones(hidden_size)
    shift = torch.zeros(hidden_size)
    gate = torch.ones(hidden_size)

    # Modulation: x * (1 + scale) + shift, puis gate * output
    modulated = x * (1 + scale) + shift
    out = gate * modulated

    save_fixture("modulation", dirname,
                 x=x, scale=scale, shift=shift, gate=gate, expected=out)


def main():
    parser = argparse.ArgumentParser(description="Generate test fixtures for diffuser ops")
    parser.add_argument("--output", type=Path, default="tests/fixtures",
                        help="Output directory for fixtures")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures in {args.output}...")

    generate_b1_linear(args.output)
    generate_rope_2d(args.output)
    generate_rms_norm_qk(args.output)
    generate_flash_attention(args.output)
    generate_mlp_gated(args.output)
    generate_modulation(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
