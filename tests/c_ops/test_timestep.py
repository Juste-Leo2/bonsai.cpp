"""Test timestep embedding (`vec`) + modulations (double_mod_img, double_mod_txt) vs HF."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
SAFETENSORS = PROJECT_ROOT / "diffusion_pytorch_model.safetensors"
TEST_DIR = Path(__file__).resolve().parent / "_test_outputs"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else 0.0


def test_timestep_embedding():
    """Test `vec` (output of time_in_w2 + bias → no silu) and `vec_silu` (vec → silu).

    Also test double_mod_img and double_mod_txt: vec_silu → linear.
    """
    c_binary = BUILD_DIR / "test_timestep"
    if not c_binary.exists():
        pytest.skip(f"C binary not found at {c_binary}")

    sd = load_file(str(SAFETENSORS))

    H = 3072

    w1 = sd["time_guidance_embed.timestep_embedder.linear_1.weight"].float()  # (H, 256)
    b1 = sd.get("time_guidance_embed.timestep_embedder.linear_1.bias")
    if b1 is not None:
        b1 = b1.float()
    w2 = sd["time_guidance_embed.timestep_embedder.linear_2.weight"].float()  # (H, H)
    b2 = sd.get("time_guidance_embed.timestep_embedder.linear_2.bias")
    if b2 is not None:
        b2 = b2.float()

    w_mod_img = sd["double_stream_modulation_img.linear.weight"].float()  # (6*H, H)
    w_mod_txt = sd["double_stream_modulation_txt.linear.weight"].float()  # (6*H, H)

    # Also load single mod for later use
    w_single_mod = sd["single_stream_modulation.linear.weight"].float()  # (3*H, H)

    print(f"  w1: {w1.shape}  b1: {b1.shape if b1 is not None else 'None'}")
    print(f"  w2: {w2.shape}  b2: {b2.shape if b2 is not None else 'None'}")
    print(f"  w_mod_img: {w_mod_img.shape}  w_mod_txt: {w_mod_txt.shape}")

    # ==== PyTorch reference (mirrors ggml_timestep_embedding) ====
    t = torch.tensor([1000.0])  # t_scaled = t * 1000
    max_period = 10000
    half = 128
    # ggml formula: freq[j] = exp(-log(max_period) * j / half) for j=0..half-1
    freqs = torch.exp(-torch.arange(0, half).float() * (np.log(max_period) / half))
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)
    te_emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (1, 256)

    # te_emb → w1 + b1 → silu
    te = torch.nn.functional.linear(te_emb, w1, b1)  # (1, H)
    te_pre_silu = te.clone()
    te = torch.nn.functional.silu(te)  # (1, H)
    te_post_silu = te.clone()

    # te → w2 + b2 = vec (no final silu here, silu is applied later to vec_silu)
    vec = torch.nn.functional.linear(te, w2, b2)  # (1, H)

    # vec_silu = silu(vec)
    vec_silu = torch.nn.functional.silu(vec)  # (1, H)

    # modulations
    mod_img = torch.nn.functional.linear(vec_silu, w_mod_img)  # (1, 6*H)
    mod_txt = torch.nn.functional.linear(vec_silu, w_mod_txt)  # (1, 6*H)

    print(f"  te_emb: μ={te_emb.mean():.6f} σ={te_emb.std():.6f}")
    print(f"  te_pre_silu: μ={te_pre_silu.mean():.6f} σ={te_pre_silu.std():.6f}")
    print(f"  te_post_silu: μ={te_post_silu.mean():.6f} σ={te_post_silu.std():.6f}")
    print(f"  vec: μ={vec.mean():.6f} σ={vec.std():.6f}")
    print(f"  vec_silu: μ={vec_silu.mean():.6f} σ={vec_silu.std():.6f}")
    print(f"  mod_img: μ={mod_img.mean():.6f} σ={mod_img.std():.6f}")
    print(f"  mod_txt: μ={mod_txt.mean():.6f} σ={mod_txt.std():.6f}")

    # Save inputs + weights
    t_scaled = t.numpy().astype(np.float32).ravel()  # [1.0]

    t_scaled.tofile(TEST_DIR / "t_scaled.bin")
    w1.numpy().tofile(TEST_DIR / "w1.bin")
    if b1 is not None:
        b1.numpy().tofile(TEST_DIR / "b1.bin")
    w2.numpy().tofile(TEST_DIR / "w2.bin")
    if b2 is not None:
        b2.numpy().tofile(TEST_DIR / "b2.bin")
    w_mod_img.numpy().tofile(TEST_DIR / "w_mod_img.bin")
    w_mod_txt.numpy().tofile(TEST_DIR / "w_mod_txt.bin")

    # Run C
    result = subprocess.run(
        [str(c_binary),
         "--t", str(TEST_DIR / "t_scaled.bin"),
         "--w1", str(TEST_DIR / "w1.bin"),
         "--w2", str(TEST_DIR / "w2.bin"),
         "--w-mod-img", str(TEST_DIR / "w_mod_img.bin"),
         "--w-mod-txt", str(TEST_DIR / "w_mod_txt.bin"),
         "--threads", "8"],
        capture_output=True, text=True,
        cwd=TEST_DIR,
    )
    print("C stderr:", result.stderr)
    assert result.returncode == 0, f"C binary failed: {result.stderr}"

    # Load and compare
    checks = {
        "te_emb": (te_emb, (1, 256)),
        "te_pre_silu": (te_pre_silu, (1, H)),
        "te_post_silu": (te_post_silu, (1, H)),
        "vec": (vec, (1, H)),
        "vec_silu": (vec_silu, (1, H)),
        "mod_img": (mod_img, (1, 6 * H)),
        "mod_txt": (mod_txt, (1, 6 * H)),
    }

    for name, (ref, shape) in checks.items():
        c_out = np.fromfile(TEST_DIR / f"{name}.bin", dtype=np.float32)
        c_out = c_out.reshape(shape)
        cos = _cosine(ref.numpy(), c_out)
        print(f"  {name}: cos={cos:.8f}  μ_c={c_out.mean():.6f} σ_c={c_out.std():.6f}")
        assert cos > 0.999, f"{name} mismatch: cos={cos:.6f}"

    print("All timestep/modulation tests passed!")


if __name__ == "__main__":
    test_timestep_embedding()
