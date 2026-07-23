"""Test timestep embedding + modulation: C vs HF (with flip_sin_to_cos + LayerNorm)."""
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
SAFETENSORS = PROJECT_ROOT / "diffusion_pytorch_model.safetensors"
TEST_DIR = Path(__file__).resolve().parent / "_test_outputs"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def _cos(a,b): a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64); n=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/n) if n>0 else 0

def test_timestep_hf():
    c_bin = BUILD_DIR / "test_timestep"
    assert c_bin.exists(), f"Missing {c_bin}"
    sd = load_file(str(SAFETENSORS))
    H = 3072

    # ── Load weights ──
    w1 = sd["time_guidance_embed.timestep_embedder.linear_1.weight"].float()  # (H, 256)
    w2 = sd["time_guidance_embed.timestep_embedder.linear_2.weight"].float()  # (H, H)
    w_mod_img = sd["double_stream_modulation_img.linear.weight"].float()       # (6H, H)
    w_mod_txt = sd["double_stream_modulation_txt.linear.weight"].float()       # (6H, H)
    w_single = sd["single_stream_modulation.linear.weight"].float()             # (3H, H)

    # ── HF-style timestep embedding (Timesteps with flip_sin_to_cos=True) ──
    # flip: [sin0..127, cos0..127] instead of [cos0..127, sin0..127]
    t = torch.tensor([1000.0])
    half = 128
    freqs = torch.exp(-torch.arange(half).float() * (np.log(10000.) / half))
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)
    te_flip = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [sin, cos] ← FLIP!

    # te_flip → w1 → silu → w2 (= temb = vec)
    te1 = F.linear(te_flip, w1)  # (1, H)
    te_pre_silu = te1.clone()
    te1_silu = F.silu(te1)
    vec = F.linear(te1_silu, w2)  # (1, H)

    # ── HF-style modulation: SiLU(vec) → linear (Flux2Modulation.forward) ──
    vec_silu = F.silu(vec)
    mod_img = F.linear(vec_silu, w_mod_img)  # (1, 6H)
    mod_txt = F.linear(vec_silu, w_mod_txt)  # (1, 6H)
    mod_single = F.linear(vec_silu, w_single)  # (1, 3H)

    # Stats
    for name, val in [("te_flip",te_flip),("te_pre_silu",te_pre_silu),("te_silu",te1_silu),
                       ("vec",vec),("vec_silu",vec_silu),("mod_img",mod_img),("mod_txt",mod_txt)]:
        print(f"  {name}: μ={val.mean():.6f} σ={val.std():.6f} min={val.min():.6f} max={val.max():.6f}")

    # ── Save for C ──
    def save(n,a): np.asarray(a,np.float32).ravel().tofile(TEST_DIR/n)
    save("t_scaled.bin", t.numpy())
    save("w1.bin", w1)
    save("w2.bin", w2)
    save("w_mod_img.bin", w_mod_img)
    save("w_mod_txt.bin", w_mod_txt)
    save("w_single.bin", w_single)

    # ── Run C ──
    r = subprocess.run([str(c_bin),
        "--t", str(TEST_DIR/"t_scaled.bin"),
        "--w1", str(TEST_DIR/"w1.bin"),
        "--w2", str(TEST_DIR/"w2.bin"),
        "--w-mod-img", str(TEST_DIR/"w_mod_img.bin"),
        "--w-mod-txt", str(TEST_DIR/"w_mod_txt.bin"),
        "--w-single", str(TEST_DIR/"w_single.bin"),
        "--threads", "8"],
        capture_output=True, text=True, cwd=TEST_DIR)
    print("C stderr:", r.stderr[-300:].replace('\n',' | '))
    assert r.returncode == 0, f"C rc={r.returncode}"

    # ── Compare ──
    checks = {
        "te_emb": (te_flip, (1,256)),
        "te_pre_silu": (te_pre_silu, (1,H)),
        "te_post_silu": (te1_silu, (1,H)),
        "vec": (vec, (1,H)),
        "vec_silu": (vec_silu, (1,H)),
        "mod_img": (mod_img, (1,6*H)),
        "mod_txt": (mod_txt, (1,6*H)),
        "mod_single": (mod_single, (1,3*H)),
    }
    all_ok = True
    for name, (ref, shape) in checks.items():
        c_out = np.fromfile(TEST_DIR / f"{name}.bin", np.float32).reshape(shape)
        cos = _cos(ref.numpy(), c_out)
        ok = "✅" if cos > 0.999 else "❌"
        print(f"  {ok} {name}: cos={cos:.8f}")
        if cos <= 0.999: all_ok = False
    assert all_ok, "Some checks failed"
    print("ALL PASSED!")

if __name__ == "__main__":
    test_timestep_hf()
