"""Test input projection: C h_img, h_txt vs PyTorch F.linear (x_embedder, context_embedder)."""
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_input_projection():
    """Compare h_img (x_embedder) and h_txt (context_embedder) between C and PyTorch."""
    c_binary = BUILD_DIR / "test_input_proj"
    if not c_binary.exists():
        pytest.skip(f"C binary not found at {c_binary}")

    # --- Load safetensors ---
    sd = load_file(str(SAFETENSORS))

    # x_embedder.weight: (C=128, H=3072)  -- wait, safetensors has shape (128, 3072)?
    # context_embedder.weight: (ctx_dim=7680, H=3072)
    w_img_pt = sd["x_embedder.weight"].float()   # (128, 3072)
    w_txt_pt = sd["context_embedder.weight"].float()  # (7680, 3072)

    H = 3072
    C = 128
    ctx_dim = 7680
    img_tokens = 4096
    txt_tokens = 512

    print(f"  w_img shape: {w_img_pt.shape}")
    print(f"  w_txt shape: {w_txt_pt.shape}")

    # --- Generate synthetic inputs ---
    torch.manual_seed(42)
    # act: PyTorch (img_tokens, C) → F.linear → (img_tokens, H)
    act_pt = torch.randn(img_tokens, C)
    # emb: PyTorch (txt_tokens, ctx_dim) → F.linear → (txt_tokens, H)
    emb_pt = torch.randn(txt_tokens, ctx_dim)

    # --- PyTorch reference ---
    # F.linear(x, w) computes x @ w.T
    h_img_ref = torch.nn.functional.linear(act_pt, w_img_pt)  # (4096, 3072)
    h_txt_ref = torch.nn.functional.linear(emb_pt, w_txt_pt)  # (512, 3072)
    print(f"  h_img_ref: μ={h_img_ref.mean():.6f} σ={h_img_ref.std():.6f}")
    print(f"  h_txt_ref: μ={h_txt_ref.mean():.6f} σ={h_txt_ref.std():.6f}")

    # --- Save weights as raw float32 (same layout as torch) ---
    w_img_pt.numpy().tofile(TEST_DIR / "w_img.bin")
    w_txt_pt.numpy().tofile(TEST_DIR / "w_txt.bin")

    # --- Save activations as raw float32 ---
    act_pt.numpy().tofile(TEST_DIR / "act.bin")
    emb_pt.numpy().tofile(TEST_DIR / "emb.bin")

    # --- Run C binary (output goes to CWD) ---
    result = subprocess.run(
        [str(c_binary),
         "--act", str(TEST_DIR / "act.bin"),
         "--emb", str(TEST_DIR / "emb.bin"),
         "--w-img", str(TEST_DIR / "w_img.bin"),
         "--w-txt", str(TEST_DIR / "w_txt.bin"),
         "--threads", "8"],
        capture_output=True, text=True,
        cwd=TEST_DIR,
    )
    print("C stderr:", result.stderr)
    print("C stdout:", result.stdout)
    assert result.returncode == 0, f"C binary failed: {result.stderr}"

    # --- Load C outputs ---
    # ggml h_img: ne[0]=img_tokens, ne[1]=H → flat [tok0_feat0, tok1_feat0, ..., tok0_feat1, ...]
    # PyTorch (img_tokens, H) flat: [tok0_feat0, tok0_feat1, ..., tok1_feat0, ...]
    # To match: reshape as (H, img_tokens) then .T, or reshape as (img_tokens, H, order='F')
    h_img_c_raw = np.fromfile(TEST_DIR / "h_img.bin", dtype=np.float32)
    h_txt_c_raw = np.fromfile(TEST_DIR / "h_txt.bin", dtype=np.float32)

    # ggml [feat, tok] flat → PyTorch (tok, feat) flat
    # ne[0]=H(feat), ne[1]=img_tokens(tok) → flat: feat0_tok0, feat1_tok0, ..., feat0_tok1, ...
    # This matches PyTorch (img_tokens, H) C-order flat
    h_img_c = h_img_c_raw.reshape(img_tokens, H)
    h_txt_c = h_txt_c_raw.reshape(txt_tokens, H)

    print(f"  h_img_c: shape={h_img_c.shape} μ={h_img_c.mean():.6f} σ={h_img_c.std():.6f}")
    print(f"  h_txt_c: shape={h_txt_c.shape} μ={h_txt_c.mean():.6f} σ={h_txt_c.std():.6f}")

    # --- Compare ---
    cos_img = _cosine(h_img_ref.numpy(), h_img_c)
    cos_txt = _cosine(h_txt_ref.numpy(), h_txt_c)
    print(f"  Cosine h_img: {cos_img:.8f}")
    print(f"  Cosine h_txt: {cos_txt:.8f}")

    assert cos_img > 0.99, f"h_img mismatch: cos={cos_img:.6f}"
    assert cos_txt > 0.99, f"h_txt mismatch: cos={cos_txt:.6f}"


if __name__ == "__main__":
    test_input_projection()
