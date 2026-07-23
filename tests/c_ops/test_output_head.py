"""Test output head: norm_out + proj_out C vs HF. ~3s."""
import subprocess, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
SAFETENSORS = PROJECT_ROOT / "diffusion_pytorch_model.safetensors"
TEST_DIR = Path(__file__).resolve().parent / "_test_outputs"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def _cos(a,b): a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64); n=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/n) if n>0 else 0

def test_output_head():
    c_bin=BUILD_DIR/"test_output_head"; assert c_bin.exists()
    H,C=3072,128; it,tt=4096,512
    sd=load_file(str(SAFETENSORS))

    # inputs: random combined [txt, img], extract img tokens
    torch.manual_seed(42)
    combined=torch.randn(it+tt, H)
    final_img_hf=combined[tt:]  # img tokens only: (4096, H)

    # ── HF-style vec + modulation ──
    t=torch.tensor([1000.0]); hp=128
    fr=torch.exp(-torch.arange(hp).float()*(np.log(10000.)/hp))
    te_flip=torch.cat([torch.sin(t.unsqueeze(1)*fr.unsqueeze(0)),torch.cos(t.unsqueeze(1)*fr.unsqueeze(0))],-1)
    w1=sd["time_guidance_embed.timestep_embedder.linear_1.weight"].float()
    w2=sd["time_guidance_embed.timestep_embedder.linear_2.weight"].float()
    vec=F.linear(F.silu(F.linear(te_flip,w1)),w2)
    vec_norm=F.layer_norm(vec,(H,),eps=1e-6)

    # HF output head: AdaLayerNormContinuous(dim, dim) → LayerNorm + modulate + proj_out
    w_norm_out=sd["norm_out.linear.weight"].float()  # (2H, H)
    w_proj=sd["proj_out.weight"].float()               # (C, H)

    mod_raw=F.linear(vec_norm, w_norm_out)  # (1, 2H)
    scale=mod_raw[0,:H]; shift=mod_raw[0,H:2*H]

    hf_img_norm=F.layer_norm(final_img_hf,(H,),eps=1e-6)
    hf_mod=hf_img_norm*(1+scale.unsqueeze(0))+shift.unsqueeze(0)
    hf_out=F.linear(hf_mod,w_proj)  # (4096, C)

    print(f"hf_out: μ={hf_out.mean():.6f} σ={hf_out.std():.6f} [{hf_out.shape}]")

    # ── Save for C ──
    def save(n,a): np.asarray(a,np.float32).ravel().tofile(TEST_DIR/n)
    save("combined.bin", combined)          # (total_t, H)
    save("vec_norm.bin", vec_norm[0])       # (H,)
    save("w_norm_out.bin", w_norm_out)      # (2H, H)
    save("w_proj_out.bin", w_proj)          # (C, H)

    r=subprocess.run([str(c_bin),"--threads","8"],capture_output=True,text=True,cwd=TEST_DIR)
    assert r.returncode==0,f"C rc={r.returncode}\n{r.stderr[-500:]}"

    c_out=np.fromfile(TEST_DIR/"out.bin",np.float32).reshape(it,C)
    cos=_cos(hf_out.numpy(),c_out)
    print(f"C μ={c_out.mean():.6f} σ={c_out.std():.6f} | cos={cos:.6f}")
    assert cos>0.9999,f"cos={cos:.6f}"
    print("PASSED!")

if __name__=="__main__": test_output_head()
