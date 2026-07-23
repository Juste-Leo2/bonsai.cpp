"""Test single block: HF Flux2SingleTransformerBlock vs C impl. ~5s."""
import subprocess, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import Flux2Transformer2DModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
SAFETENSORS = PROJECT_ROOT / "diffusion_pytorch_model.safetensors"
TEST_DIR = Path(__file__).resolve().parent / "_test_outputs"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def _cos(a,b): a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64); n=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/n) if n>0 else 0
def _b1_pack(a):
    BS,od,id_=32,a.shape[0],a.shape[1]; nb=id_//BS; d=a.reshape(od,nb,BS)
    scales=np.mean(np.abs(d),-1).astype(np.float16); bits=(d>=0).astype(np.uint8)
    packed=np.packbits(bits,-1,bitorder='little'); raw=np.zeros((od,nb,6),np.uint8)
    raw[:,:,:2]=scales.view(np.uint8).reshape(od,nb,2); raw[:,:,2:6]=packed; return raw.ravel()

def _build_cos_sin_64(ids_np, axes_dim=(32,32,32,32), theta=2000):
    n_axes=4; max_half=max(a//2 for a in axes_dim)
    ft=np.zeros((n_axes,max_half),np.float32)
    for a in range(n_axes):
        for j in range(axes_dim[a]//2):
            ft[a,j]=1.0/(theta**((j*2)/axes_dim[a]))
    aa=[ft[a:a+1,:axes_dim[a]//2].T@ids_np[:,a:a+1].T for a in range(n_axes)]
    aa=np.concatenate(aa,0)
    return torch.from_numpy(np.cos(aa)), torch.from_numpy(np.sin(aa))

def test_single_block():
    c_bin=BUILD_DIR/"test_single_block"; assert c_bin.exists()
    H,C,ctx,n_heads,hd=3072,128,7680,24,128; it,tt=4096,512
    total_t=it+tt; mlp_hd=H*3
    sd=load_file(str(SAFETENSORS))

    # ── inputs: random combined [txt, img] ──
    torch.manual_seed(42)
    h_img=F.linear(torch.randn(it,C), sd["x_embedder.weight"].float())
    h_txt=F.linear(torch.randn(tt,ctx), sd["context_embedder.weight"].float())
    combined_pt=torch.cat([h_txt, h_img], dim=0)  # (total_t, H)

    # ── C-style timestep+modulation (HF-compatible: flip + LayerNorm) ──
    t=torch.tensor([1000.0]); hp=128
    fr=torch.exp(-torch.arange(hp).float()*(np.log(10000.)/hp))
    te_flip=torch.cat([torch.sin(t.unsqueeze(1)*fr.unsqueeze(0)), torch.cos(t.unsqueeze(1)*fr.unsqueeze(0))],-1)
    w1=sd["time_guidance_embed.timestep_embedder.linear_1.weight"].float()
    w2=sd["time_guidance_embed.timestep_embedder.linear_2.weight"].float()
    vec=F.linear(F.silu(F.linear(te_flip,w1)), w2)
    vec_norm=F.layer_norm(vec, (H,), eps=1e-6)
    mod_single=F.linear(vec_norm, sd["single_stream_modulation.linear.weight"].float())[0]  # (3H,)

    # ── RoPE for combined sequence ──
    img_ids_np=np.zeros((it,4),np.float32)
    for y in range(64):
        for x in range(64): img_ids_np[y*64+x,2]=y; img_ids_np[y*64+x,3]=x
    txt_ids_np=np.zeros((tt,4),np.float32)
    cos_img,sin_img=_build_cos_sin_64(img_ids_np)
    cos_txt,sin_txt=_build_cos_sin_64(txt_ids_np)
    cos_c=np.concatenate([cos_txt,cos_img],1)  # (64, total_t)
    sin_c=np.concatenate([sin_txt,sin_img],1)

    # ── HF ref: use pos_embed for RoPE, block with C-style modulation ──
    ref_model=Flux2Transformer2DModel(patch_size=1,in_channels=C,out_channels=C,num_layers=5,num_single_layers=1,attention_head_dim=hd,num_attention_heads=n_heads,joint_attention_dim=ctx,mlp_ratio=3.0,axes_dims_rope=[32,32,32,32],rope_theta=2000,guidance_embeds=False)
    block_sd={}
    for k in sd:
        if k.startswith("single_transformer_blocks.0") or k.startswith("transformer_blocks.0"):
            block_sd[k]=sd[k]
    ref_model.load_state_dict(block_sd, strict=False); ref_model=ref_model.float()
    block=ref_model.single_transformer_blocks[0]

    iid=torch.from_numpy(img_ids_np); tid=torch.from_numpy(txt_ids_np)
    ci,si=ref_model.pos_embed(iid); ct,st=ref_model.pos_embed(tid)
    concat_rope=(torch.cat([ct,ci],0), torch.cat([st,si],0))

    # ── HF ref ──
    t0=time.time()
    with torch.no_grad():
        hf_out=block(hidden_states=h_img.unsqueeze(0),
                     encoder_hidden_states=h_txt.unsqueeze(0),
                     temb_mod=mod_single.unsqueeze(0),
                     image_rotary_emb=concat_rope,
                     split_hidden_states=False)
    print(f"HF fwd: {time.time()-t0:.2f}s")
    hf_out=hf_out[0].numpy()  # (total_t, H)

    # ── Save for C ──
    def save(n,a): np.asarray(a,np.float32).ravel().tofile(TEST_DIR/n)
    save("combined.bin", combined_pt)  # (total_t, H)
    save("mod_single.bin", mod_single)  # (3H,)
    save("cos_combined.bin", cos_c)
    save("sin_combined.bin", sin_c)

    prefix=next(k[:k.index('.attn')] for k in sd if k.startswith("single_transformer_blocks.0"))
    blk0=prefix  # "single_transformer_blocks.0"
    for s, dims in [("attn.to_qkv_mlp_proj.weight", (H, 3*H+mlp_hd*2)),
                     ("attn.to_out.weight", (H+mlp_hd, H)),
                     ("attn.norm_q.weight", (hd,)),
                     ("attn.norm_k.weight", (hd,))]:
        w=sd[f"{blk0}.{s}"].float()
        if "norm" in s: save(f"w_{s}", w)
        else: _b1_pack(w.numpy()).tofile(TEST_DIR/f"w_{s}")

    # ── Run C ──
    t0=time.time()
    r=subprocess.run([str(c_bin),"--threads","8"],capture_output=True,text=True,cwd=TEST_DIR)
    print(f"C fwd: {time.time()-t0:.2f}s")
    if r.returncode!=0:
        raise RuntimeError(f"C rc={r.returncode}\n{r.stderr[-500:]}")
    c_out=np.fromfile(TEST_DIR/"combined_out.bin",np.float32).reshape(total_t,H)

    cos=_cos(hf_out,c_out)
    print(f"HF μ={hf_out.mean():.6f} σ={hf_out.std():.6f} | C μ={c_out.mean():.6f} σ={c_out.std():.6f} | cos={cos:.6f}")
    if cos<0.95: raise AssertionError(f"MISMATCH: cos={cos:.6f}")
    print("PASSED!")

if __name__=="__main__": test_single_block()
