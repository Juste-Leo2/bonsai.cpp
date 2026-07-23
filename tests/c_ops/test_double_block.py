"""Test double block: HF Flux2TransformerBlock vs C impl. ~5s."""
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
    """Same as C build_rope_cos_sin: produce cos/sin of shape (64, seq)."""
    n_axes=4; max_half=max(a//2 for a in axes_dim)
    ft=np.zeros((n_axes,max_half),np.float32)
    for a in range(n_axes):
        for j in range(axes_dim[a]//2):
            ft[a,j]=1.0/(theta**((j*2)/axes_dim[a]))
    aa=[ft[a:a+1,:axes_dim[a]//2].T@ids_np[:,a:a+1].T for a in range(n_axes)]
    aa=np.concatenate(aa,0)  # (64, seq)
    return torch.from_numpy(np.cos(aa)), torch.from_numpy(np.sin(aa))

def test_double_block():
    c_bin=BUILD_DIR/"test_double_block"; assert c_bin.exists()
    H,C,ctx,n_heads,hd=3072,128,7680,24,128; it,tt=4096,512
    sd=load_file(str(SAFETENSORS))

    # ── inputs (same as C main) ──
    torch.manual_seed(42)
    h_img_pt=F.linear(torch.randn(it,C), sd["x_embedder.weight"].float())
    h_txt_pt=F.linear(torch.randn(tt,ctx), sd["context_embedder.weight"].float())

    # ── C-style modulation (silu-based) ──
    t=torch.tensor([1000.0]); hp=128
    fr=torch.exp(-torch.arange(hp).float()*(np.log(10000.)/hp))
    te=torch.cat([torch.cos(t.unsqueeze(1)*fr.unsqueeze(0)),torch.sin(t.unsqueeze(1)*fr.unsqueeze(0))],-1)
    w1=sd["time_guidance_embed.timestep_embedder.linear_1.weight"].float()
    w2=sd["time_guidance_embed.timestep_embedder.linear_2.weight"].float()
    vs=F.silu(F.linear(F.silu(F.linear(te,w1)),w2))
    mod_i=F.linear(vs, sd["double_stream_modulation_img.linear.weight"].float())[0]
    mod_t=F.linear(vs, sd["double_stream_modulation_txt.linear.weight"].float())[0]

    # ── RoPE: C-style (64,seq) for C binary; HF concat for HF block reference ──
    img_ids_np=np.zeros((it,4),np.float32)
    for y in range(64):
        for x in range(64): img_ids_np[y*64+x,2]=y; img_ids_np[y*64+x,3]=x
    txt_ids_np=np.zeros((tt,4),np.float32)
    cos_img,sin_img=_build_cos_sin_64(img_ids_np)  # (64, 4096)
    cos_txt,sin_txt=_build_cos_sin_64(txt_ids_np)  # (64, 512)

    # HF block: need pos_embed RoPE
    iid=torch.from_numpy(img_ids_np); tid=torch.from_numpy(txt_ids_np)
    ref_model = Flux2Transformer2DModel(patch_size=1,in_channels=C,out_channels=C,num_layers=5,num_single_layers=20,attention_head_dim=hd,num_attention_heads=n_heads,joint_attention_dim=ctx,mlp_ratio=3.0,axes_dims_rope=[32,32,32,32],rope_theta=2000,guidance_embeds=False)
    block_sd={k:sd[k] for k in sd if k.startswith("transformer_blocks.0")}
    ref_model.load_state_dict(block_sd, strict=False); ref_model=ref_model.float()
    ci,si=ref_model.pos_embed(iid); ct,st=ref_model.pos_embed(tid)
    concat_rope=(torch.cat([ct,ci],0), torch.cat([st,si],0))
    block=ref_model.transformer_blocks[0]

    # ── HF reference ──
    t0=time.time()
    with torch.no_grad():
        hf_tx,hf_img=block(hidden_states=h_img_pt.unsqueeze(0),
                         encoder_hidden_states=h_txt_pt.unsqueeze(0),
                         temb_mod_img=mod_i.unsqueeze(0),
                         temb_mod_txt=mod_t.unsqueeze(0),
                         image_rotary_emb=concat_rope)
    print(f"HF fwd: {time.time()-t0:.2f}s")
    hf_img,hf_tx=hf_img[0].numpy(),hf_tx[0].numpy()

    # ── Save for C ──
    def save(n,a): np.asarray(a,np.float32).ravel().tofile(TEST_DIR/n)
    save("h_img.bin",h_img_pt); save("h_txt.bin",h_txt_pt)
    save("mod_img.bin",mod_i); save("mod_txt.bin",mod_t)
    save("cos_img.bin",cos_img); save("sin_img.bin",sin_img)
    save("cos_txt.bin",cos_txt); save("sin_txt.bin",sin_txt)
    pf="transformer_blocks.0"
    for s in ["attn.to_q.weight","attn.to_k.weight","attn.to_v.weight","attn.to_out.0.weight","attn.add_q_proj.weight","attn.add_k_proj.weight","attn.add_v_proj.weight","attn.to_add_out.weight","attn.norm_q.weight","attn.norm_k.weight","attn.norm_added_q.weight","attn.norm_added_k.weight","ff.linear_in.weight","ff.linear_out.weight","ff_context.linear_in.weight","ff_context.linear_out.weight"]:
        w=sd[f"{pf}.{s}"].float(); save(f"w_{s}",w) if "norm" in s else _b1_pack(w.numpy()).tofile(TEST_DIR/f"w_{s}")

    # ── C ──
    t0=time.time()
    r=subprocess.run([str(c_bin),"--threads","8"],capture_output=True,text=True,cwd=TEST_DIR)
    print(f"C fwd: {time.time()-t0:.2f}s")
    assert r.returncode==0,f"C rc={r.returncode}\n{r.stderr[-500:]}"

    c_i=np.fromfile(TEST_DIR/"h_img_out.bin",np.float32)
    c_t=np.fromfile(TEST_DIR/"h_txt_out.bin",np.float32)
    print(f"C sizes: img={c_i.shape} ({c_i.nbytes}B) txt={c_t.shape} ({c_t.nbytes}B)")
    c_i=c_i.reshape(it,H); c_t=c_t.reshape(tt,H)

    ci=_cos(hf_img,c_i); ct=_cos(hf_tx,c_t)
    print(f"HF img μ={hf_img.mean():.6f} σ={hf_img.std():.6f} | C img μ={c_i.mean():.6f} σ={c_i.std():.6f} | cos={ci:.6f}")
    print(f"HF txt μ={hf_tx.mean():.6f} σ={hf_tx.std():.6f} | C txt μ={c_t.mean():.6f} σ={c_t.std():.6f} | cos={ct:.6f}")
    if ci<0.95 or ct<0.95:
        print(f"REF[0:3]={hf_img[0,:3]} C[0:3]={c_i[0,:3]}")
        raise AssertionError(f"MISMATCH: img={ci:.6f} txt={ct:.6f}")
    print("PASSED!")

if __name__=="__main__": test_double_block()
