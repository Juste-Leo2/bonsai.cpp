"""HF Flux2Transformer reference for 1-step verification vs bonsai_diffuser B1_0."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2Transformer2DModel
from safetensors.torch import load_file


@dataclass
class DiffuserParams:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: list = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    img_tokens: int = 4096
    txt_tokens: int = 512


def load_hf_model(safetensors_path, params: DiffuserParams, device="cpu"):
    model = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=params.in_channels,
        out_channels=params.in_channels,
        num_layers=params.depth,
        num_single_layers=params.depth_single_blocks,
        attention_head_dim=params.head_dim,
        num_attention_heads=params.num_heads,
        joint_attention_dim=params.context_in_dim,
        mlp_ratio=params.mlp_ratio,
        axes_dims_rope=params.axes_dim,
        rope_theta=params.theta,
        guidance_embeds=False,
    )

    sd = load_file(safetensors_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB "
              f"(of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    return model


def generate_synthetic_input(params: DiffuserParams, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, params.img_tokens, params.in_channels, generator=gen)
    embeddings = torch.randn(1, params.txt_tokens, params.context_in_dim, generator=gen)
    timestep = torch.tensor([1.0])

    H = W = int(params.img_tokens ** 0.5)
    img_ids = torch.zeros(1, params.img_tokens, 4)
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            img_ids[0, idx, 2] = y
            img_ids[0, idx, 3] = x

    txt_ids = torch.zeros(1, params.txt_tokens, 4)

    return dict(latents=latents, embeddings=embeddings, timestep=timestep,
                img_ids=img_ids, txt_ids=txt_ids)


@torch.no_grad()
def forward_one_step(model, inputs, device="cpu"):
    inp = {k: v.to(device, dtype=torch.bfloat16) for k, v in inputs.items()}
    out = model(
        hidden_states=inp["latents"],
        timestep=inp["timestep"],
        guidance=None,
        encoder_hidden_states=inp["embeddings"],
        txt_ids=inp["txt_ids"],
        img_ids=inp["img_ids"],
        return_dict=False,
    )[0]
    if device == "cuda":
        torch.cuda.synchronize()
    return out.float()  # (1, img_tokens, 128)


def noise_pred_c_format(model, inputs, device="cpu"):
    out = forward_one_step(model, inputs, device)
    return out.squeeze(0).t().contiguous()  # (128, img_tokens)


def save_bin(tensor, path):
    arr = tensor.cpu().numpy() if torch.is_tensor(tensor) else tensor
    arr.astype(np.float32).tofile(path)


def generate_and_save(model_path, params: DiffuserParams, output_dir, device="cpu", seed=42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading HF model...")
    model = load_hf_model(model_path, params, device)

    print("Generating synthetic inputs...")
    inputs = generate_synthetic_input(params, seed)

    lat = inputs["latents"][0].t().contiguous().float()
    save_bin(lat, out / "x_input.bin")
    print(f"  x_input.bin: {list(lat.shape)}")

    emb = inputs["embeddings"][0].t().contiguous().float()
    save_bin(emb, out / "embeddings.bin")
    print(f"  embeddings.bin: {list(emb.shape)}")

    print("Running HF forward...")
    np_ref = noise_pred_c_format(model, inputs, device)
    save_bin(np_ref, out / "noise_pred_ref.bin")
    print(f"  noise_pred_ref.bin: {list(np_ref.shape)}")

    return inputs, np_ref