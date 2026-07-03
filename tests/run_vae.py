"""Flux VAE decoder reference (diffusers naming) + test utilities."""

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Modules (naming matches safetensors keys exactly)
# ---------------------------------------------------------------------------

def swish(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if hasattr(self, 'conv_shortcut'):
            x = self.conv_shortcut(x)
        return x + h


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x):
        h = self.group_norm(x)
        b, c, hh, ww = h.shape
        h = h.flatten(2).transpose(1, 2)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1)
        h = self.to_out[0](h)
        h = h.transpose(1, 2).reshape(b, c, hh, ww)
        return x + h


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock(channels, channels),
            ResnetBlock(channels, channels),
        ])
        self.attentions = nn.ModuleList([Attention(channels)])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class UpsampleConv(nn.Module):
    """Wraps a Conv2d so state_dict key has .conv.* (matching diffusers)."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, has_upsampler):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels, out_channels),
            ResnetBlock(out_channels, out_channels),
            ResnetBlock(out_channels, out_channels),
        ])
        self.has_upsampler = has_upsampler
        if has_upsampler:
            self.upsamplers = nn.ModuleList([UpsampleConv(out_channels)])

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.has_upsampler:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.upsamplers[0](x)
        return x


class FluxDecoder(nn.Module):
    """Decoder matching the diffusers safetensors naming exactly.

    State dict keys:
      post_quant_conv.*
      conv_in.*
      mid_block.resnets.{0,1}.*
      mid_block.attentions.0.*
      up_blocks.{0,1,2,3}.resnets.{0,1,2}.*
      up_blocks.{0,1,2}.upsamplers.0.conv.*
      conv_norm_out.*
      conv_out.*
    """

    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(32, 32, 1)
        self.conv_in = nn.Conv2d(32, 512, 3, padding=1)
        self.mid_block = MidBlock(512)

        ch = 128  # base channels
        self.up_blocks = nn.ModuleList([
            UpBlock(512, 512, has_upsampler=True),
            UpBlock(512, 512, has_upsampler=True),
            UpBlock(512, 256, has_upsampler=True),
            UpBlock(256, 128, has_upsampler=False),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x):
        x = self.post_quant_conv(x)
        x = self.conv_in(x)
        x = self.mid_block(x)
        for ub in self.up_blocks:
            x = ub(x)
        x = self.conv_norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_decoder_weights(model_path, device='cpu'):
    """Load flux2-vae.safetensors into a FluxDecoder."""
    full_sd = load_file(model_path)
    decoder_sd = {}
    for k, v in full_sd.items():
        if k.startswith('decoder.'):
            decoder_sd[k[len('decoder.'):]] = v
        elif k.startswith('post_quant_conv.'):
            decoder_sd[k] = v

    decoder = FluxDecoder()
    decoder.load_state_dict(decoder_sd)
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


def load_bn_stats(model_path):
    """Load BatchNorm statistics [128] for latent pre-processing."""
    sd = load_file(model_path)
    return sd['bn.running_mean'], sd['bn.running_var']


# ---------------------------------------------------------------------------
# Synthetic latent generation (matches C++ LCG)
# ---------------------------------------------------------------------------

def generate_synthetic_latent(h=16, w=16, seed=42):
    """Generate a (128, h, w) latent matching bonsai_vae's LCG."""
    import warnings
    num = 128 * h * w
    s = np.uint32(seed)
    vals = np.empty(num, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(num):
            s = np.uint32(s * np.uint32(1664525) + np.uint32(1013904223))
            vals[i] = float(s & 0xFFFF) / 32768.0 - 1.0
    return vals.reshape(128, h, w)


# ---------------------------------------------------------------------------
# Pre-processing: inv_normalize + 2x2 spatial tile
# ---------------------------------------------------------------------------

def preprocess_latent(latent_tensor, running_mean, running_var, eps=1e-4):
    """Apply inv_normalize + 2x2 tile → (1, 32, 2*H, 2*W)."""
    std = torch.sqrt(running_var + eps)
    h, w = latent_tensor.shape[-2:]
    normed = latent_tensor * std.view(-1, 1, 1) + running_mean.view(-1, 1, 1)

    c, i, j = 32, 2, 2
    tiled = normed.reshape(c, i, j, h, w).permute(0, 3, 1, 4, 2).reshape(c, h * i, w * j)
    return tiled.unsqueeze(0)


# ---------------------------------------------------------------------------
# Post-processing: sigmoid → uint8 RGB PNG
# ---------------------------------------------------------------------------

def decode_to_png(decoder_output):
    """Apply sigmoid, clamp, return uint8 RGB numpy array (H, W, 3)."""
    arr = decoder_output.detach().float().cpu().numpy()
    arr = 1.0 / (1.0 + np.exp(-arr))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)  # CHW → HWC


def save_tensor_as_png(tensor, path):
    """Save (B, C, H, W) tensor as PNG."""
    from PIL import Image
    arr = decode_to_png(tensor[0])
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(model_path, latent_h=16, latent_w=16, device='cpu'):
    """Load decoder, generate latent, run decode, return (decoder_output, png_array)."""
    decoder = load_decoder_weights(model_path, device=device)
    running_mean, running_var = load_bn_stats(model_path)
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)

    latent_np = generate_synthetic_latent(latent_h, latent_w)
    latent = torch.from_numpy(latent_np).to(device)
    x = preprocess_latent(latent, running_mean, running_var)
    with torch.no_grad():
        out = decoder(x)
    png_arr = decode_to_png(out[0])
    return out, png_arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(png_path_a, png_path_b):
    """Return (MSE, PSNR) between two PNGs on [0, 1] float scale."""
    from PIL import Image
    a = np.array(Image.open(png_path_a), dtype=np.float64) / 255.0
    b = np.array(Image.open(png_path_b), dtype=np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    psnr = float('inf') if mse == 0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    return float(mse), psnr
