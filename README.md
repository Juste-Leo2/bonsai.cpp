# bonsai.cpp

*An ultra-lightweight, bare-metal inference engine for 1-bit diffusion models on mobile and edge devices.*

> [!WARNING]
> **🚫 Project on hold**
> 
> This project is currently on hold and will most likely not be resumed, for the following reasons:
> - **Project complexity** : This project required a massive time investment to get to this stage. I had to overcome many challenges, especially recurring memory management issues.
> - **Too niche** : This is a very specific diffusion model, and I believe the software ecosystem for on-device AI on Android isn't quite mature yet — we'll have to wait for better tool accessibility on this platform.
> - **Spreading too thin** : I've taken on too many projects at once.
> 
> **🤝 Open to contributions:**
> Even though the project is on hold, you are free to fork and reuse the code. I'd be thrilled if someone picked up my work and continued mobile support! 🙂

## Pipeline

```txt
 prompt.txt     embeddings.bin      latent.bin        out.png
     │               │                  │                │
     ▼               ▼                  ▼                ▼
┌──────────┐   ┌────────────┐     ┌──────────┐     ┌─────────┐
│ Encoder  │──▶│  Diffuser  │────▶│   VAE    │────▶│  Image  │
│(llama.cpp│   │  (WebGPU)  │     │ (ggml)   │     │  (PNG)  │
│   CPU)   │   │   [1-bit]  │     │   CPU    │     │         │
└──────────┘   └────────────┘     └──────────┘     └─────────┘
```

## Status

| Component | Status | Math Verified | WebGPU | Build |
|---|---|---|---|---|
| **Encoder** | ✅ Impl. | ✅ ([test.md](test.md)) | — | `--target bonsai_encoder` |
| **Diffuser** | ✅ Impl. | ✅ CPU 69% / GPU 88% cos | ✅ RTX 4070 Ti | `--target bonsai_diffuser[_webgpu]` |
| **VAE** | ✅ Impl. | ✅ ([test.md](test.md)) | ❌ | `--target bonsai_vae` |

- **Encoder** — wraps `llama.cpp` to produce text embeddings. CPU-only.
- **Diffuser** — custom 1-bit quantized Flux diffusion engine. Cosine similarity vs PyTorch reference: **69% (CPU AVX2), 88% (WebGPU/Vulkan on RTX 4070 Ti)**.
- **VAE** — decodes latents into PNGs. Mathematically verified against PyTorch reference. WebGPU acceleration planned.

Build instructions for each target are in [build.md](build.md).

**License: MIT**
