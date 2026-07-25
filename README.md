# bonsai.cpp

*An ultra-lightweight, bare-metal inference engine for 1-bit diffusion models on mobile and edge devices.*

> [!WARNING]
> **🚧 Work in Progress:** This project is currently in its very early stages. It is a long-term, passion-driven project, and I am taking my time to architect and implement it properly from the ground up. The codebase is highly experimental and not yet functional. 
> 
> **🤝 Contributions are welcome!** If you're interested in mobile AI, 1-bit quantization, or squeezing every drop of performance out of Android GPUs, feel free to drop ideas, open issues, or submit PRs. I'm building this at my own pace, but I'd love to collaborate!

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
