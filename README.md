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
│(llama.cpp│   │ (ggml CPU) │     │ (ggml)   │     │  (PNG)  │
│   CPU)   │   │   [1-bit]  │     │   CPU    │     │         │
└──────────┘   └────────────┘     └──────────┘     └─────────┘
```

## Status

| Component | Status | Math Verified | Vulkan Opt. | Build |
|---|---|---|---|---|
| **Encoder** | ✅ Impl. | ✅ ([test.md](test.md)) | — | `--target bonsai_encoder` |
| **Diffuser** | ✅ Impl. | ❌ TBD | ❌ | `--target bonsai_diffuser` |
| **VAE** | ✅ Impl. | ✅ ([test.md](test.md)) | ❌ | `--target bonsai_vae` |

- **Encoder** — wraps `llama.cpp` to produce text embeddings. No Vulkan optimization planned; llama.cpp is already efficient on CPU.
- **Diffuser** — custom GGUF-based engine with 1-bit quantized DiT tensors. Implemented but not yet mathematically verified.
- **VAE** — decodes latents into PNGs. Mathematically verified against a PyTorch reference (see [test.md](test.md)). Vulkan acceleration not yet implemented.

Build instructions for each target are in [build.md](build.md).

**License: MIT**
