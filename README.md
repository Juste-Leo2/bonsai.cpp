# bonsai.cpp

*An ultra-lightweight, bare-metal inference engine for 1-bit diffusion models on mobile and edge devices.*

> [!WARNING]
> **🚧 Work in Progress:** This project is currently in its very early stages. It is a long-term, passion-driven project, and I am taking my time to architect and implement it properly from the ground up. The codebase is highly experimental and not yet functional. 
> 
> **🤝 Contributions are welcome!** If you're interested in mobile AI, 1-bit quantization, or squeezing every drop of performance out of Android GPUs, feel free to drop ideas, open issues, or submit PRs. I'm building this at my own pace, but I'd love to collaborate!


## The Problem
Running large diffusion models (like Flux or SD3) on mobile GPUs often leads to Out-Of-Memory (OOM) errors, Vulkan driver buffer overflows, and segmentation faults. Monolithic architectures attempt to load the Text Encoder, the Diffuser, and the VAE simultaneously, which crashes mobile devices.

## The Solution
`bonsai.cpp` implements a **strictly sequential** hybrid architecture. By executing three distinct engines one after the other, the peak RAM usage is kept drastically low, and GPU memory allocations are safely chunked.

## Pipeline Architecture

### Step 1: Text Encoder (`llama.cpp` wrapper)
Instead of reinventing the wheel, we compile only the essential elements of `llama.cpp`. 
- Loads the LLM (e.g., Qwen3 4B).
- Performs a single forward pass to extract the hidden states (embeddings).
- Instantly destroys the instance and frees the memory.

### Step 2: Custom Diffuser Engine
- **Custom GGUF:** The original safetensors are converted into a heavily stripped-down, custom `.gguf` format containing only the required DiT tensors.
- **Bare-metal Engine:** Built directly on top of the `ggml` library.
- **1-Bit Optimization:** Designed to maximize the speed of 1-bit (`q1_0`) quantized tensors (using look-up tables or specialized grouped matrix operations).
- **Vulkan Chunking:** Memory allocations are manually chunked to bypass Adreno driver limitations.

### Step 3: VAE Decoder
The final step transforms the denoised latent tensor into a PNG image.
Depending on the final implementation, this will either use a stripped-down `AutoEncoderKL` C++ class (extracted from `stable-diffusion.cpp`) or an exported ONNX/TFLite model running directly on the mobile CPU/NPU.

**License: MIT**
