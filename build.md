# Build Guide

## Prerequisites

- CMake >= 3.16
- C++17 compatible compiler (gcc, clang, etc.)

## Clone

```sh
git clone --recursive https://github.com/Juste-Leo2/bonsai.cpp
cd bonsai.cpp
```

If you already cloned without `--recursive`, run:

```sh
git submodule update --init --recursive
```

## Build All Targets

```sh
cmake -B build
cmake --build build -j$(nproc)
```

## Build Individual Targets

### bonsai_encoder

Encodes prompt texts into T5 embeddings using llama.cpp's embedding model.

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_encoder
```

### bonsai_diffuser

Runs the quantized Flux diffusion engine (B1_0, step 2 of the pipeline).

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_diffuser
```

### bonsai_vae

Decodes latent tensors into images using the Flux VAE (step 3 of the pipeline).

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_vae
```

## Output

All binaries are placed in `build/bin/`.
