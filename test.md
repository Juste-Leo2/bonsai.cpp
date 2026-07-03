# Test Guide

## Python Environment

```sh
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

---

## 1. Encoder

### Download the GGUF Model

Place the Qwen3-4B GGUF in the `models/` directory:

```sh
mkdir -p models
wget -O models/Qwen3-4B-UD-Q4_K_XL.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-UD-Q4_K_XL.gguf?download=true"
```

### Build

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_encoder
```

The binary is placed at `build/bonsai_encoder`.

### Run

```sh
uv run pytest tests/test_encoder.py -v
```

All arguments have sensible defaults — override with `--gguf`, `--encoder`, `--prompt`, `--layers`.

### How it works

1. **HF reference** loads `Qwen/Qwen3-4B` via `transformers.AutoModel` (bf16) and hooks `model.layers[N].mlp` to capture `ffn_out`.
2. **C encoder** runs the same prompt through llama.cpp's GGUF model and intercepts tensors named `ffn_out-N` via `cb_eval`.
3. Both outputs are compared with cosine similarity — values ≥ 0.98 are expected.

---

## 2. VAE Decoder

### Build

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_vae
```

The binary is placed at `build/bonsai_vae`.

### Run

```sh
uv run pytest tests/test_vae.py -v
```

All arguments have sensible defaults — override with `--vae-model` or `--vae-binary`:

```sh
uv run pytest tests/test_vae.py -v \
  --vae-model /path/to/flux2-vae.safetensors \
  --vae-binary /path/to/bonsai_vae
```

### How it works

1. **Python reference** (`FluxDecoder` in `tests/run_vae.py`) loads the same safetensors, generates a synthetic latent with LCG seed 42, decodes it, and produces a PNG.
2. **C binary** (`bonsai_vae`) receives the same latent `.bin` via `--latent` and produces its own PNG.
3. Both PNGs are compared with **MSE + PSNR** — a PSNR > 40 dB is expected (typical result: ~96 dB).

---

## Run all tests

```sh
uv run pytest tests/ -v
```
