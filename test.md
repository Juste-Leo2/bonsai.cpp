# Test Guide — Encoder

## Python Environment

```sh
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

## Download the GGUF Model

Place the Qwen3-4B GGUF in the `models/` directory:

```sh
mkdir -p models
wget -O models/Qwen3-4B-UD-Q4_K_XL.gguf \
  "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-UD-Q4_K_XL.gguf?download=true"
```

## Build the Encoder

```sh
cmake -B build
cmake --build build -j$(nproc) --target bonsai_encoder
```

The binary is placed at `build/bonsai_encoder`.

## Run Tests

```sh
uv run pytest tests/ -v
```

### What is tested

| Test | Description |
|---|---|
| `test_encoder_matches_hf` | Runs `bonsai_encoder` and HuggingFace reference on the same prompt, then compares `ffn_out` hidden states for layers 9, 18, 27 via cosine similarity (threshold: 0.98). |
| `test_each_layer_extracted[9/18/27]` | Smoke tests — verifies the encoder produces a non-empty `.bin` for each requested layer. |

### Customising the test (optional)

All arguments have sensible defaults — you only need them for custom paths or values:

```sh
# all defaults (model in models/, binary in build/)
uv run pytest tests/ -v

# fully custom
uv run pytest tests/ -v \
  --gguf /path/to/model.gguf \
  --encoder /path/to/bonsai_encoder \
  --prompt "my test prompt" \
  --layers "5,10,15"
```

### How the comparison works

1. **HF reference** loads `Qwen/Qwen3-4B` via `transformers.AutoModel` (bf16) and hooks `model.layers[N].mlp` to capture `ffn_out`.
2. **C encoder** runs the same prompt through llama.cpp's GGUF model and intercepts tensors named `ffn_out-N` via `cb_eval`.
3. Both outputs are compared with cosine similarity — values ≥ 0.98 are expected for Q4_K quantized models against the bf16 reference.
