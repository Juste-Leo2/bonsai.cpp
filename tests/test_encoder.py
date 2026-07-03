import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / norm)


def extract_hf_activations(model, tokenizer, prompt: str, layers: list[int]):
    activations = {}

    def make_hook(layer_idx):
        def hook(_mod, _inp, out):
            activations[layer_idx] = out.detach().float().cpu().numpy()
        return hook

    handles = []
    for layer in layers:
        mlp = model.layers[layer].mlp
        handles.append(mlp.register_forward_hook(make_hook(layer)))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_hf_tokens = inputs.input_ids.shape[1]
    print(f"  HF tokens ({n_hf_tokens}): {inputs.input_ids[0].tolist()}")
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return activations


def test_encoder_matches_hf(
    gguf_path: Path,
    encoder_binary: Path,
    hf_model,
    hf_tokenizer,
    test_prompt: str,
    target_layers: list[int],
):
    # ── 1. HF reference ──────────────────────────────────────────────
    hf_acts = extract_hf_activations(hf_model, hf_tokenizer, test_prompt, target_layers)

    # ── 2. Run C encoder ─────────────────────────────────────────────
    layers_str = ",".join(str(l) for l in target_layers)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [str(encoder_binary), str(gguf_path), test_prompt, "--layers", layers_str],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, \
            f"Encoder exited with code {result.returncode}\n{result.stderr}"

        # ── 3. Compare per layer ─────────────────────────────────────
        results = []
        for layer in target_layers:
            bin_path = Path(tmpdir) / f"layer_{layer}_hidden_state.bin"
            assert bin_path.exists(), \
                f"Missing output file for layer {layer}"

            c_data = np.fromfile(bin_path, dtype=np.float32)
            hf_data = hf_acts[layer]

            assert c_data.size == hf_data.size, \
                (f"Layer {layer} element count mismatch: "
                 f"C encoder={c_data.size}, HF={hf_data.size} "
                 f"(HF shape={hf_data.shape})")

            c_data_2d = c_data.reshape(hf_data.shape)
            sim = cosine_similarity(c_data_2d, hf_data)
            results.append((layer, sim))

        for layer, sim in results:
            status = "OK" if sim > 0.98 else "LOW"
            print(f"  Layer {layer}: {status}  shape={hf_acts[layer].shape}  "
                  f"cosine_sim={sim:.6f}")

        failed = [(l, s) for l, s in results if s <= 0.98]
        if failed:
            msg = "\n".join(f"  Layer {l}: cosine_sim={s:.6f}" for l, s in failed)
            pytest.fail(f"Cosine similarity below 0.98:\n{msg}")


@pytest.mark.parametrize("layer", [9, 18, 27])
def test_each_layer_extracted(
    gguf_path: Path,
    encoder_binary: Path,
    test_prompt: str,
    layer: int,
):
    """Minimal smoke-test: the encoder produces a non-empty .bin for each
    requested layer (run separately to avoid cross-layer interference)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [str(encoder_binary), str(gguf_path), test_prompt, "--layers", str(layer)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, \
            f"Encoder failed for layer {layer}:\n{result.stderr}"

        bin_path = Path(tmpdir) / f"layer_{layer}_hidden_state.bin"
        assert bin_path.exists(), f"No output for layer {layer}"

        data = np.fromfile(bin_path, dtype=np.float32)
        assert data.size > 0, f"Empty output for layer {layer}"
        print(f"  Layer {layer}: {data.size} elements extracted")
