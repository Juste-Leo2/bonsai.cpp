#!/usr/bin/env bash
set -euo pipefail

# ── Color helpers ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

passed=0
failed=0

section() {
    echo -e "\n${CYAN}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $*${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════════${NC}"
}

run_test() {
    local label="$1"
    shift
    echo -e "\n${YELLOW}▶ ${label}${NC}"
    echo -e "  ${CYAN}$*${NC}"
    if "$@"; then
        echo -e "  ${GREEN} ✅ PASS${NC}"
        passed=$((passed + 1))
    else
        echo -e "  ${RED} ❌ FAIL${NC}"
        failed=$((failed + 1))
    fi
}

# ── Locate project root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Activate virtual environment ───────────────────────────────────
section "Python environment"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✓ Virtualenv activated: $(which python)"
else
    echo "❌ No .venv found — run: uv venv --python 3.11 .venv"
    exit 1
fi

# Install deps (idempotent)
python -m pip install -q diffusers 2>/dev/null || true

# ── Build C test binaries ──────────────────────────────────────────
section "Build C test binaries"
cmake -B build -S . 2>&1 | tail -3
cmake --build build -j"$(nproc)" \
    --target test_b1_linear \
    --target test_rope_2d \
    --target test_input_proj \
    --target test_timestep \
    --target test_double_block \
    --target test_single_block \
    --target test_output_head \
    2>&1 | tail -5
echo -e "${GREEN}✓ Build complete${NC}"

# ── Verify safetensors ─────────────────────────────────────────────
if [ ! -f diffusion_pytorch_model.safetensors ]; then
    echo -e "${RED}❌ diffusion_pytorch_model.safetensors not found${NC}"
    exit 1
fi

# ====================================================================
# TIER 1 — Pure Python math tests (no C binary needed)
# ====================================================================
section "TIER 1 — Pure Python math tests"

run_test "b1_linear: quantification + stabilité" \
    uv run python -m pytest tests/ops/test_b1_linear.py -v -s

run_test "RoPE 2D: freqs table, cos²+sin²=1, rotation 90°" \
    uv run python -m pytest tests/ops/test_rope_2d.py -v -s

run_test "RMS Norm QK: basic, zero-input, tiny values" \
    uv run python -m pytest tests/ops/test_rms_norm_qk.py -v -s

run_test "Flash Attention: equivalence, scale, causal mask" \
    uv run python -m pytest tests/ops/test_flash_attention.py -v -s

run_test "MLP Gated: basic, zero-input, identity" \
    uv run python -m pytest tests/ops/test_mlp_gated.py -v -s

run_test "Modulation: scale/shift/gate, identity, gate=0, combined" \
    uv run python -m pytest tests/ops/test_modulation.py -v -s

# ====================================================================
# TIER 2 — C binary vs Python/HF reference tests
# ====================================================================
section "TIER 2 — C binary vs Python/HuggingFace reference"

run_test "all_ops: 7 standard + 2 custom ops (b1_linear, rope_2d via C)" \
    uv run python -m pytest tests/ops/test_all_ops.py -v -s

run_test "c_ops/b1_linear: C quantized kernel vs PyTorch fp32" \
    uv run python -m pytest tests/c_ops/test_b1_linear.py -v -s

run_test "c_ops/rope_2d: C RoPE kernel vs Python reference" \
    uv run python -m pytest tests/c_ops/test_rope_2d.py -v -s

run_test "c_ops/input_proj: x_embedder + context_embedder C vs PyTorch" \
    uv run python -m pytest tests/c_ops/test_input_proj.py -v -s

run_test "c_ops/timestep: embedding + modulation C vs HF (flip_sin_to_cos)" \
    uv run python -m pytest tests/c_ops/test_timestep.py -v -s

run_test "c_ops/double_block: full double block C vs HF Flux2TransformerBlock" \
    uv run python -m pytest tests/c_ops/test_double_block.py -v -s

run_test "c_ops/single_block: full single block C vs HF Flux2SingleTransformerBlock" \
    uv run python -m pytest tests/c_ops/test_single_block.py -v -s

run_test "c_ops/output_head: norm_out + proj_out C vs HF" \
    uv run python -m pytest tests/c_ops/test_output_head.py -v -s

# ====================================================================
# Summary
# ====================================================================
section "SUMMARY"
total=$((passed + failed))
echo -e "  Total:  ${total}"
echo -e "  Passed: ${GREEN}${passed}${NC}"
if [ "$failed" -gt 0 ]; then
    echo -e "  Failed: ${RED}${failed}${NC}"
    exit 1
else
    echo -e "  Failed: ${failed}"
fi

echo -e "\n${GREEN}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ALL TESTS PASSED ✓${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════════${NC}"
