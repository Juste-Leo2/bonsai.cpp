import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser):
    parser.addoption("--gguf", action="store", default=None,
                     help="Path to the Qwen3 GGUF model")
    parser.addoption("--encoder", action="store", default=None,
                     help="Path to the bonsai_encoder binary")
    parser.addoption("--prompt", action="store", default="a magnificient landscape",
                     help="Test prompt")
    parser.addoption("--layers", action="store", default="9,18,27",
                     help="Comma-separated layer indices to extract")
    parser.addoption("--vae-model", action="store", default=None,
                     help="Path to the flux2-vae.safetensors model")
    parser.addoption("--vae-binary", action="store", default=None,
                     help="Path to the bonsai_vae binary")
    parser.addoption("--diffuser-model", action="store", default=None,
                     help="Path to the full-precision safetensors (reference)")
    parser.addoption("--diffuser-gguf", action="store", default=None,
                     help="Path to the B1_0 quantized GGUF")
    parser.addoption("--backend", action="store", default="cpu", choices=["cpu", "gpu"],
                     help="Backend: cpu or gpu")
    parser.addoption("--diffuser-binary", action="store", default=None,
                     help="Path to the bonsai_diffuser binary")
    parser.addoption("--diffuser-webgpu-binary", action="store", default=None,
                     help="Path to the bonsai_diffuser_webgpu binary")


@pytest.fixture
def gguf_path(request) -> Path:
    path = request.config.getoption("--gguf")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "models" / "Qwen3-4B-UD-Q4_K_XL.gguf"
    if default.exists():
        return default
    pytest.skip(f"GGUF model not found at {default}. "
                f"Provide a path with --gguf <path>")


@pytest.fixture
def encoder_binary(request) -> Path:
    path = request.config.getoption("--encoder")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "build" / "bonsai_encoder"
    if default.exists():
        return default
    pytest.skip(f"Encoder binary not found at {default}. "
                f"Build it first or provide a path with --encoder <path>")


@pytest.fixture
def vae_model(request) -> Path:
    path = request.config.getoption("--vae-model")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "models" / "flux2-vae.safetensors"
    if default.exists():
        return default
    pytest.skip(f"VAE model not found at {default}. "
                f"Provide a path with --vae-model <path>")


@pytest.fixture
def vae_binary(request) -> Path:
    path = request.config.getoption("--vae-binary")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "build" / "bonsai_vae"
    if default.exists():
        return default
    pytest.skip(f"VAE binary not found at {default}. "
                f"Build it first or provide a path with --vae-binary <path>")


@pytest.fixture
def diffuser_model(request) -> Path:
    path = request.config.getoption("--diffuser-model")
    if path:
        return Path(path).resolve()
    # The full-precision safetensors at project root
    default = PROJECT_ROOT / "diffusion_pytorch_model.safetensors"
    if default.exists():
        return default
    pytest.skip(f"Diffuser reference model not found at {default}. "
                f"Provide a path with --diffuser-model <path>")


@pytest.fixture
def diffuser_gguf(request) -> Path:
    path = request.config.getoption("--diffuser-gguf")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "models" / "flux2_4b_1bit.gguf"
    if default.exists():
        return default
    pytest.skip(f"Diffuser GGUF not found at {default}. "
                f"Provide a path with --diffuser-gguf <path>")


@pytest.fixture
def diffuser_packed(request) -> Path:
    path = request.config.getoption("--diffuser-gguf")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "models" / "flux2_4b_1bit.safetensors"
    if default.exists():
        return default
    pytest.skip(f"Diffuser packed safetensors not found at {default}. "
                f"Run convert_to_b1_0_packed.py first.")


@pytest.fixture
def diffuser_binary(request) -> Path:
    path = request.config.getoption("--diffuser-binary")
    if path:
        return Path(path).resolve()
    default = PROJECT_ROOT / "build" / "bonsai_diffuser"
    if default.exists():
        return default
    # Windows fallback
    default_win = PROJECT_ROOT / "build" / "Release" / "bonsai_diffuser.exe"
    if default_win.exists():
        return default_win
    pytest.skip(f"Diffuser binary not found at {default}. "
                f"Build it first or provide a path with --diffuser-binary <path>")


@pytest.fixture
def diffuser_webgpu_binary(request) -> Path:
    path = request.config.getoption("--diffuser-webgpu-binary")
    if path:
        return Path(path).resolve()
    candidates = [
        PROJECT_ROOT / "build" / "Release" / "bonsai_diffuser_webgpu.exe",
        PROJECT_ROOT / "build" / "bonsai_diffuser_webgpu",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@pytest.fixture
def backend(request) -> str:
    return request.config.getoption("--backend")


@pytest.fixture
def test_prompt(request) -> str:
    return request.config.getoption("--prompt")


@pytest.fixture
def target_layers(request) -> list[int]:
    raw = request.config.getoption("--layers")
    return [int(x.strip()) for x in raw.split(",")]


@pytest.fixture(scope="session")
def hf_model_name() -> str:
    return "Qwen/Qwen3-4B"


@pytest.fixture(scope="session")
def hf_model(hf_model_name):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        hf_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def hf_tokenizer(hf_model_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    return tokenizer
