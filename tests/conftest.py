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
