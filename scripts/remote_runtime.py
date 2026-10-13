"""Shared experiment paths and reproducible environment configuration."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def configure():
    for key, value in {
        "HF_HOME": ROOT/"models/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": ROOT/"models/cache/huggingface/hub",
        "TORCH_HOME": ROOT/"models/cache/torch",
        "XDG_CACHE_HOME": ROOT/".cache",
        "TMPDIR": ROOT/"tmp",
        "PIP_CACHE_DIR": ROOT/".cache/pip",
        "CONDA_PKGS_DIRS": ROOT/".cache/conda",
    }.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
