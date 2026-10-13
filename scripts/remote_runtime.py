"""Shared experiment paths and reproducible environment configuration."""
import os,json,socket,sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def configure():
    for key in ["LIBGL_ALWAYS_SOFTWARE","MESA_LOADER_DRIVER_OVERRIDE"]:os.environ.pop(key,None)
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"]="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:os.environ.setdefault(key,"8")
    if "torch" in sys.modules:sys.modules["torch"].set_num_threads(8)
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
    proxy=network_proxy()
    if proxy:
        for key in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:os.environ[key]=proxy

def network_proxy():
    path=ROOT/"environment/network_relay.json"
    if not path.exists():return None
    config=json.loads(path.read_text())
    if not config.get("enabled"):return None
    try:
        with socket.create_connection(("127.0.0.1",19791),timeout=0.3):pass
    except OSError:return None
    return config["http_proxy"]
