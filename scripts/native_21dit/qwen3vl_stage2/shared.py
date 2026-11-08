"""Utilities owned by the text experiment; no image-baseline imports."""
import datetime
import json
import os
from pathlib import Path
import subprocess

from .text import sha

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def batch_settings(training):
    """The training YAML is the single source of batch/accumulation settings."""
    return {
        "batch_size": positive_int(training["dataset"]["params"]["batch_size"], "batch_size"),
        "accumulation": positive_int(training["training"]["update_every"], "update_every"),
    }


def load_settings(path):
    path = Path(path).resolve()
    config = read(path)
    if Path(config["root"]).resolve() != ROOT:
        raise ValueError("Settings project root differs from the current checkout")
    if config["gpu_index"] != 1 or config["gpu_uuid"] != "GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609":
        raise ValueError("Only physical GPU1 is authorized")
    positive_int(config["max_updates"], "max_updates")
    positive_int(config["checkpoint_every"], "checkpoint_every")
    if "microbatch" in config or "accumulation" in config:
        raise ValueError("Configure batch_size/update_every only in the training YAML")
    config["_settings_path"] = str(path)
    return config


def verify_upstream(config):
    upstream = ROOT / "src/Hunyuan3D-2.1"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["upstream_commit"]:
        raise RuntimeError("Upstream Hunyuan3D-2.1 commit mismatch")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Upstream Hunyuan3D-2.1 submodule is dirty")
    if sha(upstream / "hy3dshape/tools/watertight/watertight_and_sample.py") != config["watertight_script_sha256"]:
        raise RuntimeError("Official surface sampler SHA mismatch")
