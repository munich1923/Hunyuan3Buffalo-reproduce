"""Stage 2 paths, provenance and explicit device admission; never imports the old runner."""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "src/Hunyuan3D-2.1/hy3dshape"
SOURCE = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"


def path(value):
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path outside userdata workspace: {candidate} -> {resolved}")
    return resolved


def configure():
    sys.dont_write_bytecode = True
    if not str(ROOT).startswith("/root/userdata/"):
        raise RuntimeError("Stage 2 must execute inside the remote userdata workspace")
    destinations = {
        "HF_HOME": "huggingface", "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "HF_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "huggingface/transformers",
        "HF_DATASETS_CACHE": "huggingface/datasets", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "matplotlib",
        "PIP_CACHE_DIR": "pip", "CONDA_PKGS_DIRS": "conda",
        "__GL_SHADER_DISK_CACHE_PATH": "gl_shaders",
    }
    for key, suffix in destinations.items():
        target = path("environment/cache/stage2/" + suffix)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    temporary = path("environment/tmp/stage2")
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ["TMPDIR", "TMP", "TEMP"]:
        os.environ[key] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    sys.path.insert(0, str(UPSTREAM))


configure()


def clock():
    now = dt.datetime.now(dt.timezone.utc)
    return {"utc": now.isoformat(), "asia_shanghai": now.astimezone(
        dt.timezone(dt.timedelta(hours=8))).isoformat()}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(file):
    h = hashlib.sha256()
    with path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(file, value):
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_name(file.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, file)


def atomic_lines(file, rows):
    file = path(file)
    temp = file.with_name(file.name + f".{os.getpid()}.tmp")
    file.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, file)


def append(file, value):
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as stream:
        stream.write(canonical({**clock(), **value}) + "\n")
        stream.flush()


def read_json(file):
    return json.loads(path(file).read_text())


def read_lines(file):
    return [json.loads(line) for line in path(file).read_text().splitlines() if line.strip()]


def seed(master, *names):
    return int(fingerprint([master, *names])[:8], 16)


def config(file):
    import yaml
    cfg = yaml.safe_load(path(file).read_text())
    if cfg["source_commit"] != SOURCE:
        raise ValueError("Unexpected pinned upstream")
    actual = subprocess.check_output(["git", "-C", str(UPSTREAM.parent),
                                      "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE:
        raise RuntimeError("Upstream commit changed")
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM.parent),
                                     "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError("Pinned upstream has modifications; use project-side adapters")
    if cfg["training"]["microbatch"] != 1:
        raise ValueError("This variable-length implementation requires microbatch=1")
    for key in ["output_root"]:
        path(cfg[key])
    return cfg


def parser(description, gpu=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/stage2_nano10k.yaml")
    if gpu:
        p.add_argument("--gpu", type=int, required=True,
                       help="Physical GPU explicitly assigned; shared use requires explicit environment opt-in")
    return p


@contextlib.contextmanager
def lock(name):
    target = path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Already running: {target}") from exc
        yield


def shared_memory_limit(total_mib, free_mib, reserve_fraction=0.1):
    if not 0 < free_mib <= total_mib or not 0 < reserve_fraction < 1:
        raise ValueError("Invalid shared GPU memory budget")
    return int(free_mib * (1-reserve_fraction))


@contextlib.contextmanager
def gpu_lease(index, run=None):
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("GPU admission must happen before CUDA initialization")
    with lock("environment/stage2_gpu.lock"), lock(f"environment/gpu_{index}.lock"):
        line = subprocess.check_output([
            "nvidia-smi", f"--id={index}",
            "--query-gpu=uuid,memory.used,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip().split(",")
        uuid, used, total, free, utilization = [x.strip() for x in line]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader"], text=True)
        shared = os.environ.get("STAGE2_ALLOW_SHARED_GPU") == str(index)
        if not shared and (float(used) > 1024 or float(utilization) > 0 or any(
                x.split(",")[0].strip() == uuid for x in processes.splitlines())):
            raise RuntimeError(f"GPU {index} is occupied: {','.join(line)}; no work started")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        details = {"physical_gpu": index, "uuid": uuid, "total_mib": float(total),
                   "initial_free_mib":float(free), "initial_used_mib":float(used),
                   "initial_utilization":float(utilization), "shared":shared}
        if shared:
            limit=shared_memory_limit(float(total),float(free))
            if limit < 4096: raise RuntimeError("Insufficient shared GPU memory budget")
            import torch
            torch.cuda.set_per_process_memory_fraction(limit/float(total),device=0)
            details.update(process_memory_limit_mib=limit,reserve_fraction=0.1)
        started = time.monotonic()
        details["leased_at_monotonic"] = started
        if run: append(path(run)/"gpu_usage.jsonl",{"event":"start",**details})
        try:
            yield details
        finally:
            if run: append(path(run)/"gpu_usage.jsonl",{
                "event":"end","gpu":index,"allocated_seconds":time.monotonic()-started})


def provenance(cfg):
    return {
        **clock(), "config": cfg, "config_fingerprint": fingerprint(cfg),
        "project_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                text=True).strip(),
        "worktree_diff_sha256": fingerprint(subprocess.check_output([
            "git", "-C", str(ROOT), "diff", "HEAD"], text=True)),
        "source_commit": SOURCE, "command": sys.argv, "cwd": str(Path.cwd()),
    }


def weight_digest(file):
    cache = path("environment/cache/stage2/weight_integrity") / (
        fingerprint(str(path(file))) + ".json")
    stat = path(file).stat()
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache.exists():
        old = read_json(cache)
        if old.get("identity") == identity:
            return old["sha256"]
    value = digest(file)
    atomic_json(cache, {"identity": identity, "sha256": value, "file": str(path(file))})
    return value


def model_identity(cfg, kind):
    """Bind declared revision and local payload stat plus config/tokenizer content hashes."""
    folder = path(cfg["models"][kind])
    files = []
    for file in sorted(folder.rglob("*")):
        if not file.is_file() or file.suffix not in {
                ".json", ".yaml", ".safetensors", ".bin", ".ckpt", ".txt"}:
            continue
        path(file)
        stat = file.stat()
        files.append({"file": str(file.relative_to(folder)), "size": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns,
                      "sha256": weight_digest(file)})
    if not files:
        raise RuntimeError(f"No local {kind} weights")
    manifest_name = ("manifest-Qwen--Qwen2.5-VL-7B-Instruct.json" if kind=="vlm"
                     else "manifest-tencent--Hunyuan3D-2.1.json")
    manifest = read_json("environment/"+manifest_name)
    if manifest["sha"] != cfg["models"][kind+"_revision"]:
        raise RuntimeError("Model manifest revision mismatch")
    expected = {item["rfilename"]:item for item in manifest["siblings"]}
    for item in files:
        reference = expected.get(item["file"], {})
        checksum = reference.get("lfs",{}).get("sha256")
        if checksum and item["sha256"] != checksum:
            raise RuntimeError("Model payload checksum mismatch: "+item["file"])
    return {"revision": cfg["models"][kind + "_revision"], "files": files}


def manifest_root(cfg):
    return path(cfg["dataset"]["manifest_dir"])


def manifests(cfg, split):
    root = manifest_root(cfg)
    ready = read_json(root / "ready.json")
    if ready["dataset_contract"] != dataset_contract(cfg):
        raise RuntimeError("Manifest/config mismatch; prepare a new dataset version")
    file = root / (split + ".jsonl")
    if digest(file) != ready["split_sha256"][split]:
        raise RuntimeError("Manifest was modified after preparation")
    return read_lines(file)


def dataset_contract(cfg):
    return fingerprint({"dataset": cfg["dataset"], "seed": cfg["seed"]})


def run_path(cfg, run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a single directory name")
    return path(cfg["output_root"]) / run_id


def save_tensor(file, value):
    import torch
    file = path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_name(file.name + f".{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, file)
