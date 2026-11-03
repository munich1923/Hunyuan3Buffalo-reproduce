"""CPU-only check for the frozen ABO/HSSD pairing and one training sample."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.path[:0] = [str(ROOT), str(ROOT / "src/Hunyuan3D-2.1/hy3dshape")]

import torch

from scripts.native_21dit.native21_official.data import ABOHSSDSurfaceDataset
from scripts.native_21dit.native21_official.prepare_abo_hssd_index import jsonl, validate_index
from scripts.native_21dit.native21_official.shared import load_settings


def main():
    settings_path = ROOT / "configs/native21_official_abo_hssd_1b_20261103.json"
    settings = load_settings(settings_path)
    index_root = Path(settings["paired_index_root"])
    ready = validate_index(index_root, full_hash=False)
    rows_by_split = {split: jsonl(index_root / f"{split}.jsonl") for split in ("train", "val", "test")}
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    if counts != {"train": 8626, "val": 479, "test": 479}:
        raise RuntimeError(f"Unexpected split counts: {counts}")

    seen = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            uid = row["uid"]
            if uid in seen or row["split"] != split:
                raise RuntimeError(f"Duplicate UID or wrong split: {uid}")
            if Path(row["image"]).parent.name != uid:
                raise RuntimeError(f"Image does not match UID: {uid}")
            if Path(row["surface"]).name != f"{uid}_surface.npz":
                raise RuntimeError(f"Geometry does not use official surface file: {uid}")
            seen.add(uid)

    train_rows = rows_by_split["train"][:2]
    contract = {"paired_ready_sha256": "cpu-interface-check"}
    first = ABOHSSDSurfaceDataset(train_rows, contract, seed=settings["train_seed"])
    second = ABOHSSDSurfaceDataset(train_rows, contract, seed=settings["train_seed"])
    sample_a = first.sample_uid(train_rows[0]["uid"])
    sample_b = second.sample_uid(train_rows[0]["uid"])
    for name, shape in (("surface", (81920, 7)), ("image", (3, 518, 518)), ("mask", (1, 518, 518))):
        if tuple(sample_a[name].shape) != shape or not torch.isfinite(sample_a[name]).all():
            raise RuntimeError(f"Invalid {name} tensor: {tuple(sample_a[name].shape)}")
        if not torch.equal(sample_a[name], sample_b[name]):
            raise RuntimeError(f"Sampling is not deterministic for fixed seed: {name}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU interface check initialized CUDA")

    print(json.dumps({"passed": True, "counts": counts, "coordinate_rule": ready["coordinate_rule"],
                      "sample_shapes": {name: list(sample_a[name].shape) for name in ("surface", "image", "mask")},
                      "cuda_initialized": False}, indent=2))


if __name__ == "__main__":
    main()
