"""Same accepted UID/geometry stream with independently sampled text, no images."""
import copy
from pathlib import Path

import numpy as np
import torch

from scripts.native_21dit.native21_official.data import ABOHSSDDataModule, ABOHSSDSurfaceDataset
from scripts.native_21dit.native21_official.shared import read
from .text import CaptionSampler, jsonl, load_captions, sha

COORDINATES = "official_surface_npz_unit_box_no_extra_transform"


def accepted_rows(index_root, split="train", check_geometry=True):
    """Validate frozen split identities and geometry receipts without opening images."""
    root = Path(index_root)
    ready = read(root / "ready.json")
    if not ready["passed"] or ready["coordinate_rule"] != COORDINATES:
        raise ValueError("Invalid accepted ABO/HSSD index")
    for name, digest in ready["files"].items():
        if sha(root / name) != digest:
            raise ValueError("Accepted index changed: " + name)
    seen = set()
    groups = {}
    for name in ("train", "val", "test"):
        rows = jsonl(root / (name + ".jsonl"))
        if len(rows) != ready["counts"][name]:
            raise ValueError("Accepted split count changed: " + name)
        for row in rows:
            if row["uid"] in seen or row["split"] != name or row["coordinate_rule"] != COORDINATES:
                raise ValueError("UID overlap or changed split/coordinates")
            seen.add(row["uid"])
        groups[name] = rows
    if check_geometry:
        for row in groups[split]:
            surface = Path(row["surface"])
            stat = surface.stat()
            if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
                raise ValueError("Changed accepted surface: " + row["uid"])
            if sha(row["geometry_receipt"]) != row["geometry_receipt_sha256"]:
                raise ValueError("Changed geometry receipt: " + row["uid"])
    return groups[split], ready


class ABOHSSDTextDataset(ABOHSSDSurfaceDataset):
    def __init__(self, rows, contract, captions, seed=20261101, caption_seed=20261027):
        self.captions = captions
        self.caption_sampler = CaptionSampler(caption_seed)
        super().__init__(rows, contract, seed)

    def _snapshot(self):
        return {**super()._snapshot(), "caption": copy.deepcopy(self.caption_sampler.state_dict())}

    def decode(self, uid):
        with np.load(self.by_uid[uid]["surface"], allow_pickle=False) as archive:
            return {"random_surface": archive["random_surface"].copy(),
                    "sharpedge_surface": archive["sharp_surface"].copy()}

    def load_render(self, image_paths):
        raise RuntimeError("Stage2 text data must never load images")

    def sample_uid(self, uid):
        arrays = self.decode(uid)
        surface, geo_points = self.load_surface_sdf_points(
            self.point_rng, arrays["random_surface"], arrays["sharpedge_surface"])
        if not bool(torch.isfinite(surface).all()):
            raise ValueError("Nonfinite surface: " + uid)
        return {"surface": surface, "geo_points": geo_points,
                **self.caption_sampler.choose(self.captions[uid])}

    def state_dict(self):
        return {**super().state_dict(), "caption": copy.deepcopy(self._committed["caption"])}

    def load_state_dict(self, state):
        self.caption_sampler.load_state_dict(state["caption"])
        super().load_state_dict(state)


class ABOHSSDTextDataModule(ABOHSSDDataModule):
    def __init__(self, paired_index_root, metadata_path, metadata_revision,
                 batch_size=1, num_workers=0, val_num_workers=0, seed=20261101,
                 caption_seed=20261027, pc_size=81920):
        super().__init__(paired_index_root, batch_size, num_workers, val_num_workers,
                         seed, pc_size=pc_size)
        self.metadata_path = Path(metadata_path)
        self.metadata_revision = metadata_revision
        self.caption_seed = int(caption_seed)

    def setup(self, stage=None):
        if self.dataset is not None:
            return
        rows, ready = accepted_rows(self.paired_index_root)
        captions = load_captions(self.metadata_path, rows, self.metadata_revision)
        contract = {
            "dataset_kind": "abo_hssd_text", "coordinate_rule": COORDINATES,
            "paired_ready_sha256": sha(self.paired_index_root / "ready.json"),
            "train_manifest_sha256": sha(self.paired_index_root / "train.jsonl"),
            "metadata_sha256": sha(self.metadata_path), "metadata_revision": self.metadata_revision,
            "caption_seed": self.caption_seed, "caption_selection": "independent_rng_uniform_nonempty_entries",
            "accepted_count": len(rows), "upstream_commit": ready["upstream_commit"],
            "sampling": "uniform_with_replacement", "num_workers": 0,
        }
        self.dataset = ABOHSSDTextDataset(rows, contract, captions, self.seed, self.caption_seed)
        if self._restore_state is not None:
            self.dataset.load_state_dict(self._restore_state)
            self._restore_state = None
