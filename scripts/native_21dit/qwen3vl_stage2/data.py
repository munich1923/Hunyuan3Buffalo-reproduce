"""Same accepted UID/geometry stream with independently sampled text, no images."""
import copy
from pathlib import Path
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from pytorch_lightning import LightningDataModule
from hy3dshape.data.dit_asl import AlignedShapeLatentDataset

from .shared import positive_int, read
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


class ABOHSSDTextDataset(IterableDataset):
    """Single-sample text/geometry stream with committed, recoverable RNG state."""
    def __init__(self, rows, contract, captions, seed=20261101, caption_seed=20261027):
        super().__init__()
        if not rows:
            raise ValueError("The accepted training split is empty")
        self.rows = list(rows)
        self.data_list = [row["uid"] for row in rows]
        self.by_uid = {row["uid"]: row for row in rows}
        self.contract = copy.deepcopy(contract)
        self.seed = int(seed)
        self.uid_rng = random.Random(self.seed)
        self.point_rng = np.random.default_rng(self.seed + 1)
        self.captions = captions
        self.caption_sampler = CaptionSampler(caption_seed)
        # Only the upstream geometry method is reused, never its image setup.
        self.pc_size = 81920
        self.pc_sharpedge_size = 0
        self.sharpedge_label = True
        self.return_normal = True
        self.emitted_samples = 0
        self.consumed_samples = 0
        self.consumed_batches = 0
        self.pending = {}
        self.last_committed_uid = None
        self._committed = self._snapshot()

    def _snapshot(self):
        return {
            "uid_rng": copy.deepcopy(self.uid_rng.getstate()),
            "point_rng": copy.deepcopy(self.point_rng.bit_generator.state),
            "caption": copy.deepcopy(self.caption_sampler.state_dict()),
        }

    def decode(self, uid):
        with np.load(self.by_uid[uid]["surface"], allow_pickle=False) as archive:
            return {"random_surface": archive["random_surface"].copy(),
                    "sharpedge_surface": archive["sharp_surface"].copy()}

    def sample_uid(self, uid):
        arrays = self.decode(uid)
        surface, geo_points = AlignedShapeLatentDataset.load_surface_sdf_points(
            self, self.point_rng, arrays["random_surface"], arrays["sharpedge_surface"])
        if not bool(torch.isfinite(surface).all()):
            raise ValueError("Nonfinite surface: " + uid)
        return {"surface": surface, "geo_points": geo_points,
                **self.caption_sampler.choose(self.captions[uid])}

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Committed sampler recovery currently requires num_workers=0")
        while True:
            uid = self.data_list[self.uid_rng.randint(0, len(self.data_list) - 1)]
            sample = self.sample_uid(uid)
            self.emitted_samples += 1
            position = self.emitted_samples
            # Retain uncommitted lookahead; do not impose a fixed sample/batch cap.
            self.pending[position] = self._snapshot()
            sample.update(abo_hssd_uid=uid, abo_hssd_sample_position=position)
            yield sample

    def commit_batch(self, batch):
        value = batch["abo_hssd_sample_position"]
        positions = value.detach().cpu().reshape(-1).tolist() if torch.is_tensor(value) else list(value)
        expected = list(range(self.consumed_samples + 1, self.consumed_samples + 1 + len(positions)))
        if not positions or positions != expected or len(batch["abo_hssd_uid"]) != len(positions):
            raise RuntimeError("Noncontiguous or empty consumed sample batch")
        last = int(positions[-1])
        if last not in self.pending:
            raise RuntimeError("Missing pending sampler snapshot")
        self._committed = self.pending[last]
        self.consumed_samples = last
        self.consumed_batches += 1
        self.last_committed_uid = batch["abo_hssd_uid"][-1]
        self.pending = {key: value for key, value in self.pending.items() if key > last}

    def state_dict(self):
        return {
            "version": 2, "contract": copy.deepcopy(self.contract), "seed": self.seed,
            "consumed_samples": self.consumed_samples, "consumed_batches": self.consumed_batches,
            "last_committed_uid": self.last_committed_uid,
            **copy.deepcopy(self._committed),
        }

    def load_state_dict(self, state):
        if state["version"] != 2 or state["contract"] != self.contract or state["seed"] != self.seed:
            raise RuntimeError("Text dataset checkpoint contract differs from current data")
        self.uid_rng.setstate(state["uid_rng"])
        self.point_rng.bit_generator.state = copy.deepcopy(state["point_rng"])
        self.caption_sampler.load_state_dict(state["caption"])
        self.consumed_samples = int(state["consumed_samples"])
        self.consumed_batches = int(state["consumed_batches"])
        self.emitted_samples = self.consumed_samples
        self.last_committed_uid = state["last_committed_uid"]
        self.pending.clear()
        self._committed = self._snapshot()


class ABOHSSDTextDataModule(LightningDataModule):
    def __init__(self, paired_index_root, metadata_path, metadata_revision,
                 batch_size=1, num_workers=0, val_num_workers=0, seed=20261101,
                 caption_seed=20261027, pc_size=81920):
        super().__init__()
        self.batch_size = positive_int(batch_size, "batch_size")
        if num_workers != 0 or val_num_workers != 0:
            raise ValueError("Committed sampler recovery currently requires num_workers=0")
        if pc_size != 81920:
            raise ValueError("Keep the accepted official 81920-point sampling rule")
        self.paired_index_root = Path(paired_index_root)
        self.seed = int(seed)
        self.dataset = None
        self._restore_state = None
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
            "batch_size": self.batch_size,
        }
        self.dataset = ABOHSSDTextDataset(rows, contract, captions, self.seed, self.caption_seed)
        if self._restore_state is not None:
            self.dataset.load_state_dict(self._restore_state)
            self._restore_state = None

    def train_dataloader(self):
        self.setup("fit")
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0,
                          pin_memory=True, drop_last=True,
                          generator=torch.Generator().manual_seed(self.seed + 2))

    def commit_batch(self, batch):
        self.dataset.commit_batch(batch)

    @property
    def consumed_samples(self):
        return 0 if self.dataset is None else self.dataset.consumed_samples

    @property
    def consumed_batches(self):
        return 0 if self.dataset is None else self.dataset.consumed_batches

    def state_dict(self):
        if self.dataset is None:
            raise RuntimeError("Cannot checkpoint an uninitialized text data module")
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        if self.dataset is None:
            self._restore_state = copy.deepcopy(state_dict)
        else:
            self.dataset.load_state_dict(state_dict)
