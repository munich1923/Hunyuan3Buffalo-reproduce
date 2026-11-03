"""ABO/HSSD single-image data adapter for the official Hunyuan3D-2.1 trainer."""
import copy
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, get_worker_info
from pytorch_lightning import LightningDataModule

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "src/Hunyuan3D-2.1/hy3dshape"

import sys
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from hy3dshape.data.dit_asl import AlignedShapeLatentDataset, padding
from scripts.native_21dit.native21_official.prepare_abo_hssd_index import validate_index


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(518),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class ABOHSSDSurfaceDataset(AlignedShapeLatentDataset):
    """Checkpointable UID and surface-point sampling over a frozen paired index."""
    def __init__(self, rows, contract, seed=20261101, padding_ratio=1.15):
        super().__init__(
            data_list=[row["uid"] for row in rows],
            image_transform=_image_transform(),
            pc_size=81920,
            pc_sharpedge_size=0,
            sharpedge_label=True,
            return_normal=True,
            padding=True,
            padding_ratio_range=[padding_ratio, padding_ratio],
        )
        self.rows = list(rows)
        self.by_uid = {row["uid"]: row for row in rows}
        self.contract = copy.deepcopy(contract)
        self.seed = int(seed)
        self.uid_rng = random.Random(self.seed)
        self.point_rng = np.random.default_rng(self.seed + 1)
        self.emitted_samples = 0
        self.consumed_samples = 0
        self.pending = {}
        self.last_committed_uid = None
        self._committed = self._snapshot()

    def _snapshot(self):
        return dict(
            emitted_samples=self.emitted_samples,
            uid_rng=copy.deepcopy(self.uid_rng.getstate()),
            point_rng=copy.deepcopy(self.point_rng.bit_generator.state),
        )

    def decode(self, uid):
        row = self.by_uid[uid]
        with np.load(row["surface"], allow_pickle=False) as archive:
            return dict(
                image=[row["image"]],
                random_surface=archive["random_surface"].copy(),
                sharpedge_surface=archive["sharp_surface"].copy(),
            )

    def load_render(self, image_paths):
        image = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected readable RGB/RGBA image: {image_paths[0]}")

        if image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255
        else:
            alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        image = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (alpha[:, :, 0] * 255).astype(np.uint8)

        if self.padding:
            height, width = image.shape[:2]
            coords = np.argwhere(mask > 0.3)
            if not len(coords):
                raise ValueError("Empty alpha mask")
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            selection = (
                slice(max(x_min - 5, 0), min(x_max + 5, height)),
                slice(max(y_min - 5, 0), min(y_max + 5, width)),
            )
            image, mask = padding(
                image[selection], mask[selection], center=True,
                padding_ratio_range=self.padding_ratio_range,
            )

        image = self.image_transform(image)
        mask = np.stack((mask, mask, mask), axis=-1)
        mask = self.image_transform(mask)[:1]
        return image, mask

    def sample_uid(self, uid):
        sample = self.decode(uid)
        surface, geo_points = self.load_surface_sdf_points(
            self.point_rng, sample["random_surface"], sample["sharpedge_surface"]
        )
        image, mask = self.load_render(sample["image"])
        for name, value in (("surface", surface), ("image", image), ("mask", mask)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite {name}: {uid}")
        return dict(surface=surface, geo_points=geo_points, image=image, mask=mask)

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Exact sampler recovery requires num_workers=0")
        while True:
            uid = self.data_list[self.uid_rng.randint(0, len(self.data_list) - 1)]
            sample = self.sample_uid(uid)
            self.emitted_samples += 1
            position = self.emitted_samples
            self.pending[position] = self._snapshot()
            if len(self.pending) > 32:
                raise RuntimeError("Data callback has not committed consumed batches")
            sample.update(abo_hssd_uid=uid, abo_hssd_sample_position=position)
            yield sample

    def commit_batch(self, batch):
        value = batch["abo_hssd_sample_position"]
        positions = value.detach().cpu().reshape(-1).tolist() if torch.is_tensor(value) else list(value)
        expected = list(range(self.consumed_samples + 1, self.consumed_samples + 1 + len(positions)))
        if positions != expected:
            raise RuntimeError(f"Noncontiguous consumed positions: {positions}, expected {expected}")
        last = int(positions[-1])
        if last not in self.pending:
            raise RuntimeError(f"No pending sampler state for position {last}")
        self._committed = self.pending[last]
        self.consumed_samples = last
        self.last_committed_uid = list(batch["abo_hssd_uid"])[-1]
        self.pending = {key: value for key, value in self.pending.items() if key > last}

    def state_dict(self):
        return dict(
            version=1,
            contract=copy.deepcopy(self.contract),
            seed=self.seed,
            consumed_samples=self.consumed_samples,
            last_committed_uid=self.last_committed_uid,
            uid_rng=copy.deepcopy(self._committed["uid_rng"]),
            point_rng=copy.deepcopy(self._committed["point_rng"]),
            sampling="uniform_with_replacement",
            num_workers=0,
        )

    def load_state_dict(self, state):
        if state["version"] != 1 or state["contract"] != self.contract or state["seed"] != self.seed:
            raise RuntimeError("Dataset checkpoint contract differs from current data")
        self.uid_rng.setstate(state["uid_rng"])
        self.point_rng.bit_generator.state = copy.deepcopy(state["point_rng"])
        self.consumed_samples = int(state["consumed_samples"])
        self.emitted_samples = self.consumed_samples
        self.last_committed_uid = state.get("last_committed_uid")
        self.pending.clear()
        self._committed = self._snapshot()


class ABOHSSDDataModule(LightningDataModule):
    def __init__(self, paired_index_root, batch_size=1, num_workers=0,
                 val_num_workers=0, seed=20261101, image_size=518,
                 pc_size=81920, padding_ratio=1.15):
        super().__init__()
        if batch_size != 1 or num_workers != 0 or val_num_workers != 0:
            raise ValueError("Exact recovery requires batch_size=1 and num_workers=0")
        if image_size != 518 or pc_size != 81920 or padding_ratio != 1.15:
            raise ValueError("Data settings differ from the accepted ABO/HSSD contract")
        self.paired_index_root = Path(paired_index_root)
        self.seed = int(seed)
        self._restore_state = None
        self.dataset = None

    def setup(self, stage=None):
        if self.dataset is not None:
            return
        ready = validate_index(self.paired_index_root, full_hash=False, splits=("train",))
        train_manifest = self.paired_index_root / "train.jsonl"
        rows = _jsonl(train_manifest)
        contract = dict(
            dataset_kind="abo_hssd",
            paired_ready_sha256=_sha(self.paired_index_root / "ready.json"),
            train_manifest_sha256=_sha(train_manifest),
            accepted_count=len(rows),
            coordinate_rule=ready["coordinate_rule"],
            upstream_commit=ready["upstream_commit"],
            image_preprocessing="RGB/RGBA white composite, padding1.15, resize518",
            sampling="uniform_with_replacement",
            num_workers=0,
        )
        self.dataset = ABOHSSDSurfaceDataset(rows, contract, self.seed)
        if self._restore_state is not None:
            self.dataset.load_state_dict(self._restore_state)
            self._restore_state = None

    def train_dataloader(self):
        self.setup("fit")
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed + 2),
        )

    def commit_batch(self, batch):
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized")
        self.dataset.commit_batch(batch)

    @property
    def consumed_samples(self):
        return 0 if self.dataset is None else self.dataset.consumed_samples

    def state_dict(self):
        if self.dataset is None:
            raise RuntimeError("Cannot checkpoint an uninitialized data module")
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        if self.dataset is None:
            self._restore_state = copy.deepcopy(state_dict)
        else:
            self.dataset.load_state_dict(state_dict)
