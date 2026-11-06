"""CPU-only conditioner and recovery checks. No pretrained weights or servers."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from importlib.machinery import ModuleSpec
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src/Hunyuan3D-2.1/hy3dshape")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Avoid the upstream root __init__'s unrelated Windows mesh-plugin import.
# Load real submodules and their actual source; do not stub data/optimizer code.
shape_package = ModuleType("hy3dshape")
shape_package.__path__ = [str(ROOT / "src/Hunyuan3D-2.1/hy3dshape/hy3dshape")]
shape_package.__spec__ = ModuleSpec("hy3dshape", loader=None, is_package=True)
sys.modules["hy3dshape"] = shape_package

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
import yaml

from hy3dshape.models.diffusion.flow_matching_sit import Diffuser
from scripts.native_21dit.native21_official.data import ABOHSSDSurfaceDataset
from scripts.native_21dit.qwen3vl_stage2.conditioner import Qwen3VLTextConditioner
from scripts.native_21dit.qwen3vl_stage2.data import ABOHSSDTextDataset, accepted_rows
from scripts.native_21dit.qwen3vl_stage2.text import CaptionSampler, load_captions, messages_for, sha
from scripts.native_21dit.qwen3vl_stage2.callbacks import TextRunStateCallback, check_optimizer
from scripts.native_21dit.qwen3vl_stage2.contracts import qwen_files
from scripts.native_21dit.qwen3vl_stage2.scalars import TextUpdateScalarCallback, grad_norm

torch.set_num_threads(2)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        text = messages[0]["content"]
        ids = [1] + [ord(c) % 100 + 3 for c in text] + [2]
        if "<image>" in text:
            ids.append(120)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 4096)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        assert not use_cache
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096), image_token_id=120)
        self.model = FakeCore()
        self.register_buffer("rotary_probe", torch.tensor([0.123456789], dtype=torch.float32))
        self.model.to(dtype=torch.bfloat16)

    def generate(self, **kwargs):
        raise AssertionError("Training conditioning must not generate language")


def conditioner(**kwargs):
    return Qwen3VLTextConditioner("unused", backbone=FakeQwen(), tokenizer=FakeTokenizer(), **kwargs)


def fixture(root):
    array = np.zeros((81930, 6), dtype=np.float16)
    array[:, 0] = np.linspace(-0.9, 0.9, len(array))
    array[:, 3] = 1
    surface = root / "surface.npz"
    np.savez(surface, random_surface=array, sharp_surface=array)
    rows = [{"uid": uid, "surface": str(surface), "image": "MUST_NOT_BE_OPENED"}
            for uid in ("abo_test", "hssd_test")]
    captions = {r["uid"]: {"captions": ["", "chair", "chair", "wooden chair"],
                            "nonempty_caption_indices": [1, 2, 3]} for r in rows}
    return rows, captions


def dataset(rows, captions):
    return ABOHSSDTextDataset(rows, {"metadata_sha256": "toy"}, captions)


def commit(ds, item):
    ds.commit_batch({"abo_hssd_sample_position": [item["abo_hssd_sample_position"]],
                     "abo_hssd_uid": [item["abo_hssd_uid"]]})


class ConditionerTests(unittest.TestCase):
    def test_text_only_gradient_path_and_variable_tokens(self):
        cond = conditioner()
        self.assertFalse(cond.backbone.training)
        cond.train()
        first = cond(text=["chair"])["main"]
        second = cond(text=["a tall wooden chair"])["main"]
        self.assertEqual(first.shape[-1], 1024)
        self.assertNotEqual(first.shape[1], second.shape[1])
        second.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in cond.connector.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in cond.backbone.parameters()))
        self.assertFalse(cond.backbone.training)
        self.assertFalse(cond.vlm.training)
        self.assertTrue(cond.tokenizer.last_kwargs["add_generation_prompt"])
        self.assertEqual(cond.unconditional_embedding(1)["main"].shape, second.shape)

    def test_rejects_images_video_tokens_and_silent_truncation(self):
        cond = conditioner()
        for kwargs in ({"image": torch.zeros(1), "text": "chair"},
                       {"mask": torch.zeros(1), "text": "chair"},
                       {"pixel_values": torch.zeros(1), "text": "chair"},
                       {"text": "<image>chair"}, {"text": ["chair", "table"]}):
            with self.assertRaises(ValueError):
                cond(**kwargs)
        cond.max_length = 3
        with self.assertRaises(ValueError):
            cond(text="chair")

    def test_parent_cast_preserves_qwen_without_fp16_roundtrip(self):
        cond = conditioner()
        with torch.no_grad():
            next(cond.vlm.parameters())[0, 0] = 1e6
        original = copy.deepcopy(cond.vlm.state_dict())
        nn.Sequential(cond).half().train()
        self.assertEqual(cond.connector[0].weight.dtype, torch.float16)
        for name, value in cond.vlm.state_dict().items():
            self.assertTrue(torch.equal(original[name], value), name)
            self.assertEqual(original[name].dtype, value.dtype)
        self.assertFalse(cond.vlm.training)

    def test_prefix_can_be_disabled(self):
        cond = conditioner(task_prefix="")
        cond(text="a chair")
        self.assertEqual(cond.tokenizer.last_messages, [{"role": "user", "content": "a chair"}])
        self.assertIn("\na chair", messages_for("a chair")[0]["content"])

    def test_real_qwen_class_accepts_text_without_visual_forward(self):
        # A randomly initialized tiny architecture validates Transformers 4.57.1
        # API usage; this is not an 8B model or a pretrained-weight download.
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        cfg = Qwen3VLConfig(
            text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                         "vocab_size": 128,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 64, "deepstack_visual_indexes": []})
        vlm = Qwen3VLForConditionalGeneration(cfg).eval()
        with patch.object(vlm.model.visual, "forward", side_effect=AssertionError("vision called")):
            output = vlm.model(input_ids=torch.tensor([[1, 2, 3]]),
                               attention_mask=torch.ones(1, 3, dtype=torch.long), use_cache=False, return_dict=True)
        self.assertEqual(output.last_hidden_state.shape, (1, 3, 64))


class DataTests(unittest.TestCase):
    def test_geometry_validation_never_requires_image_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "surface.npz"
            payload.write_bytes(b"geometry-payload")
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            row = {"uid": "abo_x", "split": "train", "coordinate_rule": "official_surface_npz_unit_box_no_extra_transform",
                   "surface": str(payload), "size": payload.stat().st_size, "mtime_ns": payload.stat().st_mtime_ns,
                   "geometry_receipt": str(receipt), "geometry_receipt_sha256": sha(receipt),
                   "image": "THIS_IMAGE_DOES_NOT_EXIST"}
            for name, rows in (("train", [row]), ("val", []), ("test", [])):
                (root / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
            files = {name + ".jsonl": sha(root / (name + ".jsonl")) for name in ("train", "val", "test")}
            (root / "ready.json").write_text(json.dumps({"passed": True, "coordinate_rule": row["coordinate_rule"],
                                                       "counts": {"train": 1, "val": 0, "test": 0}, "files": files}))
            self.assertEqual(accepted_rows(root)[0], [row])
            payload.write_bytes(b"changed geometry")
            with self.assertRaises(ValueError):
                accepted_rows(root)

    def test_caption_randomness_is_independent_and_recoverable(self):
        sampler = CaptionSampler()
        record = {"captions": ["", "chair", "chair", "table"], "nonempty_caption_indices": [1, 2, 3]}
        global_state = random.getstate()
        start = sampler.state_dict()
        choices = [sampler.choose(record) for _ in range(50)]
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual({x["caption_index"] for x in choices}, {1, 2, 3})
        sampler.load_state_dict(start)
        self.assertEqual(choices, [sampler.choose(record) for _ in range(50)])

    def test_exact_metadata_identity_missing_and_duplicate(self):
        rows = [{"uid": "abo_a", "source": "ABO", "raw_sha256": "a"}]
        original = {**rows[0], "metadata_revision": "rev", "captions": ["", "chair"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            def write(records):
                path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            write([original])
            self.assertEqual(load_captions(path, rows, "rev")["abo_a"]["nonempty_caption_indices"], [1])
            for records in ([], [original, original], [{**original, "uid": "alias"}],
                            [{**original, "source": "HSSD"}], [{**original, "metadata_revision": "wrong"}]):
                write(records)
                with self.assertRaises(ValueError):
                    load_captions(path, rows, "rev")

    def test_preserves_baseline_uid_and_surface_stream_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, captions = fixture(Path(directory))
            text = dataset(rows, captions)
            baseline = ABOHSSDSurfaceDataset(rows, {"metadata_sha256": "toy"})
            baseline.load_render = lambda paths: (torch.zeros(3, 2, 2), torch.zeros(1, 2, 2))
            with patch("cv2.imread", side_effect=AssertionError("image opened")):
                for a, b in zip(iter(text), iter(baseline)):
                    self.assertNotIn("image", a)
                    self.assertNotIn("mask", a)
                    self.assertEqual(a["abo_hssd_uid"], b["abo_hssd_uid"])
                    self.assertTrue(torch.equal(a["surface"], b["surface"]))
                    commit(text, a)
                    commit(baseline, b)
                    if text.consumed_samples == 8:
                        break

    def test_restore_discards_prefetch_and_restores_caption_state(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, captions = fixture(Path(directory))
            first = dataset(rows, captions)
            stream = iter(first)
            for _ in range(8):
                commit(first, next(stream))
            saved = copy.deepcopy(first.state_dict())
            future = [next(stream) for _ in range(3)]
            # Uncommitted samples must not advance the checkpoint's RNG states.
            restored = dataset(rows, captions)
            restored.load_state_dict(saved)
            stream2 = iter(restored)
            for expected in future:
                actual = next(stream2)
                for key in ("text", "caption_index", "caption_sha256", "abo_hssd_uid", "abo_hssd_sample_position"):
                    self.assertEqual(expected[key], actual[key])
                self.assertTrue(torch.equal(expected["surface"], actual["surface"]))
                commit(restored, actual)


class ToyModel(Diffuser):
    def __init__(self):
        pl.LightningModule.__init__(self)
        self.model = nn.Linear(1024, 1)
        self.cond_stage_model = conditioner(task_prefix="")
        self.first_stage_model = nn.Linear(1, 1).requires_grad_(False)
        self.scale_by_std = False
        self.ckpt_path = None
        self.learning_rate = 1e-5
        self.optimizer_cfg = OmegaConf.create({
            "train_image_encoder": True, "image_encoder_lr": 1e-5,
            "optimizer": {"target": "torch.optim.AdamW", "params": {"weight_decay": 0.01, "eps": 1e-6, "betas": [0.9, 0.99]}},
            "scheduler": {"target": "hy3dshape.utils.trainings.lr_scheduler.LambdaWarmUpCosineFactorScheduler",
                          "params": {"warm_up_steps": 50, "f_start": 1e-6, "f_min": 0.001, "f_max": 1.0}}})
        self.samples = []

    def on_train_epoch_start(self):
        pl.seed_everything(0)

    def on_train_batch_end(self, *args, **kwargs):
        pass  # No EMA in this tiny test model.

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        context = self.cond_stage_model(text=batch["text"])["main"]
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        self.samples.append((batch["abo_hssd_uid"][0], int(batch["caption_index"][0]), noise))
        loss = (self.model(context.mean(1)) - noise).square().mean()
        self.log("train/total_loss", loss.detach(), on_step=True, on_epoch=False)
        return loss


class OptimizerTests(unittest.TestCase):
    def test_model_snapshot_requires_matching_revision_receipts(self):
        from scripts.native_21dit.qwen3vl_stage2.text import MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json",
                     "chat_template.json", "generation_config.json", "merges.txt", "vocab.json", "shard.safetensors"]
            for name in names:
                (root / name).write_text("{}")
            (root / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 4096}}))
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
            with self.assertRaises(ValueError):
                qwen_files(root)
            receipts = root / ".cache/huggingface/download"
            receipts.mkdir(parents=True)
            for name in names:
                (receipts / (name + ".metadata")).write_text(MODEL_REVISION + "\n")
            self.assertEqual(set(qwen_files(root)), set(names))
            (receipts / "shard.safetensors.metadata").write_text("wrong_revision\n")
            with self.assertRaises(ValueError):
                qwen_files(root)

    def test_official_groups_and_shared_scheduler(self):
        model = ToyModel()
        model._trainer = SimpleNamespace(max_steps=1000000000)
        optimizers, schedulers = model.configure_optimizers()
        optimizer = optimizers[0]
        check_optimizer(model, optimizer)
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.01, 0.05, 0])
        self.assertTrue(all(g["initial_lr"] == 1e-5 for g in optimizer.param_groups))
        model.cond_stage_model(text="chair")["main"].square().mean().backward()
        optimizer.step()
        schedulers[0]["scheduler"].step()
        self.assertEqual(len({g["lr"] for g in optimizer.param_groups}), 1)

    def test_norms_are_separate_and_leave_gradients_untouched(self):
        model = ToyModel()
        for p in model.model.parameters():
            p.grad = torch.ones_like(p)
        for p in model.cond_stage_model.connector.parameters():
            p.grad = torch.full_like(p, 2)
        before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        callback = TextUpdateScalarCallback(every_n_updates=500)
        callback.on_before_optimizer_step(SimpleNamespace(global_step=499), model, None)
        self.assertEqual(callback.pending_norm[1], grad_norm(model.model))
        self.assertEqual(callback.connector_norm[1], grad_norm(model.cond_stage_model.connector))
        self.assertTrue(all(torch.equal(a, b.grad) for a, b in zip(before, [p for p in model.parameters() if p.grad is not None])))

    def test_baseline_configuration_invariants(self):
        old = yaml.safe_load((ROOT / "configs/native21_official_abo_hssd_1b_20261103.yaml").read_text())
        new = yaml.safe_load((ROOT / "configs/qwen3vl_stage2_abo_hssd_20261105.yaml").read_text())
        for key in ("target",):
            self.assertEqual(new["model"][key], old["model"][key])
        a, b = new["model"]["params"], old["model"]["params"]
        for key in ("first_stage_config", "denoiser_cfg", "scheduler_cfg", "z_scale_factor", "pipeline_cfg"):
            self.assertEqual(a[key], b[key])
        for key in ("optimizer", "scheduler"):
            self.assertEqual(a["optimizer_cfg"][key], b["optimizer_cfg"][key])
        for key in old["training"]:
            if key != "output_dir":
                self.assertEqual(new["training"][key], old["training"][key])
        self.assertIsNone(a["image_processor_cfg"])


class ToyData(pl.LightningDataModule):
    def __init__(self, root):
        super().__init__()
        rows, captions = fixture(root)
        self.dataset = dataset(rows, captions)

    @property
    def consumed_samples(self):
        return self.dataset.consumed_samples

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=1, num_workers=0,
                                           generator=torch.Generator().manual_seed(20261103))

    def commit_batch(self, batch):
        self.dataset.commit_batch(batch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state):
        self.dataset.load_state_dict(state)


def resume_probe(mode, root):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    root.mkdir(parents=True, exist_ok=True)
    folder = root / mode
    folder.mkdir()
    pl.seed_everything(20261101)
    dm = ToyData(folder)
    model = ToyModel()
    callback = object.__new__(TextRunStateCallback)
    callback.c = {"gpu_uuid": "cpu-probe", "training_entry": "unmodified_official_entry"}
    callback.run = folder
    callback.pending_rng = None
    callback.restored_step = None
    callback.last_logged_step = -1
    callback.initial_step = None
    callback.contract = None
    checkpoint = ModelCheckpoint(dirpath=folder / "ckpt", filename="{step}", every_n_train_steps=1, save_top_k=-1)
    scalars = TextUpdateScalarCallback(every_n_updates=2)
    trainer = pl.Trainer(max_steps=1 if mode == "first" else 2, accelerator="cpu", devices=1,
                         accumulate_grad_batches=8, precision=32, callbacks=[callback, scalars, checkpoint],
                         logger=TensorBoardLogger(save_dir=str(folder / "log"), name="tensorboard"),
                         enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                         val_check_interval=3000, check_val_every_n_epoch=None, log_every_n_steps=500)
    resume = str(next((root / "first/ckpt").glob("*.ckpt"))) if mode == "resume" else None
    # Paths/large-file contracts are covered separately; exercise the production
    # callbacks and standard Lightning loop/checkpoint in fresh CPU processes.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "cpu-probe"}), patch(
            "scripts.native_21dit.qwen3vl_stage2.callbacks.integration_contract",
            return_value={"metadata_sha256": "toy", "probe": True}):
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
    path = max((folder / "ckpt").glob("*.ckpt"), key=lambda p: p.stat().st_mtime_ns)
    torch.save({"checkpoint": torch.load(path, weights_only=False), "samples": model.samples,
                "consumed": dm.consumed_samples, "gpu_initialized": torch.cuda.is_initialized()}, folder / "result.pt")


class RecoveryTests(unittest.TestCase):
    def test_fresh_process_resume_equals_continuous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("continuous", "first", "resume"):
                result = subprocess.run([sys.executable, "-B", __file__, "--resume-probe", mode, str(root)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            continuous = torch.load(root / "continuous/result.pt", weights_only=False)
            first = torch.load(root / "first/result.pt", weights_only=False)
            restored = torch.load(root / "resume/result.pt", weights_only=False)
            self.assertEqual(continuous["samples"], first["samples"] + restored["samples"])
            self.assertEqual(restored["consumed"], 16)
            self.assertFalse(restored["gpu_initialized"])
            def same(a, b):
                if torch.is_tensor(a):
                    self.assertTrue(torch.equal(a, b))
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        same(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        same(x, y)
                elif isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    self.assertEqual(a, b)
            for key in ("state_dict", "optimizer_states", "lr_schedulers"):
                same(continuous["checkpoint"][key], restored["checkpoint"][key])
            # ModelCheckpoint contains intentionally different output paths.
            for result in (continuous, restored):
                self.assertIn("TextUpdateScalarCallback", result["checkpoint"]["callbacks"])
            same(continuous["checkpoint"]["callbacks"]["TextUpdateScalarCallback"],
                 restored["checkpoint"]["callbacks"]["TextUpdateScalarCallback"])
            # Same random streams, including independent caption state.
            same(continuous["checkpoint"]["abo_hssd_run_state"]["rng"], restored["checkpoint"]["abo_hssd_run_state"]["rng"])
            a = continuous["checkpoint"]["ToyData"]["caption"]
            b = restored["checkpoint"]["ToyData"]["caption"]
            same(a, b)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resume-probe":
        resume_probe(sys.argv[2], Path(sys.argv[3]))
    else:
        unittest.main()
