"""CPU-only conditioner and recovery checks. No pretrained weights or servers."""
import copy
from contextlib import nullcontext
import importlib.abc
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

if "--isolation-probe" in sys.argv:
    class BlockBaseline(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("scripts.native_21dit.native21_official"):
                raise ImportError("Image experiment import is forbidden")
    sys.meta_path.insert(0, BlockBaseline())

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
import yaml

from hy3dshape.models.diffusion.flow_matching_sit import Diffuser
from hy3dshape.data.dit_asl import AlignedShapeLatentDataset
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
from scripts.native_21dit.qwen3vl_stage2.conditioner import Qwen3VLTextConditioner
from scripts.native_21dit.qwen3vl_stage2.data import ABOHSSDTextDataset, ABOHSSDTextDataModule, accepted_rows
from scripts.native_21dit.qwen3vl_stage2.denoiser import MaskedHunYuanDiTPlain
from scripts.native_21dit.qwen3vl_stage2.text import CaptionSampler, load_captions, messages_for, sha
from scripts.native_21dit.qwen3vl_stage2.callbacks import STATE_KEY, TextRunStateCallback, check_optimizer
from scripts.native_21dit.qwen3vl_stage2.contracts import qwen_files, source_hashes
from scripts.native_21dit.qwen3vl_stage2.shared import batch_settings
from scripts.native_21dit.qwen3vl_stage2.scalars import TextUpdateScalarCallback, grad_norm

torch.set_num_threads(2)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        sequences = []
        for conversation in messages:
            text = conversation[0]["content"]
            ids = [1] + [ord(c) % 100 + 3 for c in text] + [2]
            if "<image>" in text:
                ids.append(120)
            sequences.append(ids)
        length = max(map(len, sequences))
        return {"input_ids": torch.tensor([[0] * (length - len(ids)) + ids for ids in sequences]),
                "attention_mask": torch.tensor([[0] * (length - len(ids)) + [1] * len(ids) for ids in sequences])}


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
                       {"text": "<image>chair"}, {"text": []}, {"text": ["chair", 1]},
                       {"text": ["chair", " "]}):
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
        self.assertEqual(cond.tokenizer.last_messages, [[{"role": "user", "content": "a chair"}]])
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
            ids = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]])
            valid = ids != 0
            batched = vlm.model(input_ids=ids, attention_mask=valid.long(), use_cache=False,
                                return_dict=True).last_hidden_state
            for i in range(2):
                single = vlm.model(input_ids=ids[i:i+1, valid[i]],
                                   attention_mask=torch.ones(1, int(valid[i].sum()), dtype=torch.long),
                                   use_cache=False, return_dict=True).last_hidden_state
                torch.testing.assert_close(batched[i, valid[i]], single[0], atol=1e-5, rtol=1e-4)
        self.assertEqual(output.last_hidden_state.shape, (1, 3, 64))

    def test_real_tokenizer_chat_template_batches_without_truncation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "chair": 2,
                                       "a": 3, "wooden": 4, "table": 5}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                            model_input_names=["input_ids", "attention_mask"])
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
        cond = Qwen3VLTextConditioner("unused", task_prefix="", backbone=FakeQwen(), tokenizer=tokenizer)
        inputs = cond.tokenize_text(["chair", "a wooden table"])
        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(int(inputs["attention_mask"][0, 0]), 0)
        result = cond(text=["chair", "a wooden table"])
        self.assertTrue(torch.equal(result["main_mask"], inputs["attention_mask"].bool()))

    def test_batched_text_masks_and_unconditional_embeddings(self):
        cond = conditioner(task_prefix="")
        texts = ["chair", "a tall wooden chair", "table"]
        result = cond(text=texts)
        self.assertEqual(result["main"].shape[:2], result["main_mask"].shape)
        self.assertEqual(result["main_mask"].dtype, torch.bool)
        self.assertEqual(result["main_mask"].sum(1).tolist(), [len(t) + 2 for t in texts])
        self.assertTrue(torch.equal(result["main"][~result["main_mask"]], torch.zeros_like(result["main"][~result["main_mask"]])))
        unconditional = cond.unconditional_embedding(3)
        self.assertTrue(torch.equal(unconditional["main_mask"], result["main_mask"]))
        self.assertEqual(unconditional["main"].count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            cond.unconditional_embedding(2)
        for i, text in enumerate(texts):
            single = cond(text=text)
            torch.testing.assert_close(result["main"][i, result["main_mask"][i]], single["main"][0])


def tiny_dit(cls=MaskedHunYuanDiTPlain):
    return cls(input_size=4, in_channels=4, hidden_size=64, context_dim=1024, depth=4,
               num_heads=4, num_moe_layers=0, use_attention_pooling=False, qk_norm=True)


class MaskedDiTTests(unittest.TestCase):
    def setUp(self):
        # The official self-attention forces CUDA-only backend flags. On CPU,
        # exercise the same attention math with default backend selection.
        guard = patch("torch.backends.cuda.sdp_kernel", new=lambda **kwargs: nullcontext())
        guard.start()
        self.addCleanup(guard.stop)

    def test_same_parameter_keys_initialization_and_official_loading(self):
        torch.manual_seed(101)
        original = tiny_dit(HunYuanDiTPlain)
        rng = torch.get_rng_state()
        torch.manual_seed(101)
        masked = tiny_dit()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original.state_dict().keys(), masked.state_dict().keys())
        self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in masked.parameters()))
        masked.load_state_dict(original.state_dict(), strict=True)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, masked.state_dict()[key]), key)
        x, t, cond = torch.randn(2, 4, 4), torch.rand(2), torch.randn(2, 7, 1024)
        reference = original(x, t, {"main": cond})
        torch.testing.assert_close(masked(x, t, {"main": cond}), reference, rtol=0, atol=0)
        torch.testing.assert_close(masked(x, t, {"main": cond, "main_mask": torch.ones(2, 7, dtype=torch.bool)}), reference)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(original.state_dict(), root / "model.ckpt")
            params = {"input_size": 4, "in_channels": 4, "hidden_size": 64, "context_dim": 1024,
                      "depth": 4, "num_heads": 4, "num_moe_layers": 0, "use_attention_pooling": False, "qk_norm": True}
            (root / "config.yaml").write_text(yaml.safe_dump({"model": {"params": params}}))
            loaded = MaskedHunYuanDiTPlain.from_single_file(
                str(root / "model.ckpt"), str(root / "config.yaml"), device="cpu", dtype=torch.float32)
            self.assertIsInstance(loaded, MaskedHunYuanDiTPlain)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_masked_batch_matches_unpadded_official_outputs_and_gradients(self):
        masked, original = tiny_dit(), tiny_dit(HunYuanDiTPlain)
        original.load_state_dict(masked.state_dict(), strict=True)
        x = torch.randn(2, 4, 4, requires_grad=True)
        t = torch.rand(2)
        cond = torch.randn(2, 7, 1024, requires_grad=True)
        valid = torch.tensor([[False] * 4 + [True] * 3, [True] * 7])
        result = masked(x, t, {"main": cond, "main_mask": valid})
        result.square().mean().backward()
        single_x = x.detach().clone().requires_grad_()
        single_cond = cond.detach().clone().requires_grad_()
        expected = torch.cat([original(single_x[i:i+1], t[i:i+1], {"main": single_cond[i:i+1, valid[i]]})
                              for i in range(2)])
        expected.square().mean().backward()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(x.grad, single_x.grad)
        torch.testing.assert_close(cond.grad, single_cond.grad)
        for (name, a), (_, b) in zip(masked.named_parameters(), original.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4, msg=name)
        changed = cond.detach().clone()
        changed[~valid] = 1e6
        torch.testing.assert_close(masked(x, t, {"main": changed, "main_mask": valid}), result)
        extra = torch.cat([changed, torch.randn(2, 5, 1024) * 1e6], dim=1)
        extra_valid = torch.cat([valid, torch.zeros(2, 5, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(masked(x, t, {"main": extra, "main_mask": extra_valid}), result)
        for bad in (torch.zeros_like(valid), valid.float(), valid[:1]):
            with self.assertRaises(ValueError):
                masked(x, t, {"main": cond, "main_mask": bad})

    def test_connector_and_dit_batched_gradients_match_individual_texts(self):
        cond, model = conditioner(task_prefix=""), tiny_dit()
        reference_cond, reference_model = copy.deepcopy(cond), copy.deepcopy(model)
        texts = ["chair", "a wooden table", "lamp"]
        x, t = torch.randn(3, 4, 4), torch.rand(3)
        contexts = cond(text=texts)
        result = model(x, t, contexts)
        result.square().mean().backward()
        expected = torch.cat([reference_model(x[i:i+1], t[i:i+1], reference_cond(text=text))
                              for i, text in enumerate(texts)])
        expected.square().mean().backward()
        torch.testing.assert_close(result, expected)
        for component, reference in ((cond.connector, reference_cond.connector), (model, reference_model)):
            for (name, a), (_, b) in zip(component.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=2e-4, msg=name)
        self.assertTrue(all(p.grad is None for p in cond.backbone.parameters()))


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

    def test_preserves_official_uid_and_surface_stream_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, captions = fixture(Path(directory))
            text = dataset(rows, captions)
            uid_rng = random.Random(20261101)
            point_rng = np.random.default_rng(20261102)
            with patch("cv2.imread", side_effect=AssertionError("image opened")):
                for a in iter(text):
                    self.assertNotIn("image", a)
                    self.assertNotIn("mask", a)
                    uid = rows[uid_rng.randint(0, len(rows) - 1)]["uid"]
                    self.assertEqual(a["abo_hssd_uid"], uid)
                    arrays = text.decode(uid)
                    expected, _ = AlignedShapeLatentDataset.load_surface_sdf_points(
                        SimpleNamespace(pc_size=81920, pc_sharpedge_size=0, sharpedge_label=True, return_normal=True),
                        point_rng, arrays["random_surface"], arrays["sharpedge_surface"])
                    self.assertTrue(torch.equal(a["surface"], expected))
                    commit(text, a)
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

    def test_data_module_uses_configured_batch_and_has_no_fixed_pending_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, captions = fixture(root)
            for size in (1, 2, 8, 40):
                module = ABOHSSDTextDataModule(root, root / "metadata.jsonl", "toy", batch_size=size)
                module.dataset = dataset(rows, captions)
                loader = module.train_dataloader()
                self.assertEqual(loader.batch_size, size)
                batch = next(iter(loader))
                self.assertEqual(batch["surface"].shape, (size, 81920, 7))
                self.assertEqual(len(batch["text"]), size)
                module.commit_batch(batch)
                self.assertEqual(module.consumed_samples, size)
                self.assertEqual(module.consumed_batches, 1)
            for size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    ABOHSSDTextDataModule(root, root / "metadata.jsonl", "toy", batch_size=size)


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
        contexts = self.cond_stage_model(text=batch["text"])
        context, valid = contexts["main"], contexts["main_mask"]
        pooled = context.sum(1) / valid.sum(1, keepdim=True)
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        self.samples.append((tuple(batch["abo_hssd_uid"]), tuple(batch["caption_index"].tolist()), noise))
        loss = (self.model(pooled) - noise).square().mean()
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

    def test_configuration_and_batch_settings(self):
        new = yaml.safe_load((ROOT / "configs/qwen3vl_stage2_abo_hssd_20261105.yaml").read_text())
        self.assertEqual(new["model"]["target"], "hy3dshape.models.diffusion.flow_matching_sit.Diffuser")
        a = new["model"]["params"]
        self.assertEqual(a["denoiser_cfg"]["target"], "scripts.native_21dit.qwen3vl_stage2.denoiser.MaskedHunYuanDiTPlain")
        self.assertEqual(a["first_stage_config"]["target"], "hy3dshape.models.autoencoders.ShapeVAE")
        self.assertEqual(a["z_scale_factor"], 1.0039506158752403)
        self.assertEqual(a["optimizer_cfg"]["optimizer"]["target"], "torch.optim.AdamW")
        self.assertEqual(a["optimizer_cfg"]["optimizer"]["params"], {"betas": [0.9, 0.99], "eps": 1e-6, "weight_decay": 0.01})
        self.assertEqual(new["training"]["steps"], 1000000000)
        self.assertEqual(new["training"]["every_n_train_steps"], 3000)
        self.assertEqual(batch_settings(new), {"batch_size": 1, "accumulation": 8})
        new["dataset"]["params"]["batch_size"] = 2
        new["training"]["update_every"] = 3
        self.assertEqual(batch_settings(new), {"batch_size": 2, "accumulation": 3})
        for name, path in (("batch_size", new["dataset"]["params"]), ("update_every", new["training"])):
            with self.subTest(name=name):
                path[name] = 0
                with self.assertRaises(ValueError):
                    batch_settings(new)
                path[name] = 2
        self.assertIsNone(a["image_processor_cfg"])
        settings = json.loads((ROOT / "configs/qwen3vl_stage2_abo_hssd_20261105.json").read_text())
        self.assertNotIn("microbatch", settings)
        self.assertNotIn("accumulation", settings)

    def test_source_contract_is_independent_of_image_experiment(self):
        hashes = source_hashes()
        self.assertTrue(hashes)
        self.assertTrue(all("qwen3vl_stage2" in name for name in hashes))
        self.assertFalse(any("native21_official" in name for name in hashes))
        self.assertFalse(any(name.startswith("scripts.native_21dit.native21_official") for name in sys.modules))
        # A fresh interpreter with these imports blocked must load every text
        # runtime module; no test-time pre-import can hide a baseline dependency.
        result = subprocess.run([sys.executable, "-B", __file__, "--isolation-probe"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ToyData(ABOHSSDTextDataModule):
    def __init__(self, root, batch_size):
        super().__init__(root, root / "metadata.jsonl", "toy", batch_size=batch_size)
        rows, captions = fixture(root)
        self.dataset = dataset(rows, captions)


def resume_probe(mode, root, batch_size, accumulation):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    root.mkdir(parents=True, exist_ok=True)
    folder = root / mode
    folder.mkdir()
    pl.seed_everything(20261101)
    dm = ToyData(folder, batch_size)
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
                         accumulate_grad_batches=accumulation, precision=32, callbacks=[callback, scalars, checkpoint],
                         logger=TensorBoardLogger(save_dir=str(folder / "log"), name="tensorboard"),
                         enable_progress_bar=False, enable_model_summary=False, limit_val_batches=0,
                         val_check_interval=3000, check_val_every_n_epoch=None, log_every_n_steps=500)
    resume = str(next((root / "first/ckpt").glob("*.ckpt"))) if mode == "resume" else None
    # Paths/large-file contracts are covered separately; exercise the production
    # callbacks and standard Lightning loop/checkpoint in fresh CPU processes.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "cpu-probe"}), patch(
            "scripts.native_21dit.qwen3vl_stage2.callbacks.integration_contract",
            return_value={"metadata_sha256": "toy", "probe": True,
                          "batching": {"batch_size": batch_size, "accumulation": accumulation}}):
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
    path = max((folder / "ckpt").glob("*.ckpt"), key=lambda p: p.stat().st_mtime_ns)
    torch.save({"checkpoint": torch.load(path, weights_only=False), "samples": model.samples,
                "consumed": dm.consumed_samples, "batches": dm.consumed_batches,
                "gpu_initialized": torch.cuda.is_initialized()}, folder / "result.pt")


class RecoveryTests(unittest.TestCase):
    def test_fresh_process_resume_equals_continuous(self):
        for batch_size, accumulation in ((1, 8), (2, 3), (8, 1)):
            with self.subTest(batch_size=batch_size, accumulation=accumulation):
                self.check_recovery(batch_size, accumulation)

    def check_recovery(self, batch_size, accumulation):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("continuous", "first", "resume"):
                result = subprocess.run([sys.executable, "-B", __file__, "--resume-probe", mode, str(root),
                                         str(batch_size), str(accumulation)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            continuous = torch.load(root / "continuous/result.pt", weights_only=False)
            first = torch.load(root / "first/result.pt", weights_only=False)
            restored = torch.load(root / "resume/result.pt", weights_only=False)
            self.assertEqual(continuous["samples"], first["samples"] + restored["samples"])
            self.assertEqual(restored["consumed"], 2 * batch_size * accumulation)
            self.assertEqual(restored["batches"], 2 * accumulation)
            self.assertFalse(restored["gpu_initialized"])
            def same(a, b, location="checkpoint"):
                if torch.is_tensor(a):
                    self.assertTrue(torch.equal(a, b), location)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys(), location)
                    for key in a:
                        same(a[key], b[key], location + "." + str(key))
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b), location)
                    for i, (x, y) in enumerate(zip(a, b)):
                        same(x, y, location + "." + str(i))
                elif isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    self.assertEqual(a, b, location)
            for key in ("state_dict", "optimizer_states", "lr_schedulers"):
                same(continuous["checkpoint"][key], restored["checkpoint"][key])
            # ModelCheckpoint contains intentionally different output paths.
            for result in (continuous, restored):
                self.assertIn("TextUpdateScalarCallback", result["checkpoint"]["callbacks"])
            same(continuous["checkpoint"]["callbacks"]["TextUpdateScalarCallback"],
                 restored["checkpoint"]["callbacks"]["TextUpdateScalarCallback"])
            # Same random streams, including independent caption state.
            same(continuous["checkpoint"][STATE_KEY]["rng"], restored["checkpoint"][STATE_KEY]["rng"])
            # Lightning re-enters epoch hooks after restart, so epoch ready/
            # started lifecycle counts may differ. Compare the actual batch
            # and logger cursors that determine the next training update.
            for key in ("epoch_loop.batch_progress", "epoch_loop.state_dict"):
                same(continuous["checkpoint"]["loops"]["fit_loop"][key],
                     restored["checkpoint"]["loops"]["fit_loop"][key], key)
            self.assertEqual(restored["checkpoint"]["global_step"], 2)
            self.assertEqual(restored["checkpoint"]["epoch"], continuous["checkpoint"]["epoch"])
            same(continuous["checkpoint"]["ToyData"], restored["checkpoint"]["ToyData"])
            a = continuous["checkpoint"]["ToyData"]["caption"]
            b = restored["checkpoint"]["ToyData"]["caption"]
            same(a, b)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--resume-probe":
        resume_probe(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--isolation-probe":
        for name in ("callbacks", "conditioner", "contracts", "data", "denoiser", "scalars", "shared", "text", "compare_prefix"):
            __import__("scripts.native_21dit.qwen3vl_stage2." + name)
    else:
        unittest.main()
