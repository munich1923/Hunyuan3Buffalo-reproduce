"""Text experiment data/RNG recovery and finite checks, independent of the baseline."""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback

from .contracts import integration_contract
from .shared import load_settings, read, stamp, write

STATE_KEY = "qwen3vl_stage2_run_state"


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def finalize_checkpoint_progress(checkpoint, consumed_batches, step):
    """Repair PL1.9.5's serialized in-batch save cursor, counting batches, not samples."""
    loops = copy.deepcopy(checkpoint["loops"])
    fit = loops["fit_loop"]
    progress = fit["epoch_loop.batch_progress"]
    adjustments = {}
    for side in ("current", "total"):
        counter = progress[side]
        if not (counter["ready"] == counter["started"] == counter["processed"] == consumed_batches):
            raise RuntimeError("Unexpected Lightning batch progress at checkpoint")
        before = counter["completed"]
        if before not in (consumed_batches - 1, consumed_batches):
            raise RuntimeError("Unexpected Lightning completed-batch cursor")
        counter["completed"] = consumed_batches
        adjustments[side] = {"before": before, "after": consumed_batches}
    state = fit["epoch_loop.state_dict"]
    before = state["_batches_that_stepped"]
    if before not in (step - 1, step):
        raise RuntimeError("Unexpected Lightning logger update cursor")
    state["_batches_that_stepped"] = step
    adjustments["logger_steps"] = {"before": before, "after": step}
    checkpoint["loops"] = loops
    return adjustments


def tensor_dtypes(module):
    result = {}
    for parameter in module.parameters():
        key = str(parameter.dtype)
        result[key] = result.get(key, 0) + parameter.numel()
    return result


def check_optimizer(model, optimizer):
    expected = list(model.model.parameters()) + list(model.cond_stage_model.connector.parameters())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in expected}:
        raise RuntimeError("Optimizer must contain exactly DiT and connector, once each")


class TextRunStateCallback(Callback):
    def __init__(self, settings_path):
        self.c = load_settings(settings_path)
        self.run = Path(self.c["run_dir"])
        self.pending_rng = None
        self.restored_step = None
        self.initial_step = None
        self.last_step = None
        self.contract = None

    def _check_position(self, samples, batches, step):
        spec = self.contract["batching"]
        if batches != step * spec["accumulation"] or samples != batches * spec["batch_size"]:
            raise RuntimeError("Sample/batch/update checkpoint positions are inconsistent")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        saved = checkpoint.get(STATE_KEY)
        self.contract = integration_contract(self.c)
        if saved is None or saved.get("version") != 2 or saved["contract"] != self.contract:
            raise RuntimeError("Not a matching text-stage2 checkpoint/data/model contract")
        self.pending_rng = saved["rng"]
        self.restored_step = int(checkpoint["global_step"])
        self._check_position(saved["consumed_samples"], saved["consumed_batches"], self.restored_step)

    def on_fit_start(self, trainer, pl_module):
        self.contract = integration_contract(self.c)
        spec = self.contract["batching"]
        if trainer.datamodule.batch_size != spec["batch_size"] or trainer.accumulate_grad_batches != spec["accumulation"]:
            raise RuntimeError("Runtime DataLoader/accumulation settings differ from the training YAML")
        self.initial_step = int(trainer.global_step)
        cond = pl_module.cond_stage_model
        if not all(p.requires_grad for p in pl_module.model.parameters()):
            raise RuntimeError("DiT must be fully trainable")
        if not all(p.requires_grad for p in cond.connector.parameters()):
            raise RuntimeError("Connector must be fully trainable")
        if any(p.requires_grad for p in pl_module.first_stage_model.parameters()) or any(
                p.requires_grad for p in cond.backbone.parameters()):
            raise RuntimeError("VAE and Qwen must be frozen")
        if cond.backbone.training or any(p.dtype != torch.bfloat16 for p in cond.backbone.parameters()):
            raise RuntimeError("Qwen must stay in eval mode with BF16 weights")
        if not cond.disable_drop or cond.drop_ratio != 0.1:
            raise RuntimeError("Keep the reviewed conditioner dropout policy")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self.c["gpu_uuid"]:
            raise RuntimeError("Expose only the authorized physical GPU1 UUID")
        check_optimizer(pl_module, trainer.optimizers[0])
        if trainer.datamodule.dataset.contract["metadata_sha256"] != self.contract["metadata_sha256"]:
            raise RuntimeError("Metadata changed after data setup")
        write(self.run / f"entry_identity_pid_{os.getpid()}.json", {
            "official_entry": self.c["training_entry"], "contract": self.contract,
            "module_class": type(pl_module).__module__ + "." + type(pl_module).__name__,
            "dit_class": type(pl_module.model).__module__ + "." + type(pl_module.model).__name__,
            "condition_mask": "main_mask_bool_true_is_valid",
            "dtypes": {"dit": tensor_dtypes(pl_module.model), "vae": tensor_dtypes(pl_module.first_stage_model),
                       "qwen": tensor_dtypes(cond.backbone), "connector": tensor_dtypes(cond.connector)},
            "input": "text_only", "optimizer_replaced": False, "training_step_replaced": False,
            "pid": os.getpid(), "time": stamp(),
        })

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.contract is None:
            raise RuntimeError("Text experiment contract must be established before saving")
        step = int(trainer.global_step)
        data = trainer.datamodule
        self._check_position(data.consumed_samples, data.consumed_batches, step)
        bookkeeping = finalize_checkpoint_progress(checkpoint, data.consumed_batches, step)
        checkpoint[STATE_KEY] = {
            "version": 2, "contract": self.contract, "rng": capture_rng(),
            "consumed_samples": data.consumed_samples, "consumed_batches": data.consumed_batches,
            "checkpoint_bookkeeping": bookkeeping, "pid": os.getpid(), "time": stamp(),
        }

    def on_train_start(self, trainer, pl_module):
        self.initial_step = self.last_step = int(trainer.global_step)
        data = trainer.datamodule
        self._check_position(data.consumed_samples, data.consumed_batches, self.initial_step)
        if self.restored_step is not None:
            if self.restored_step != self.initial_step or self.pending_rng is None:
                raise RuntimeError("Incomplete text-stage2 resume")
            from pytorch_lightning.callbacks import ModelCheckpoint
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    callback._last_global_step_saved = self.initial_step
            for optimizer in trainer.optimizers:
                if any(float(state["step"]) != self.initial_step for state in optimizer.state.values() if "step" in state):
                    raise RuntimeError("Adam step restore mismatch")
            write(self.run / f"resume_{self.initial_step:06d}.json", {
                "standard_lightning_state_restored": True, "rng_restore_pending": True,
                "consumed_samples": data.consumed_samples, "consumed_batches": data.consumed_batches,
                "step": self.initial_step, "pid": os.getpid(), "time": stamp(),
            })

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Restore after the upstream epoch-start seed hook and before forward.
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
            path = self.run / f"resume_{self.initial_step:06d}.json"
            value = read(path)
            value.update(rng_restore_pending=False, rng_restored=True, time=stamp())
            write(path, value)
        if "image" in batch or "mask" in batch:
            raise RuntimeError("Images/masks are forbidden in text-stage2 batches")
        if not torch.isfinite(batch["surface"]).all():
            raise RuntimeError("Nonfinite surface")
        texts = batch["text"]
        if len(texts) != batch["surface"].shape[0] or len(texts) != trainer.datamodule.batch_size or any(
                not isinstance(text, str) or not text.strip() for text in texts):
            raise RuntimeError("Expected one nonempty original caption per geometry sample")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        for prefix, module in (("dit", pl_module.model), ("connector", pl_module.cond_stage_model.connector)):
            for name, parameter in module.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Nonfinite gradient " + prefix + "." + name)
        for module in (pl_module.first_stage_model, pl_module.cond_stage_model.backbone):
            if any(parameter.grad is not None for parameter in module.parameters()):
                raise RuntimeError("Frozen VAE/Qwen received gradients")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        data = trainer.datamodule
        data.commit_batch(batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is None or not bool(torch.isfinite(loss).all()):
            raise RuntimeError("Missing/nonfinite training loss")
        step = int(trainer.global_step)
        if step == self.last_step:
            return
        if step != self.last_step + 1:
            raise RuntimeError("Unexpected optimizer update sequence")
        self._check_position(data.consumed_samples, data.consumed_batches, step)
        for module in (pl_module.model, pl_module.cond_stage_model.connector):
            if any(not bool(torch.isfinite(parameter).all()) for parameter in module.parameters()):
                raise RuntimeError("Nonfinite trainable parameter")
        optimizer = trainer.optimizers[0]
        state = next(iter(optimizer.state.values()), {})
        record = {
            "step": step, "consumed_samples": data.consumed_samples, "consumed_batches": data.consumed_batches,
            "last_uid": data.dataset.last_committed_uid, "loss": float(loss.detach()),
            "lr": optimizer.param_groups[0]["lr"], "parameter_dtype": str(next(pl_module.model.parameters()).dtype),
            "adam_state_dtypes": {key: str(value.dtype) for key, value in state.items() if torch.is_tensor(value)},
            "conditioner_disable_drop": pl_module.cond_stage_model.disable_drop,
            "parameters_finite": True, "gradients_finite": True, "pid": os.getpid(), "time": stamp(),
        }
        with (self.run / "train_updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        write(self.run / "training_status.json", record)
        self.last_step = step
