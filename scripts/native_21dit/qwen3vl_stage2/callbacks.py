"""Text-specific assertions around the existing committed-data/RNG bookkeeping."""
import os

import torch

from scripts.native_21dit.native21_official.callbacks import (
    RunStateCallback, restore_rng, tensor_dtypes,
)
from scripts.native_21dit.native21_official.shared import read, stamp, write
from .contracts import integration_contract


def check_optimizer(model, optimizer):
    expected = list(model.model.parameters()) + list(model.cond_stage_model.connector.parameters())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in expected}:
        raise RuntimeError("Optimizer must contain exactly DiT and connector, once each")


class TextRunStateCallback(RunStateCallback):
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        saved = checkpoint.get("abo_hssd_run_state")
        self.contract = integration_contract(self.c)
        if saved is None or saved["contract"] != self.contract:
            raise RuntimeError("Not a matching text-stage2 checkpoint/data/model contract")
        self.pending_rng = saved["rng"]
        self.restored_step = int(checkpoint["global_step"])
        if saved["consumed_samples"] != self.restored_step * 8:
            raise RuntimeError("Checkpoint data position inconsistent")

    def on_fit_start(self, trainer, pl_module):
        self.contract = integration_contract(self.c)
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
            "dtypes": {"dit": tensor_dtypes(pl_module.model), "vae": tensor_dtypes(pl_module.first_stage_model),
                       "qwen": tensor_dtypes(cond.backbone), "connector": tensor_dtypes(cond.connector)},
            "input": "text_only", "optimizer_replaced": False, "training_step_replaced": False,
            "pid": os.getpid(), "time": stamp(),
        })

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.contract is None:
            raise RuntimeError("Text experiment contract must be established before saving")
        # Reuse accumulation-boundary/loop-cursor repair with the new contract.
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

    # Inherit on_train_start for loop cursor and optimizer-step recovery.
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
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
        if len(batch["text"]) != 1 or not batch["text"][0].strip():
            raise RuntimeError("Expected one nonempty original caption")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        for prefix, module in (("dit", pl_module.model), ("connector", pl_module.cond_stage_model.connector)):
            for name, param in module.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError("Nonfinite gradient " + prefix + "." + name)
        for module in (pl_module.first_stage_model, pl_module.cond_stage_model.backbone):
            if any(p.grad is not None for p in module.parameters()):
                raise RuntimeError("Frozen VAE/Qwen received gradients")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.datamodule.consumed_samples % 8 == 0:
            if any(not torch.isfinite(p).all() for p in pl_module.cond_stage_model.connector.parameters()):
                raise RuntimeError("Nonfinite connector parameter")
