"""Update-aligned loss and separate pre-clip DiT/connector gradient norms."""
import json
import math
from pathlib import Path

import torch
from pytorch_lightning import Callback

from .shared import positive_int


def grad_norm(module):
    norms = [p.grad.detach().float().norm(2) for p in module.parameters() if p.grad is not None]
    if not norms:
        raise RuntimeError("Missing trainable-component gradients")
    result = float(torch.stack(norms).double().norm(2).item())
    if not math.isfinite(result):
        raise RuntimeError("Nonfinite accumulated gradient norm")
    return result


class TextUpdateScalarCallback(Callback):
    def __init__(self, every_n_updates=500):
        self.every = positive_int(every_n_updates, "every_n_updates")
        self.loss_sum = 0.0
        self.micro_count = 0
        self.window_loss = 0.0
        self.window_updates = 0
        self.last_step = None
        self.pending_norm = None
        self.connector_norm = None

    def state_dict(self):
        return {key: getattr(self, key) for key in ("loss_sum", "micro_count", "window_loss", "window_updates")}

    def load_state_dict(self, state):
        for key in ("loss_sum", "micro_count", "window_loss", "window_updates"):
            setattr(self, key, state[key])

    def on_train_start(self, trainer, pl_module):
        self.last_step = int(trainer.global_step)
        if self.micro_count:
            raise RuntimeError("Scalar checkpoint was not at an optimizer boundary")
        self.pending_norm = self.connector_norm = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        # PL1.9.5 calls this after accumulation/unscaling, before clipping.
        step = int(trainer.global_step) + 1
        if step % self.every == 0:
            self.pending_norm = (step, grad_norm(pl_module.model))
            self.connector_norm = (step, grad_norm(pl_module.cond_stage_model.connector))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            raise RuntimeError("Missing training loss for scalar logging")
        # Every microbatch has the configured fixed B. PL returns its mean loss
        # divided by accumulation, so summing one update gives its batch mean.
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite scalar loss")
        self.loss_sum += value
        self.micro_count += 1
        step = int(trainer.global_step)
        if step == self.last_step:
            return
        if step != self.last_step + 1 or self.micro_count != int(trainer.accumulate_grad_batches):
            raise RuntimeError("Scalar logger update/accumulation mismatch")
        update_loss = self.loss_sum
        self.window_loss += update_loss
        self.window_updates += 1
        self.loss_sum = 0.0
        self.micro_count = 0
        self.last_step = step
        if step % self.every:
            return
        if self.pending_norm is None or self.pending_norm[0] != step or self.connector_norm is None or self.connector_norm[0] != step:
            raise RuntimeError("Missing pre-clip norms for completed update")
        connector_metrics = {"train/connector_grad_norm_l2_pre_clip": self.connector_norm[1]}
        metrics = {
            "train/loss_update": update_loss,
            "train/loss_500_updates": self.window_loss / self.window_updates,
            "train/grad_norm_l2_pre_clip": self.pending_norm[1],
            "train/lr_update": float(trainer.optimizers[0].param_groups[0]["lr"]),
            **connector_metrics,
        }
        trainer.logger.log_metrics(metrics, step=step)
        trainer.logger.experiment.flush()
        root = Path(trainer.logger.save_dir).parent
        for name, record in (
            ("scalar_updates.jsonl", {"step": step, "window_updates": self.window_updates, **metrics}),
            ("connector_scalar_updates.jsonl", {"step": step, **connector_metrics}),
        ):
            with (root / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        self.window_loss = 0.0
        self.window_updates = 0
        self.pending_norm = self.connector_norm = None
