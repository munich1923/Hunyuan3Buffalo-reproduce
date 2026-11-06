"""Use the existing scalar windows, measuring DiT and connector separately."""
import json
import math
from pathlib import Path

import torch

from scripts.native_21dit.native21_official.monitoring import UpdateScalarCallback


def grad_norm(module):
    norms = [p.grad.detach().float().norm(2) for p in module.parameters() if p.grad is not None]
    if not norms:
        raise RuntimeError("Missing trainable-component gradients")
    result = float(torch.stack(norms).double().norm(2).item())
    if not math.isfinite(result):
        raise RuntimeError("Nonfinite accumulated gradient norm")
    return result


class TextUpdateScalarCallback(UpdateScalarCallback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        step = int(trainer.global_step) + 1
        if step % self.every == 0:
            self.pending_norm = (step, grad_norm(pl_module.model))
            self.connector_norm = (step, grad_norm(pl_module.cond_stage_model.connector))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        previous_step = self.last_step
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        step = int(trainer.global_step)
        if step != previous_step and step % self.every == 0:
            if self.connector_norm[0] != step:
                raise RuntimeError("Connector norm/update boundary differs")
            metrics = {"train/connector_grad_norm_l2_pre_clip": self.connector_norm[1]}
            trainer.logger.log_metrics(metrics, step=step)
            trainer.logger.experiment.flush()
            path = Path(trainer.logger.save_dir).parent / "connector_scalar_updates.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, **metrics}) + "\n")
