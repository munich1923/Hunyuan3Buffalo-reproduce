"""Read-only update-aligned TensorBoard scalars; no model/gradient/RNG mutation."""
import json
import math
from pathlib import Path
import torch
from pytorch_lightning import Callback


class UpdateScalarCallback(Callback):
    """Log the effective-batch loss and pre-clip accumulated-gradient L2 norm."""
    def __init__(self, every_n_updates=500):
        self.every = int(every_n_updates)
        if self.every < 1:
            raise ValueError('every_n_updates must be positive')
        self.loss_sum = 0.0
        self.micro_count = 0
        self.window_loss = 0.0
        self.window_updates = 0
        self.last_step = None
        self.pending_norm = None

    def state_dict(self):
        return dict(loss_sum=self.loss_sum, micro_count=self.micro_count,
                    window_loss=self.window_loss, window_updates=self.window_updates)

    def load_state_dict(self, state_dict):
        for key in ('loss_sum', 'micro_count', 'window_loss', 'window_updates'):
            setattr(self, key, state_dict[key])

    def on_train_start(self, trainer, pl_module):
        self.last_step = int(trainer.global_step)
        if self.micro_count:
            raise RuntimeError('Scalar checkpoint was not at an optimizer boundary')
        self.pending_norm = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        step = int(trainer.global_step) + 1
        if step % self.every:
            return
        # Lightning 1.9.5 calls this after unscaling and before gradient clipping.
        # Convert only temporary norm operands to FP32; never modify .grad.
        norms = [p.grad.detach().float().norm(2) for group in optimizer.param_groups
                 for p in group['params'] if p.grad is not None]
        if not norms:
            raise RuntimeError('No gradients at scalar logging boundary')
        value = float(torch.stack(norms).double().norm(2).item())
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite gradient norm')
        self.pending_norm = (step, value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is None:
            raise RuntimeError('Missing training loss for scalar logging')
        # PL1.9.5 returns loss divided by accumulate_grad_batches here.
        # Summing one complete accumulation gives its effective-batch mean.
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite scalar loss')
        self.loss_sum += value
        self.micro_count += 1
        step = int(trainer.global_step)
        if step == self.last_step:
            return
        if step != self.last_step + 1 or self.micro_count != int(trainer.accumulate_grad_batches):
            raise RuntimeError('Scalar logger update/accumulation mismatch')
        update_loss = self.loss_sum
        self.window_loss += update_loss
        self.window_updates += 1
        self.loss_sum = 0.0
        self.micro_count = 0
        self.last_step = step
        if step % self.every:
            return
        if self.pending_norm is None or self.pending_norm[0] != step:
            raise RuntimeError('Missing pre-clip norm for completed update')
        metrics = {'train/loss_update': update_loss,
                   'train/loss_500_updates': self.window_loss / self.window_updates,
                   'train/grad_norm_l2_pre_clip': self.pending_norm[1],
                   'train/lr_update': float(trainer.optimizers[0].param_groups[0]['lr'])}
        trainer.logger.log_metrics(metrics, step=step)
        trainer.logger.experiment.flush()
        path = Path(trainer.default_root_dir)
        # Official main sets its output in the logger, not default_root_dir.
        path = Path(trainer.logger.save_dir).parent
        with (path / 'scalar_updates.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step, window_updates=self.window_updates,
                                         **metrics)) + '\n')
        self.window_loss = 0.0
        self.window_updates = 0
        self.pending_norm = None
