"""Data/RNG bookkeeping and finite checks; no custom training-stop condition."""
import copy,hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import Callback
from .shared import ROOT,read,write,sha,stamp,load_settings,integration_contract

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu']);torch.cuda.set_rng_state_all(s['torch_cuda'])
def finalize_checkpoint_progress(checkpoint,consumed,step):
    """Serialize a completed batch at PL1.9.5's in-batch save boundary.

    ModelCheckpoint saves before Lightning increments completed/log counters.
    Copy the serialized loop tree so no live training-loop state is changed.
    """
    loops=copy.deepcopy(checkpoint['loops']);fit=loops['fit_loop'];progress=fit['epoch_loop.batch_progress'];adjustments={}
    for side in ['current','total']:
        counter=progress[side]
        if not (counter['ready']==counter['started']==counter['processed']==consumed):raise RuntimeError('Unexpected Lightning batch progress at checkpoint')
        before=counter['completed']
        if before not in [consumed-1,consumed]:raise RuntimeError('Unexpected Lightning completed-batch cursor')
        counter['completed']=consumed;adjustments[side]=dict(before=before,after=consumed)
    state=fit['epoch_loop.state_dict'];before=state['_batches_that_stepped']
    if before not in [step-1,step]:raise RuntimeError('Unexpected Lightning logger update cursor')
    state['_batches_that_stepped']=step;adjustments['logger_steps']=dict(before=before,after=step)
    checkpoint['loops']=loops
    return adjustments
def tensor_dtypes(module):
    out={}
    for p in module.parameters():out[str(p.dtype)]=out.get(str(p.dtype),0)+p.numel()
    return out
class RunStateCallback(Callback):
    def __init__(self,settings_path):
        self.c=load_settings(settings_path);self.run=Path(self.c['run_dir'])
        self.pending_rng=None;self.restored_step=None;self.last_logged_step=-1;self.initial_step=None;self.contract=None
    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        saved=checkpoint.get('abo_hssd_run_state')
        if saved is None:raise RuntimeError('Resume checkpoint lacks ABO/HSSD RNG/data contract')
        self.contract=integration_contract(self.c)
        if saved['contract']!=self.contract:raise RuntimeError('Resume integration/data contract changed')
        self.pending_rng=saved['rng'];self.restored_step=int(checkpoint['global_step'])
        if saved['consumed_samples']!=self.restored_step*8:raise RuntimeError('Checkpoint data position inconsistent')
    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        step=int(trainer.global_step);consumed=trainer.datamodule.consumed_samples
        if consumed!=step*8:raise RuntimeError('Checkpoint not at committed optimizer boundary')
        bookkeeping=finalize_checkpoint_progress(checkpoint,consumed,step)
        checkpoint['abo_hssd_run_state']=dict(version=1,contract=self.contract or integration_contract(self.c),rng=capture_rng(),consumed_samples=consumed,checkpoint_bookkeeping=bookkeeping,pid=os.getpid(),time=stamp())
    def on_fit_start(self,trainer,pl_module):
        self.contract=integration_contract(self.c);self.initial_step=int(trainer.global_step)
        if not all(p.requires_grad for p in pl_module.model.parameters()):raise RuntimeError('DiT is not fully trainable')
        if any(p.requires_grad for p in pl_module.first_stage_model.parameters()):raise RuntimeError('VAE not frozen')
        if any(p.requires_grad for p in pl_module.cond_stage_model.parameters()):raise RuntimeError('DINO not frozen')
        write(self.run/f'entry_identity_pid_{os.getpid()}.json',dict(official_entry=self.c['training_entry'],module_class=type(pl_module).__module__+'.'+type(pl_module).__name__,dit_class=type(pl_module.model).__module__+'.'+type(pl_module.model).__name__,dtypes=dict(dit=tensor_dtypes(pl_module.model),vae=tensor_dtypes(pl_module.first_stage_model),conditioner=tensor_dtypes(pl_module.cond_stage_model)),pipeline_defaults_unchanged=True,conditioner_disable_drop=pl_module.cond_stage_model.disable_drop,conditioner_drop_ratio=pl_module.cond_stage_model.drop_ratio,training_step_replaced=False,optimizer_replaced=False,contract=self.contract,time=stamp(),pid=os.getpid()))
    def on_train_start(self,trainer,pl_module):
        self.initial_step=int(trainer.global_step)
        if trainer.datamodule.consumed_samples!=self.initial_step*8:raise RuntimeError('Restored dataset position differs from global_step')
        if self.restored_step is not None:
            if self.restored_step!=self.initial_step or self.pending_rng is None:raise RuntimeError('Incomplete resume')
            # PL1.9.5 does not restore this save-only cursor. Avoid rewriting
            # the restored checkpoint during an accumulation microbatch.
            from pytorch_lightning.callbacks import ModelCheckpoint
            for callback in trainer.callbacks:
                if isinstance(callback,ModelCheckpoint):callback._last_global_step_saved=self.initial_step
            for opt in trainer.optimizers:
                if any(float(s['step'])!=self.initial_step for s in opt.state.values() if 'step' in s):raise RuntimeError('Adam step restore mismatch')
            write(self.run/f'resume_{self.initial_step:06d}.json',dict(standard_lightning_state_restored=True,rng_restore_pending=True,consumed_samples=trainer.datamodule.consumed_samples,step=self.initial_step,pid=os.getpid(),time=stamp()))
    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        # Runs after the upstream on_train_epoch_start seed hook, before forward.
        if self.pending_rng is not None:
            restore_rng(self.pending_rng);self.pending_rng=None
            p=self.run/f'resume_{self.initial_step:06d}.json';d=read(p);d.update(rng_restore_pending=False,rng_restored=True,time=stamp());write(p,d)
        for key in ['surface','image','mask']:
            if not torch.isfinite(batch[key]).all():raise RuntimeError('Nonfinite training input '+key)
    def on_before_optimizer_step(self,trainer,pl_module,optimizer,optimizer_idx):
        for n,p in pl_module.model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient '+n)
        if any(p.grad is not None for p in pl_module.first_stage_model.parameters()) or any(p.grad is not None for p in pl_module.cond_stage_model.parameters()):raise RuntimeError('Frozen component received gradients')
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        trainer.datamodule.commit_batch(batch)
        loss=outputs['loss'] if isinstance(outputs,dict) else outputs
        if loss is not None and not torch.isfinite(loss).all():raise RuntimeError('Nonfinite loss')
        step=int(trainer.global_step);consumed=trainer.datamodule.consumed_samples
        if consumed%8==0:
            if consumed!=step*8:raise RuntimeError('Accumulation/global_step mismatch')
            if step!=self.last_logged_step:
                for n,p in pl_module.model.named_parameters():
                    if not torch.isfinite(p).all():raise RuntimeError('Nonfinite parameter '+n)
                opt=trainer.optimizers[0];state=next(iter(opt.state.values()),{})
                record=dict(step=step,consumed_samples=consumed,last_uid=trainer.datamodule.dataset.last_committed_uid,loss=float(loss.detach()) if loss is not None else None,lr=opt.param_groups[0]['lr'],parameter_dtype=str(next(pl_module.model.parameters()).dtype),adam_state_dtypes={k:str(v.dtype) for k,v in state.items() if torch.is_tensor(v)},conditioner_disable_drop=pl_module.cond_stage_model.disable_drop,parameters_finite=True,gradients_finite=True,pid=os.getpid(),time=stamp())
                with (self.run/'train_updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                write(self.run/'training_status.json',record);self.last_logged_step=step
