"""Small CPU-only Lightning save/fresh-process resume check; no geometry/model training."""
import os,sys,json,random,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
os.environ.update(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.path[:0]=[str(ROOT),str(ROOT/"src/Hunyuan3D-2.1/hy3dshape")]
import torch,numpy as np,pytorch_lightning as pl
from pytorch_lightning import Callback,LightningDataModule,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from scripts.native_21dit.native21_official.data import ABOHSSDSurfaceDataset
from scripts.native_21dit.native21_official.callbacks import capture_rng,restore_rng,finalize_checkpoint_progress
OUT=ROOT/'results/cpu_resume_probe'
class TinyDataset(ABOHSSDSurfaceDataset):
    def __init__(self):
        torch.utils.data.IterableDataset.__init__(self)
        self.data_list=['uid_103','uid_81'];self.contract={'cpu_probe':True};self.seed=20261101
        self.uid_rng=random.Random(self.seed);self.point_rng=np.random.default_rng(self.seed+1)
        self.emitted_samples=0;self.consumed_samples=0;self.pending={};self.last_committed_uid=None;self._committed=self._snapshot()
    def sample_uid(self,uid):return {'x':torch.tensor([float(self.point_rng.random())],dtype=torch.float32)}
class TinyData(LightningDataModule):
    def __init__(self):super().__init__();self.dataset=TinyDataset()
    @property
    def consumed_samples(self):return self.dataset.consumed_samples
    def train_dataloader(self):return torch.utils.data.DataLoader(self.dataset,batch_size=1,num_workers=0,generator=torch.Generator().manual_seed(17))
    def state_dict(self):return self.dataset.state_dict()
    def load_state_dict(self,state):self.dataset.load_state_dict(state)
    def commit_batch(self,batch):self.dataset.commit_batch(batch)
class TinyModel(LightningModule):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.2]));self.samples=[]
    def training_step(self,batch,batch_idx):
        noise=[random.random(),float(np.random.random()),float(torch.rand(()))]
        self.samples.append(dict(position=int(batch['abo_hssd_sample_position']),uid=batch['abo_hssd_uid'][0],x=float(batch['x']),noise=noise,batch_idx=batch_idx,global_step=self.global_step))
        loss=(self.weight*batch['x']-sum(noise))**2;self.log('train/total_loss',loss.squeeze(),on_step=True,on_epoch=False);return loss.mean()
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=1e-3)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1-step/100),'interval':'step'}}
    def on_train_epoch_start(self):pl.seed_everything(self.trainer.global_rank)
class TinyState(Callback):
    def __init__(self,stop):self.stop=stop;self.pending_rng=None;self.restored_first=False;self.initial=None
    def on_load_checkpoint(self,trainer,model,checkpoint):self.pending_rng=checkpoint['probe']['rng']
    def on_train_start(self,trainer,model):
        self.initial=trainer.global_step
        print('probe_on_train_start',self.initial,trainer.datamodule.consumed_samples,flush=True)
        assert trainer.datamodule.consumed_samples==trainer.global_step*8
        if self.pending_rng is not None:
            for callback in trainer.callbacks:
                if isinstance(callback,ModelCheckpoint):callback._last_global_step_saved=self.initial
            print('probe_restored_last_saved',[c._last_global_step_saved for c in trainer.callbacks if isinstance(c,ModelCheckpoint)],flush=True)
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        print('probe_batch_start',batch_idx,trainer.global_step,int(batch['abo_hssd_sample_position']),flush=True)
        if self.pending_rng is not None:restore_rng(self.pending_rng);self.pending_rng=None;self.restored_first=True
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        trainer.datamodule.commit_batch(batch)
        print('probe_batch_end',batch_idx,trainer.global_step,trainer.datamodule.consumed_samples,flush=True)
        if trainer.datamodule.consumed_samples%8==0:
            assert trainer.datamodule.consumed_samples==trainer.global_step*8
            if trainer.global_step>=self.stop:trainer.should_stop=True
    def on_save_checkpoint(self,trainer,model,checkpoint):
        assert trainer.datamodule.consumed_samples==trainer.global_step*8, dict(consumed=trainer.datamodule.consumed_samples,global_step=trainer.global_step,samples=model.samples,last_saved=[c._last_global_step_saved for c in trainer.callbacks if isinstance(c,ModelCheckpoint)])
        bookkeeping=finalize_checkpoint_progress(checkpoint,trainer.datamodule.consumed_samples,trainer.global_step)
        checkpoint['probe']={'rng':capture_rng(),'consumed_samples':trainer.datamodule.consumed_samples,'bookkeeping':bookkeeping}
mode=sys.argv[1];OUT.mkdir(exist_ok=True);folder=OUT/mode;folder.mkdir(exist_ok=True)
pl.seed_everything(20261101)
model=TinyModel();dm=TinyData();cb=TinyState(2 if mode=='first' else 4)
mc=ModelCheckpoint(dirpath=folder,filename='ckpt-{step:08d}',monitor='train/total_loss',mode='max',save_top_k=-1,every_n_train_steps=2)
trainer=pl.Trainer(max_steps=4,accelerator='cpu',devices=1,precision=32,strategy=None,callbacks=[mc,cb],accumulate_grad_batches=8,logger=False,enable_progress_bar=False,enable_model_summary=False,limit_val_batches=0,val_check_interval=3000,check_val_every_n_epoch=None)
resume=str(max((OUT/'first').glob('ckpt-step=00000002*.ckpt'),key=lambda p:p.stat().st_mtime_ns)) if mode=='resume' else None
trainer.fit(model,datamodule=dm,ckpt_path=resume)
result=dict(mode=mode,global_step=trainer.global_step,consumed_samples=dm.consumed_samples,initial=cb.initial,samples=model.samples,weight=model.weight.detach().tolist(),rng_restored=cb.restored_first,checkpoint_files=[p.name for p in folder.glob('*.ckpt')],callback_order=[type(c).__name__ for c in trainer.callbacks],gpu_initialized=torch.cuda.is_initialized())
result['production_helper_used']='scripts.native_21dit.native21_official.callbacks.finalize_checkpoint_progress'
assert not result['gpu_initialized'];(folder/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
