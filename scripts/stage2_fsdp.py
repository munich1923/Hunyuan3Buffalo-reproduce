"""Project-side FULL_SHARD blocks, explicit auxiliary losses, sharded checkpoint I/O."""
from stage2_runtime import *
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.checkpoint import checkpoint
from stage2_model import Stage2Generator, block_forward

class BlockUnit(nn.Module):
    def __init__(self,block):
        super().__init__();self.block=block
    def forward(self,x,c,condition,skip,checkpoint_blocks):
        if checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block_forward,self.block,x,c,condition,skip,
                              use_reentrant=False,preserve_rng_state=True)
        return block_forward(self.block,x,c,condition,skip)

class FSDPGenerator(Stage2Generator):
    def forward(self,noisy,t,hidden,checkpoint_blocks=True):
        condition=self.connector(hidden);d=self.dit
        c=d.t_embedder(t,condition=None)
        x=torch.cat([c,d.x_embedder(noisy)],dim=1)
        skips=[];aux=x.new_zeros((),dtype=torch.float32)
        for i,block in enumerate(d.blocks):
            skip=None if i<=d.depth//2 else skips.pop()
            x,a=block(x,c,condition,skip,checkpoint_blocks)
            aux=aux+a.float()
            if i<d.depth//2: skips.append(x)
        prediction=d.final_layer(x)
        if prediction.shape!=noisy.shape: raise RuntimeError("FSDP output shape mismatch")
        return prediction,aux

def wrap(model,device):
    # No extra copy of pretrained weights. Block checkpoint is INSIDE its FSDP unit.
    model.__class__=FSDPGenerator
    model.dit.blocks=nn.ModuleList([BlockUnit(b) for b in model.dit.blocks])
    return FSDP(model,auto_wrap_policy=ModuleWrapPolicy({BlockUnit}),
                sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,
                device_id=device,sync_module_states=True,limit_all_gathers=True)

def checkpoint_state(model,optimizer):
    from torch.distributed.checkpoint.state_dict import get_state_dict,StateDictOptions
    m,o=get_state_dict(model,optimizer,options=StateDictOptions(cpu_offload=True))
    return {"model":m,"optimizer":o}

def save_sharded(folder,model,optimizer,metadata,rank_state=None):
    import torch.distributed.checkpoint as dcp
    from stage2_checkpoint import rng_state
    rank=dist.get_rank();folder=path(folder);temporary=folder.with_name(folder.name+".pending")
    if rank==0:
        if folder.exists() or temporary.exists(): raise FileExistsError(str(folder))
        temporary.mkdir(parents=True)
    dist.barrier();started=time.monotonic()
    dcp.save(checkpoint_state(model,optimizer),checkpoint_id=temporary/"distributed")
    save_tensor(temporary/f"rank_{rank}.pt",{"metadata":metadata,"rng":rng_state(),"rank_state":rank_state})
    dist.barrier()
    if rank==0:
        payload={str(p.relative_to(temporary)):{"bytes":p.stat().st_size,"sha256":digest(p)}
                 for p in temporary.rglob("*") if p.is_file()}
        atomic_json(temporary/"complete.json",{"files":payload,"world_size":dist.get_world_size(),
                    "metadata":metadata,**clock()})
        os.rename(temporary,folder)
    dist.barrier()
    return {"seconds":time.monotonic()-started,
            "bytes":sum(v["bytes"] for v in read_json(folder/"complete.json")["files"].values())}

def load_sharded(folder,model,optimizer,metadata):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_state_dict,StateDictOptions
    from stage2_checkpoint import restore_rng
    folder=path(folder);complete=read_json(folder/"complete.json")
    if complete["metadata"]!=metadata or complete["world_size"]!=dist.get_world_size():
        raise RuntimeError("FSDP checkpoint contract mismatch")
    # Parallel verification, one rank owns each payload; all ranks must finish.
    for i,(name,record) in enumerate(sorted(complete["files"].items())):
        if i%dist.get_world_size()==dist.get_rank() and digest(folder/name)!=record["sha256"]:
            raise RuntimeError("FSDP checkpoint checksum mismatch")
    dist.barrier();state=checkpoint_state(model,optimizer)
    dcp.load(state,checkpoint_id=folder/"distributed")
    set_state_dict(model,optimizer,model_state_dict=state["model"],
                   optim_state_dict=state["optimizer"],options=StateDictOptions(cpu_offload=True))
    local=torch.load(folder/f"rank_{dist.get_rank()}.pt",map_location="cpu",weights_only=False)
    restore_rng(local["rng"])
    dist.barrier()
    return local["rank_state"]

def sampling(model,hidden,cfg,generator):
    import numpy as np
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,shift=1.0)
    scheduler.set_timesteps(sigmas=np.linspace(0,1,cfg["evaluation"]["inference_steps"]),device="cuda")
    latent=torch.randn((1,*cfg["cache"]["latent_shape"]),device="cuda",
                       dtype=torch.bfloat16,generator=generator)
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            t=(timestep/1000).expand(1).to(torch.bfloat16)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                # Never call forwarded .velocity(): it would bypass FSDP root hooks.
                velocity,_=model(latent,t,hidden,checkpoint_blocks=False)
            latent=scheduler.step(velocity,timestep,latent).prev_sample
    return latent.float().cpu()/cfg["cache"]["scale"]
