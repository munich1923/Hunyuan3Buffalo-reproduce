"""Project-side pure generation adapter; pinned upstream remains untouched."""
from stage2_runtime import path, seed
import contextlib
import types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Connector(nn.Module):
    def __init__(self, hidden=3584, context=1024):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden,context),nn.GELU(),nn.Linear(context,context))
    def forward(self, hidden):
        if hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[1]<1:
            raise ValueError("Require one unpadded, nonempty condition")
        return self.net(hidden)


def explicit_moe_forward(self, hidden_states):
    """Same routing/expert computation; auxiliary is returned at block scope, never injected."""
    identity=hidden_states
    shape=hidden_states.shape
    topk_idx,topk_weight,aux=self.gate(hidden_states)
    flat=hidden_states.reshape(-1,hidden_states.shape[-1])
    indices=topk_idx.reshape(-1)
    if self.training:
        expanded=flat.repeat_interleave(self.moe_top_k,dim=0)
        y=torch.empty_like(expanded)
        for i,expert in enumerate(self.experts):
            mask=indices==i
            y[mask]=expert(expanded[mask]).to(expanded.dtype)
        y=(y.view(*topk_weight.shape,-1)*topk_weight.unsqueeze(-1)).sum(dim=1).view(*shape)
    else:
        y=self.moe_infer(flat,indices,topk_weight.reshape(-1,1)).view(*shape)
    self._stage2_aux=aux
    return y+self.shared_experts(identity)


def adapt_moe(dit):
    from hy3dshape.models.denoisers.moe_layers import MoEBlock
    for module in dit.modules():
        if isinstance(module,MoEBlock):
            module.forward=types.MethodType(explicit_moe_forward,module)
            module._stage2_aux=None


def block_forward(block,x,c,condition,skip):
    y=block(x,c,condition,skip_value=skip)
    if block.use_moe:
        auxiliary=block.moe._stage2_aux
        block.moe._stage2_aux=None
        if auxiliary is not None: return y,auxiliary
    return y,x.new_zeros((),dtype=torch.float32)


class Stage2Generator(nn.Module):
    def __init__(self,dit,hidden_dim=3584):
        super().__init__()
        self.dit=dit
        self.connector=Connector(hidden_dim,dit.context_dim)
        if dit.use_pos_emb or dit.use_attention_pooling or dit.with_decoupled_ca:
            raise ValueError("Adapter requires the pinned plain 2.1 configuration")
        adapt_moe(dit)

    def forward(self,noisy,t,hidden,checkpoint_blocks=True):
        condition=self.connector(hidden)
        d=self.dit
        c=d.t_embedder(t,condition=None)
        x=d.x_embedder(noisy)
        x=torch.cat([c,x],dim=1)
        skips=[]
        aux=x.new_zeros((),dtype=torch.float32)
        for i,block in enumerate(d.blocks):
            skip=None if i<=d.depth//2 else skips.pop()
            if checkpoint_blocks and self.training and torch.is_grad_enabled():
                x,block_aux=checkpoint(block_forward,block,x,c,condition,skip,
                                       use_reentrant=False,preserve_rng_state=True)
            else:
                x,block_aux=block_forward(block,x,c,condition,skip)
            aux=aux+block_aux.float()
            if i<d.depth//2: skips.append(x)
        prediction=d.final_layer(x)
        # Official FinalLayer removes the first internal time token.
        if prediction.shape!=noisy.shape:
            raise RuntimeError(f"Expected {noisy.shape}, got {prediction.shape}")
        return prediction,aux

    def velocity(self,noisy,t,hidden):
        return self(noisy,t,hidden,checkpoint_blocks=False)[0]


def load_model(cfg,device="cuda"):
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    dit=HunYuanDiTPlain.from_pretrained(str(path(cfg["models"]["shape"])),
                                      device="cpu",dtype=torch.float32,use_safetensors=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed(cfg["seed"],"connector"))
        model=Stage2Generator(dit,cfg["cache"]["hidden_dim"])
    model.requires_grad_(True)
    return model.to(device=device,dtype=torch.float32)


def flow(mean,logvar,scale,generator):
    eta=torch.randn(mean.shape,device=mean.device,dtype=torch.float32,generator=generator)
    target=scale*(mean.float()+torch.exp(0.5*logvar.float())*eta)
    noise=torch.randn(target.shape,device=target.device,dtype=torch.float32,generator=generator)
    t=torch.rand(target.shape[0],device=target.device,dtype=torch.float32,generator=generator)
    noisy=(1-t[:,None,None])*noise+t[:,None,None]*target
    return noisy,t,target-noise


def objective(model,mean,logvar,hidden,scale,generator,checkpoint_blocks=True):
    noisy,t,target=flow(mean,logvar,scale,generator)
    context=torch.autocast("cuda",dtype=torch.bfloat16) if noisy.is_cuda else contextlib.nullcontext()
    with context:
        pred,aux=model(noisy.to(torch.bfloat16) if noisy.is_cuda else noisy,
                       t.to(torch.bfloat16) if noisy.is_cuda else t,
                       hidden,checkpoint_blocks=checkpoint_blocks)
    mse=(pred.float()-target).square().mean()
    return mse,aux,mse+aux


def lr_factor(update,total,warmup,minimum):
    import math
    if update<=warmup: return update/warmup
    progress=min(1.0,(update-warmup)/max(1,total-warmup))
    return minimum+(1-minimum)*0.5*(1+math.cos(math.pi*progress))
