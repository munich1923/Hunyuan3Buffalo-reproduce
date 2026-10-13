"""Frozen Hunyuan3D backbone, trainable semantic connector and optional attention LoRA."""
import math,sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from remote_runtime import ROOT
sys.path.insert(0,str(ROOT/"src/Hunyuan3D-2.1/hy3dshape"))
class Connector(nn.Module):
    def __init__(self,hidden=3584,context=1024,tokens=1370):
        super().__init__();self.tokens=tokens
        self.net=nn.Sequential(nn.Linear(hidden,context),nn.GELU(),nn.Linear(context,context))
    def forward(self,h):
        h=F.adaptive_avg_pool1d(h.transpose(1,2),self.tokens).transpose(1,2)
        return self.net(h)
class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.a=nn.Linear(base.in_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5));nn.init.zeros_(self.b.weight)
        self.to(device=base.weight.device,dtype=base.weight.dtype)
        self.a.float();self.b.float()
        for p in self.base.parameters():p.requires_grad_(False)
    def forward(self,x):return self.base(x)+self.b(self.a(x))*self.scale
def inject_lora(model,rank=8,alpha=16):
    names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and any(p in name for p in [".attn1.",".attn2."]) and name.rsplit(".",1)[-1] in ["to_q","to_k","to_v","out_proj"]:
            parent=model.get_submodule(name.rsplit(".",1)[0])
            setattr(parent,name.rsplit(".",1)[1],LoRALinear(module,rank,alpha));names.append(name)
    if not names:raise RuntimeError("No attention projection modules found for LoRA")
    return names
def load_dit(device="cuda"):
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    model=HunYuanDiTPlain.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),device=device,dtype=torch.bfloat16,use_safetensors=False)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model
class Editor(nn.Module):
    def __init__(self,dit,lora=False):
        super().__init__();self.dit=dit
        self.connector=Connector(context=dit.context_dim,tokens=dit.text_len).to(next(dit.parameters()).device)
        self.lora_names=inject_lora(dit) if lora else []
    def velocity(self,noisy,t,condition,source=None,checkpoint_blocks=False):
        """Source token updates are discarded after this call, preserving a fixed source input."""
        n=noisy.shape[1];raw=torch.cat([source,noisy],dim=1) if source is not None else noisy
        d=self.dit
        if d.use_pos_emb or d.use_attention_pooling or d.with_decoupled_ca:
            raise RuntimeError("Pilot wrapper expects the pinned 2.1 geometry checkpoint configuration")
        c=d.t_embedder(t,condition=None)
        x=d.x_embedder(raw);x=torch.cat([c,x],dim=1)
        skips=[]
        for i,block in enumerate(d.blocks):
            skip=None if i<=d.depth//2 else skips.pop()
            if checkpoint_blocks and torch.is_grad_enabled():
                x=checkpoint(block,x,c,condition,skip,use_reentrant=False)
            else:x=block(x,c,condition,skip)
            if i<d.depth//2:skips.append(x)
        result=d.final_layer(x)
        return result[:,-n:]
    def forward(self,noisy,t,h,source=None,checkpoint_blocks=True):
        return self.velocity(noisy,t,self.connector(h),source,checkpoint_blocks)
    def adapter_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if k.startswith("connector.") or ".a.weight" in k or ".b.weight" in k}
    def load_adapter(self,path):
        state=torch.load(path,map_location="cpu",weights_only=True)
        result=self.load_state_dict(state["adapter"],strict=False)
        if result.unexpected_keys:raise RuntimeError(result.unexpected_keys)
        return state
