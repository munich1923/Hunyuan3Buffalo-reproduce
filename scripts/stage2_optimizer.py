"""AdamW backend plus synchronous CPU optimizer-state offload; no DeepSpeed dependency."""
import torch


class CPUOffloadAdamW:
    """Keep FP32 master parameters and Adam states on host; explicit copies per update."""
    def __init__(self,groups,**kwargs):
        self.gpu_groups=groups
        self.cpu_groups=[]
        for group in groups:
            self.cpu_groups.append({**{k:v for k,v in group.items() if k!="params"},
                                    "params":[torch.nn.Parameter(p.detach().cpu().clone())
                                              for p in group["params"]]})
        self.inner=torch.optim.AdamW(self.cpu_groups,foreach=False,**kwargs)
        self.param_groups=self.inner.param_groups

    def zero_grad(self,set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for group in self.gpu_groups:
            for p in group["params"]:
                if set_to_none: p.grad=None
                elif p.grad is not None: p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for gpu,cpu in zip(self.gpu_groups,self.param_groups):
            for original,master in zip(gpu["params"],cpu["params"]):
                master.grad=None if original.grad is None else original.grad.detach().cpu()
        self.inner.step()
        for gpu,cpu in zip(self.gpu_groups,self.param_groups):
            for original,master in zip(gpu["params"],cpu["params"]):
                original.copy_(master.to(original.device))

    def state_dict(self):
        return {"backend":"cpu_adamw","inner":self.inner.state_dict()}

    def load_state_dict(self,state):
        if state["backend"]!="cpu_adamw": raise ValueError("Optimizer backend changed")
        # Model was restored before this call; host parameters must match it.
        for gpu,cpu in zip(self.gpu_groups,self.param_groups):
            for original,master in zip(gpu["params"],cpu["params"]):
                master.data.copy_(original.detach().cpu())
        self.inner.load_state_dict(state["inner"])

    def move_state(self,device):
        if str(device)!="cpu": return


def build_optimizer(model,cfg):
    t=cfg["training"]
    groups=[{"params":list(model.dit.parameters()),"lr":t["dit_lr"],"name":"dit"},
            {"params":list(model.connector.parameters()),"lr":t["connector_lr"],"name":"connector"}]
    kwargs={"betas":tuple(t["betas"]),"eps":t["eps"],"weight_decay":t["weight_decay"]}
    if t["optimizer"]=="adamw":
        return torch.optim.AdamW(groups,foreach=False,**kwargs)
    if t["optimizer"]=="cpu_adamw":
        return CPUOffloadAdamW(groups,**kwargs)
    raise ValueError("Unknown optimizer backend")


def move_optimizer(optimizer,device):
    if isinstance(optimizer,CPUOffloadAdamW): return
    for state in optimizer.state.values():
        for key,value in state.items():
            # Adam step scalar stays on CPU in non-capturable AdamW.
            if torch.is_tensor(value) and key!="step":
                state[key]=value.to(device)
