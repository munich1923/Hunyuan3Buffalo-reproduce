"""Meaningful tensor-contract checks using a small real upstream DiT, no model weights."""
import sys,torch
from remote_runtime import ROOT
from edit_model import Editor,inject_lora
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
def main():
    torch.manual_seed(3)
    d=HunYuanDiTPlain(input_size=8,in_channels=4,hidden_size=128,context_dim=16,depth=3,num_heads=2,text_len=6,use_attention_pooling=False,num_moe_layers=0,qkv_bias=False).cuda().to(torch.bfloat16).eval()
    for p in d.parameters():p.requires_grad_(False)
    e=Editor(d).cuda()
    # Replace connector only to reduce hidden feature width in this synthetic check.
    from edit_model import Connector
    e.connector=Connector(hidden=12,context=16,tokens=6).cuda()
    noise=torch.randn(1,8,4,device="cuda",dtype=torch.bfloat16);source=torch.randn_like(noise);t=torch.tensor([0.5],device="cuda",dtype=torch.bfloat16)
    h=torch.randn(1,11,12,device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        c=e.connector(h)
        native=d(noise,t,{"main":c})
        wrapped=e.velocity(noise,t,c,None,False)
        assert torch.allclose(native,wrapped,atol=0.02,rtol=0.02),"Wrapper must match native path when no source is used"
        pred=e(noise,t,h,source)
        assert pred.shape==noise.shape and torch.isfinite(pred).all()
        loss=pred.float().square().mean();loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in e.connector.parameters()),"Frozen backbone must still propagate connector gradients"
    assert all(p.grad is None for p in d.parameters()),"Frozen backbone must not accumulate parameter gradients"
    old=wrapped.detach().clone();names=inject_lora(d)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):new=e.velocity(noise,t,c,None,False)
    assert torch.allclose(old,new,atol=0.02,rtol=0.02),"Zero-initialized LoRA must preserve base predictions"
    d.zero_grad(set_to_none=True);e.connector.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):e(noise,t,h,source).float().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in d.named_parameters() if ".b.weight" in n)
    print("PASS native equivalence, source slicing, connector gradients, frozen weights, LoRA initialization and gradients")
if __name__=="__main__":main()
