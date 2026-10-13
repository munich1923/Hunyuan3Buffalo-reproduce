import json,torch
from remote_runtime import ROOT,configure
from edit_model import Editor,load_dit
configure()
folder=ROOT/"results/interface_smoke"
shape=torch.load(folder/"shape.pt",map_location="cpu",weights_only=True)
h=torch.load(folder/"vlm.pt",map_location="cpu",weights_only=True)["hidden"][None].cuda().float()
scale=shape["scale_factor"];source=shape["source"][None].cuda().to(torch.bfloat16)*scale;target=shape["target"][None].cuda().to(torch.bfloat16)*scale
torch.manual_seed(777);noise=torch.randn_like(target);t=torch.tensor([0.5],device="cuda",dtype=torch.bfloat16)
editor=Editor(load_dit()).cuda()
with torch.autocast("cuda",dtype=torch.bfloat16):
 pred=editor((noise+target)/2,t,h,source)
 loss=(pred.float()-(target-noise).float()).square().mean()
 loss.backward()
assert pred.shape==target.shape and torch.isfinite(pred).all()
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in editor.connector.parameters())
assert all(p.grad is None for p in editor.dit.parameters())
report={"pred_shape":list(pred.shape),"loss":float(loss.detach()),"connector_gradient":True,"backbone_frozen":True,"synthetic":True,"note":"Random connector, interface pass only."}
(folder/"forward_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
