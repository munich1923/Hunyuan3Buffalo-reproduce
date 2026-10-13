from pathlib import Path
import json,torch
p=Path("/root/userdata/zc/hunyuan3d-buffalo-1.0-editing");report={"shape":[],"vlm":[]}
for split in ["train","val"]:
 for f in (p/"assets/cache/shape"/split).glob("*.pt"):
  x=torch.load(f,map_location="cpu",weights_only=True)
  for role in ["source","target"]:
   z=x[role].float();assert z.shape==(4096,64) and torch.isfinite(z).all()
   report["shape"].append({"split":split,"id":f.stem,"role":role,"shape":list(z.shape),"std":float(z.std()),"scale":x["scale_factor"]})
 for f in (p/"assets/cache/vlm"/split).glob("*.pt"):
  x=torch.load(f,map_location="cpu",weights_only=True)
  for role in ["hidden","noop_hidden"]:
   h=x[role].float();assert h.ndim==2 and h.shape[-1]==3584 and torch.isfinite(h).all()
   report["vlm"].append({"split":split,"id":f.stem,"role":role,"shape":list(h.shape),"std":float(h.std())})
report["passed"]=True
(p/"results/summaries/diagnostic_cache_integrity.json").write_text(json.dumps(report,indent=2)+"\n")
print("PASS",len(report["shape"]),"shape",len(report["vlm"]),"VLM tensors")
