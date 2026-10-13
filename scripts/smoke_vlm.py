import json,torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from remote_runtime import ROOT,configure
from cache_vlm import hidden
configure()
folder=ROOT/"results/interface_smoke"
row=json.loads((folder/"shape_report.json").read_text())
modelpath=ROOT/"models/Qwen2.5-VL-7B-Instruct"
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).cuda().eval()
processor=AutoProcessor.from_pretrained(modelpath,min_pixels=256*28*28,max_pixels=512*28*28,local_files_only=True)
h=hidden(model,processor,row["source_views"],row["instruction"])
assert h.ndim==2 and h.shape[1]==3584 and torch.isfinite(h).all()
torch.save({"hidden":h},folder/"vlm.pt")
report={"hidden_shape":list(h.shape),"layer":-1,"generated_answer":False,"synthetic":True}
(folder/"vlm_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
