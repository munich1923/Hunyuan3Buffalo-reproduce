import torch,json
from remote_runtime import ROOT,configure
configure()
report={"passed":True,"counts":{}}
for kind in ["shape","vlm"]:
 for split in ["train","val","test"]:
  count=0
  for f in (ROOT/"assets/cache"/kind/split).glob("*.pt"):
   x=torch.load(f,map_location="cpu",weights_only=True)
   for key,value in x.items():
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all(),str(f)+key
   if kind=="shape":assert x["source"].shape==x["target"].shape==(4096,64)
   else:assert x["hidden"].shape[-1]==x["noop_hidden"].shape[-1]==3584
   if kind=="vlm" and split=="test":assert x["shuffled_hidden"] is not None
   count+=1
  report["counts"][kind+"/"+split]=count
  assert count=={"train":1024,"val":128,"test":40}[split]
(ROOT/"results/summaries/full_cache_integrity.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
