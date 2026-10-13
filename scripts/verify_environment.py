import importlib,json,subprocess,sys,hashlib
from remote_runtime import ROOT,configure
configure();sys.path.insert(0,str(ROOT/"src/Hunyuan3D-2.1/hy3dshape"))
modules=["torch","torchvision","transformers","diffusers","trimesh","scipy","pyrender","torch_cluster","hy3dshape.pipelines","hy3dshape.models.denoisers.hunyuandit"]
report={"python":sys.version,"imports":{}}
for module in modules:
    m=importlib.import_module(module);report["imports"][module]=getattr(m,"__version__","imported")
import torch
assert torch.__version__.startswith("2.5.1"),torch.__version__
from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
assert (ROOT/"models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt").exists()
required=[
    ROOT/"models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt",
    ROOT/"models/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/model.fp16.ckpt",
]
receipts=[json.loads(p.read_text()) for p in (ROOT/"logs/downloads").glob("*.json")]
for path in required:
    receipt=next((r for r in receipts if r.get("local_path")==str(path) and r.get("status")=="verified"),None)
    if receipt is None:raise RuntimeError("No verified download receipt for "+str(path))
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    if h.hexdigest()!=receipt["expected_sha256"]:raise RuntimeError("Weight changed since download: "+str(path))
report["shape_weights_rehashed"]=True
import yaml
from hy3dshape.pipelines import instantiate_from_config
checkpoint=torch.load(required[0],map_location="cpu",weights_only=True)
if "conditioner" not in checkpoint:raise RuntimeError("Packaged DINO weights missing")
config=yaml.safe_load((ROOT/"models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/config.yaml").read_text())
conditioner=instantiate_from_config(config["conditioner"])
loaded=conditioner.load_state_dict(checkpoint["conditioner"],strict=True)
report["dino"]={"packaged":True,"state_tensors":len(checkpoint["conditioner"]),"strict_load_missing":list(loaded.missing_keys),"strict_load_unexpected":list(loaded.unexpected_keys)}
del conditioner,checkpoint
(ROOT/"environment/environment_verified.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
