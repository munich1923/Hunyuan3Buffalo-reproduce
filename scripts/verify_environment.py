import importlib,json,subprocess,sys
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
(ROOT/"environment/environment_verified.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
