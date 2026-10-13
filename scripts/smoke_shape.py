"""Synthetic geometry smoke test; this is NOT a data or editing-quality gate."""
import json,sys
import torch,trimesh
from remote_runtime import ROOT,configure
configure()
from prepare_dataset import normalize_pair
from render_views import render
from cache_shape import surface
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh
from metrics import compare
folder=ROOT/"results/interface_smoke";folder.mkdir(parents=True,exist_ok=True)
source=trimesh.creation.box(extents=[1,0.6,0.5])
addition=trimesh.creation.icosphere(subdivisions=2,radius=0.15);addition.apply_translation([0,0.35,0])
target=trimesh.util.concatenate([source,addition])
source,target,center,scale=normalize_pair(source,target)
source.export(folder/"source.glb");target.export(folder/"target.glb")
views=render(source,folder/"source_views");render(target,folder/"target_views")
vae=ShapeVAE.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float16).eval()
with torch.no_grad():
 src=vae.encode(surface(folder/"source.glb",777),sample_posterior=False)
 tar=vae.encode(surface(folder/"target.glb",778),sample_posterior=False)
 decoded=vae.decode(src)
 recon=export_to_trimesh(vae.latents2mesh(decoded,bounds=1.01,mc_level=0,num_chunks=10000,octree_resolution=256,mc_algo="mc",enable_pbar=False))[0]
assert torch.isfinite(src).all() and torch.isfinite(tar).all()
assert recon is not None and len(recon.faces)>0
recon.export(folder/"source_reconstruction.glb");render(recon,folder/"reconstruction_views")
torch.save({"source":src[0].cpu(),"target":tar[0].cpu(),"scale_factor":float(vae.scale_factor)},folder/"shape.pt")
report={"source_views":views,"instruction":"Add a small sphere on top of the box.","latent_shape":list(src.shape),"reconstruction":compare(recon,source),"synthetic":True,"note":"Interface smoke only, does not certify dataset reconstruction or editing ability."}
(folder/"shape_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
