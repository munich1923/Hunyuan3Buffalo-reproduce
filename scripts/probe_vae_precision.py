"""Diagnose precision/domain/scale before the real-data reconstruction gate."""
import json,torch,trimesh
from remote_runtime import ROOT,configure
configure()
from cache_shape import surface,ShapeVAE,export_to_trimesh
from prepare_dataset import mesh,normalize_pair
from render_views import render
from metrics import compare
folder=ROOT/"results/vae_precision_probe";folder.mkdir(parents=True,exist_ok=True)
benchpaths=sorted((ROOT/"assets/Edit3D-Bench/data/glbs/add").glob("*.glb"))
assert benchpaths
bench=mesh(benchpaths[0]);bench,_,_,_=normalize_pair(bench,bench)
bench.export(folder/"bench_source.glb")
box=mesh(ROOT/"results/interface_smoke/source.glb")
vae=ShapeVAE.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float32).eval()
reports=[]
for label,obj,extent in [("box_float32_autocast",box,1.5),("bench_float32_autocast",bench,1.5),("bench_extent1_9_diagnostic",bench,1.9)]:
 m=obj.copy();m.apply_scale(extent/1.5);p=folder/(label+"_source.glb");m.export(p);render(m,folder/(label+"_source_views"))
 try:
  with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
   z=vae.encode(surface(p,777,dtype=torch.float32),sample_posterior=False)
   decoded=vae.decode(z)
   recon=export_to_trimesh(vae.latents2mesh(decoded,bounds=1.01,mc_level=0,num_chunks=10000,octree_resolution=256,mc_algo="mc",enable_pbar=False))[0]
  assert recon is not None and len(recon.faces)>0
  recon.export(folder/(label+"_reconstruction.glb"));render(recon,folder/(label+"_reconstruction_views"))
  report={"label":label,"source":str(p),"latent_std":float(z.float().std()),"metrics":compare(recon,m)}
 except Exception as e:report={"label":label,"error":repr(e)}
 reports.append(report);(folder/"reports.json").write_text(json.dumps(reports,indent=2)+"\n");print(json.dumps(report),flush=True)
