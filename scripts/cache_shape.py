import argparse,json,sys,hashlib
import numpy as np
import torch
import trimesh
from remote_runtime import ROOT,configure
configure();sys.path.insert(0,str(ROOT/"src/Hunyuan3D-2.1/hy3dshape"))
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh
def surface(path,seed):
    m=trimesh.load(path,force="mesh",process=False)
    torch.manual_seed(seed);np.random.seed(seed);pts,faces=trimesh.sample.sample_surface(m,81920)
    # Upstream point_feats=4 expects normals plus the sharp-edge marker.
    return torch.tensor(np.concatenate([pts,m.face_normals[faces],np.zeros((len(pts),1))],axis=1),dtype=torch.float16,device="cuda")[None]
def main():
    p=argparse.ArgumentParser();p.add_argument("--split",default="all");p.add_argument("--limit",type=int);a=p.parse_args()
    vae=ShapeVAE.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float16).eval()
    for x in vae.parameters():x.requires_grad_(False)
    splits=["train","val","test"] if a.split=="all" else [a.split]
    gatepath=ROOT/"environment/vae_gate.json"
    gate=gatepath.exists() and json.loads(gatepath.read_text()).get("passed",False)
    for split in splits:
        rows=[json.loads(l) for l in (ROOT/"assets/manifests"/(split+"_prepared.jsonl")).read_text().splitlines()]
        if a.limit:rows=rows[:a.limit]
        for i,row in enumerate(rows):
            out=ROOT/"assets/cache/shape"/split/(row["id"]+".pt");out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():continue
            with torch.no_grad():
                src=vae.encode(surface(row["source_normalized"],int(hashlib.sha256((row["id"]+"/source").encode()).hexdigest()[:8],16)),sample_posterior=False)
                tar=vae.encode(surface(row["target_normalized"],int(hashlib.sha256((row["id"]+"/target").encode()).hexdigest()[:8],16)),sample_posterior=False)
            if not torch.isfinite(src).all() or not torch.isfinite(tar).all():raise RuntimeError("Nonfinite VAE latent")
            if not gate:
                with torch.no_grad():
                    decoded=vae.decode(src)
                    output=vae.latents2mesh(decoded,bounds=1.01,mc_level=0,num_chunks=10000,octree_resolution=256,mc_algo="mc",enable_pbar=False)
                recon=export_to_trimesh(output)[0]
                if recon is None or len(recon.faces)==0:raise RuntimeError("VAE reconstruction gate failed: empty mesh")
                folder=ROOT/"results/vae_gate";folder.mkdir(parents=True,exist_ok=True);recon.export(folder/"source_reconstruction.glb")
                from metrics import compare
                report=compare(recon,trimesh.load(row["source_normalized"],force="mesh"))
                report["id"]=row["id"];report["scale_factor"]=float(vae.scale_factor);report["source"]=row["source_normalized"];report["latent_shape"]=list(src.shape)
                # Gross failure gate in coordinates with source extent 1.5.
                report["passed"]=report["fscore_005"]>=0.5
                (ROOT/"environment/vae_gate.json").write_text(json.dumps(report,indent=2)+"\n")
                if not report["passed"]:raise RuntimeError("VAE reconstruction gate failed: "+str(report))
                gate=True
            torch.save({"source":src[0].cpu(),"target":tar[0].cpu(),"scale_factor":float(vae.scale_factor)},out)
            if i%10==0:print("VAE",split,i,list(src.shape),flush=True)
if __name__=="__main__":main()
