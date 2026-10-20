"""Offline frozen VAE posterior parameters; no source condition is created."""
from stage2_runtime import *
from stage2_cache import cache_base,shape_key


def main():
    p=parser(__doc__,gpu=True)
    p.add_argument("--split",choices=["all","train","val","test"],default="all")
    p.add_argument("--limit",type=int)
    args=p.parse_args(); cfg=config(args.config)
    with lock("environment/stage2_shape.lock"),gpu_lease(
            args.gpu,path(cfg["output_root"])/"preparation"):
        import torch
        from stage2_data import load_mesh,surface
        from stage2_vae import load_part
        from hy3dshape.models.autoencoders.model import DiagonalGaussianDistribution
        base,key=cache_base(cfg,"shape",True)
        vae=load_part(cfg,"encoder")
        vae.requires_grad_(False)
        if abs(float(vae.scale_factor)-cfg["cache"]["scale"])>1e-12:
            raise RuntimeError("VAE scale mismatch")
        splits=["train","val","test"] if args.split=="all" else [args.split]
        processed=0
        for split in splits:
            rows=manifests(cfg,split)
            for index,row in enumerate(rows[:args.limit] if args.limit else rows):
                target=base/split/(row["uid"]+".pt")
                expected=shape_key(row,key)
                if target.exists():
                    old=torch.load(target,map_location="cpu",weights_only=True)
                    if old.get("fingerprint")!=expected: raise RuntimeError("Stale posterior")
                    continue
                if digest(row["mesh"])!=row["mesh_sha256"]:
                    raise RuntimeError("Mesh payload changed")
                mesh=load_mesh(row["mesh"]); norm=row["normalization"]
                import numpy as np
                mesh.vertices=(mesh.vertices-np.asarray(norm["center"]))*norm["scale"]
                points=torch.from_numpy(surface(mesh,cfg["dataset"]["sample_points"],
                                                 row["surface_seed"]))[None].cuda()
                # Upstream encoder uses randperm and randomized FPS in addition to surface sampling.
                encoder_seed=seed(row["surface_seed"],"vae_encoder")
                with torch.random.fork_rng(devices=[0]),torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    torch.manual_seed(encoder_seed)
                    torch.cuda.manual_seed(encoder_seed)
                    latents,_=vae.encoder(points[:,:,:3],points[:,:,3:])
                    moments=vae.pre_kl(latents)
                    posterior=DiagonalGaussianDistribution(moments,feat_dim=-1)
                mean=posterior.mean[0].float().cpu()
                logvar=posterior.logvar[0].float().cpu()
                if list(mean.shape)!=cfg["cache"]["latent_shape"] or not all(
                        torch.isfinite(v).all() for v in [mean,logvar]):
                    raise RuntimeError("Invalid VAE posterior")
                save_tensor(target,{"uid":row["uid"],"mean":mean,"logvar":logvar,
                    "scale":cfg["cache"]["scale"],"fingerprint":expected,"contract":key,
                    "surface_seed":row["surface_seed"],"encoder_seed":encoder_seed})
                processed+=1
                if (index+1)%100==0: print("SHAPE",split,index+1,flush=True)
        atomic_json(base/"progress.json",{"newly_processed":processed,
                    "partial":args.limit is not None,"command":sys.argv,**clock()})

if __name__=="__main__": main()
