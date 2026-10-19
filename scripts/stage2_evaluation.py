"""Pure-noise generation, fixed-coordinate geometry metrics and explicit semantic review."""
from stage2_runtime import *
from stage2_cache import cache_base,load_cached
import numpy as np
import torch


def evaluation_protocol(cfg):
    root=manifest_root(cfg)
    ready=read_json(root/"ready.json")
    if digest(root/"evaluation_protocol.json")!=ready["protocol_sha256"]:
        raise RuntimeError("Evaluation protocol changed")
    return read_json(root/"evaluation_protocol.json")


def row_lookup(cfg):
    return {r["uid"]:(r,s) for s in ["train","val","test"] for r in manifests(cfg,s)}


def normalized_mesh(row):
    from stage2_data import load_mesh
    if digest(row["mesh"])!=row["mesh_sha256"]:
        raise RuntimeError("Ground truth payload changed")
    mesh=load_mesh(row["mesh"])
    mesh.vertices=(mesh.vertices-np.array(row["normalization"]["center"]))*row["normalization"]["scale"]
    return mesh


def decode(vae,latent,cfg):
    from hy3dshape.pipelines import export_to_trimesh
    e=cfg["evaluation"]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        decoded=vae.decode(latent.cuda().float())
        output=vae.latents2mesh(decoded,bounds=e["bounds"],mc_level=e["mc_level"],
                    num_chunks=e["num_chunks"],octree_resolution=e["octree_resolution"],
                    mc_algo=e["mc_algo"],enable_pbar=False)
    result=export_to_trimesh(output)[0]
    if result is None or len(result.faces)<4 or len(result.vertices)<4:
        raise ValueError("Empty generated mesh")
    if not np.isfinite(result.vertices).all() or not np.isfinite(result.face_normals).all():
        raise ValueError("Nonfinite mesh")
    if result.faces.min()<0 or result.faces.max()>=len(result.vertices) or result.area<=1e-10:
        raise ValueError("Invalid/degenerate mesh")
    return result


def render_cpu(mesh,folder):
    """Fixed cameras, CPU Agg; no EGL context or extra GPU process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    folder=path(folder);folder.mkdir(parents=True,exist_ok=True)
    vertices=mesh.vertices
    faces=mesh.faces
    # Rendering budget only; exported mesh and metric surface use all triangles.
    if len(faces)>50000:
        faces=faces[np.linspace(0,len(faces)-1,50000,dtype=int)]
    triangles=vertices[faces]
    for azimuth in [0,90,180,270]:
        fig=plt.figure(figsize=(4,4),dpi=128)
        axis=fig.add_subplot(111,projection="3d")
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        intensity=0.35+0.55*np.abs(normals@np.array([0.3,0.5,0.8124]))
        collection=Poly3DCollection(triangles,facecolors=np.repeat(intensity[:,None],3,axis=1),
                                  linewidths=0,edgecolors="none",rasterized=True)
        axis.add_collection3d(collection)
        axis.set(xlim=(-1.01,1.01),ylim=(-1.01,1.01),zlim=(-1.01,1.01))
        axis.set_box_aspect((1,1,1));axis.view_init(elev=20,azim=azimuth,vertical_axis="y")
        axis.set_axis_off()
        fig.savefig(folder/f"{azimuth}.png",bbox_inches="tight",pad_inches=0)
        plt.close(fig)


def load_vae(cfg):
    from stage2_vae import load_part
    return load_part(cfg,"decoder")


def validation(model,cfg,rows,bases,deadline=None):
    from stage2_model import objective
    values=[]
    was_training=model.training;model.eval()
    with torch.no_grad():
        for row in rows:
            if deadline is not None and time.monotonic()>=deadline:
                raise TimeoutError("GPU budget exhausted during validation")
            hidden,mean,logvar=load_cached(cfg,row,"val",bases)
            generator=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"validation",row["uid"]))
            loss,_,_=objective(model,mean.cuda(),logvar.cuda(),hidden.cuda().float(),
                              cfg["cache"]["scale"],generator,False)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite validation loss")
            values.append(float(loss))
    model.train(was_training)
    return {"flow_mse":sum(values)/len(values),"count":len(values),
            "fixed_randomness":True,"moe_aux":"disabled_in_eval"}


def specs(cfg,mode):
    p=evaluation_protocol(cfg);lookup=row_lookup(cfg)
    result=[]
    def add(uid,condition_uid=None,kind="correct"):
        row,split=lookup[uid]
        result.append({"uid":uid,"condition_uid":condition_uid or uid,"kind":kind,
                       "split":split})
    if mode=="monitor":
        for uid in p["train_monitor"]+p["val_monitor"]: add(uid)
        for pair in p["monitor_pairs"]: add(pair["uid"],pair["wrong_uid"],"wrong")
    elif mode in ["val","test"]:
        for uid in p[mode+"_heldout"]: add(uid)
        for i,pair in enumerate(p[mode+"_pairs"]):
            add(pair["uid"],pair["wrong_uid"],f"wrong_{pair['kind']}_{i:02d}")
        if mode=="val":
            for uid in p["empty_val"]:
                add(uid,"__empty__","empty_diagnostic")
    else: raise ValueError(mode)
    return result


def sample(model,hidden,cfg,generator):
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,shift=1.0)
    scheduler.set_timesteps(sigmas=np.linspace(0,1,cfg["evaluation"]["inference_steps"]),
                            device="cuda")
    latent=torch.randn((1,*cfg["cache"]["latent_shape"]),generator=generator,
                       device="cuda",dtype=torch.bfloat16)
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            t=(timestep/scheduler.config.num_train_timesteps).expand(1).to(torch.bfloat16)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                velocity=model.velocity(latent,t,hidden)
            latent=scheduler.step(velocity,timestep,latent).prev_sample
    return latent.float().cpu()/cfg["cache"]["scale"]


def generation_suite(model,cfg,bases,folder,mode,writer=None,step=0,deadline=None):
    """Caller moves optimizer state off GPU. Model returns on CPU after decoder stage."""
    folder=path(folder)
    suite_contract={"config":fingerprint(cfg),"text_contract":bases["text"][1],
                    "shape_contract":bases["shape"][1],"step":step,"mode":mode,
                    "protocol_sha256":digest(manifest_root(cfg)/"evaluation_protocol.json")}
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"contract.json").exists():
        if read_json(folder/"contract.json")!=suite_contract:
            raise RuntimeError("Generation suite/config/cache mismatch")
    else:
        atomic_json(folder/"contract.json",suite_contract)
    if (folder/"complete.json").exists():
        return read_json(folder/"complete.json")
    lookup=row_lookup(cfg)
    jobs=specs(cfg,mode)
    model.eval()
    for item in jobs:
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError("GPU budget exhausted between generation operations")
        label=item["uid"]+"__"+item["kind"]
        latent_file=folder/"latents"/(label+".pt")
        if latent_file.exists(): continue
        condition=item["condition_uid"]
        if condition=="__empty__":
            hidden=torch.load(bases["text"][0]/"empty.pt",map_location="cpu",
                              weights_only=True)["hidden"][None]
        else:
            row,split=lookup[condition]
            hidden,_,_=load_cached(cfg,row,split,bases)
        generator=torch.Generator(device="cuda").manual_seed(
            seed(cfg["seed"],"generation",item["uid"]))
        started=time.monotonic()
        latent=sample(model,hidden.cuda().float(),cfg,generator)
        save_tensor(latent_file,{"latent":latent,"spec":item,"step":step,
                                 "suite_contract":fingerprint(suite_contract)})
        append(folder/"timing.jsonl",{"event":"sampling","label":label,
                                     "seconds":time.monotonic()-started})
    model.cpu();torch.cuda.empty_cache()
    vae=load_vae(cfg)
    records=[]
    from metrics import compare
    for item in jobs:
        if deadline is not None and time.monotonic()>=deadline:
            raise TimeoutError("GPU budget exhausted between generation operations")
        label=item["uid"]+"__"+item["kind"]
        record_file=folder/"records"/(label+".json")
        if record_file.exists():
            records.append(read_json(record_file));continue
        row,_=lookup[item["uid"]]
        record={**item,"step":step,"valid":False}
        try:
            started=time.monotonic()
            latent=torch.load(folder/"latents"/(label+".pt"),weights_only=True)["latent"]
            mesh=decode(vae,latent,cfg)
            target=normalized_mesh(row)
            out=folder/"meshes"/(label+".glb");out.parent.mkdir(parents=True,exist_ok=True)
            mesh.export(out)
            record.update(valid=True,mesh=str(out),geometry=compare(
                mesh,target,seed=seed(cfg["seed"],"metrics",item["uid"]),
                n=cfg["evaluation"]["metric_points"]))
            append(folder/"timing.jsonl",{"event":"decode_and_metrics","label":label,
                                         "seconds":time.monotonic()-started})
            try:
                render_cpu(mesh,folder/"views"/label)
                if not (folder/"targets"/item["uid"]).exists():
                    render_cpu(target,folder/"targets"/item["uid"])
            except Exception as exc:
                record["render_error"]=repr(exc)
            condition=item["condition_uid"]
            condition_prompt="" if condition=="__empty__" else lookup[condition][0]["prompt"]
            atomic_json(folder/"reviews"/(label+".json"),{
                **item,"prompt":condition_prompt,"target_prompt":row["prompt"],
                "subject_correct":None,"key_geometry_correct":None,
                "overall_structure_usable":None,"condition_change_correct":None,
                "pair_geometrically_distinguishable":None,
                "reviewer":None,"notes":None,"status":"pending"})
        except Exception as exc:
            record["error"]=repr(exc)
        atomic_json(record_file,record);records.append(record)
    del vae;torch.cuda.empty_cache()
    atomic_lines(folder/"metrics.jsonl",records)
    correct=[r for r in records if r["kind"]=="correct"]
    valid=[r for r in correct if r["valid"]]
    summary={"step":step,"mode":mode,"correct_count":len(correct),
             "valid_count":len(valid),"valid_fraction":len(valid)/len(correct),
             "semantic_status":"pending_explicit_review",
             "semantic_success":None,"condition_success":None,
             "geometry":{key:float(np.mean([r["geometry"][key] for r in valid]))
                         for key in ["cd_squared_sum","fscore_005"]} if valid else {},
             "by_category":{}}
    for category in sorted({lookup[r["uid"]][0]["category"] for r in correct}):
        group=[r for r in correct if lookup[r["uid"]][0]["category"]==category]
        summary["by_category"][category]={"count":len(group),
                        "valid_fraction":sum(r["valid"] for r in group)/len(group)}
    atomic_json(folder/"complete.json",summary)
    if writer:
        writer.add_scalar(f"generation/{mode}/valid_fraction",summary["valid_fraction"],step)
        for key,value in summary["geometry"].items():
            writer.add_scalar(f"generation/{mode}/{key}",value,step)
        from PIL import Image
        for item in jobs[:4]:
            f=folder/"views"/(item["uid"]+"__"+item["kind"])/"0.png"
            if f.exists():
                writer.add_image(f"generation/{mode}/{item['uid']}",
                                 np.array(Image.open(f).convert("RGB")),step,dataformats="HWC")
        writer.flush()
    return summary


def vae_gate(cfg,folder):
    folder=path(folder); folder.mkdir(parents=True,exist_ok=True)
    base,key=cache_base(cfg,"shape")
    lookup=row_lookup(cfg); protocol=evaluation_protocol(cfg)
    vae=load_vae(cfg); records=[]
    from metrics import compare
    for uid in protocol["vae"]:
        row,split=lookup[uid]
        from stage2_cache import shape_key
        cached=torch.load(base/split/(uid+".pt"),map_location="cpu",weights_only=True)
        if cached["fingerprint"]!=shape_key(row,key): raise RuntimeError("Stale posterior")
        g=torch.Generator().manual_seed(seed(cfg["seed"],"vae_posterior",uid))
        mean,logvar=cached["mean"],cached["logvar"]
        draws={"mean":mean,"posterior":mean+torch.exp(.5*logvar)*torch.randn(
            mean.shape,generator=g)}
        for kind,latent in draws.items():
            f=folder/"records"/(uid+"__"+kind+".json")
            if f.exists(): records.append(read_json(f)); continue
            r={"uid":uid,"category":row["category"],"kind":kind,"valid":False}
            try:
                mesh=decode(vae,latent[None],cfg)
                r.update(valid=True,geometry=compare(mesh,normalized_mesh(row),
                         seed=seed(cfg["seed"],"metrics",uid),
                         n=cfg["evaluation"]["metric_points"]))
                out=folder/"meshes"/(uid+"__"+kind+".glb")
                out.parent.mkdir(parents=True,exist_ok=True);mesh.export(out)
                render_cpu(mesh,folder/"views"/(uid+"__"+kind))
            except Exception as exc: r["error"]=repr(exc)
            atomic_json(f,r);records.append(r)
    del vae;torch.cuda.empty_cache()
    summary={"evaluation_contract":fingerprint(cfg["evaluation"]),
             "shape_contract":key,"protocol_sha256":digest(manifest_root(cfg)/"evaluation_protocol.json"),
             "groups":{},"passed":True}
    for kind in ["mean","posterior"]:
        group=[r for r in records if r["kind"]==kind]
        valid=[r for r in group if r["valid"]]
        fraction=len(valid)/len(group)
        median=float(np.median([r["geometry"]["fscore_005"] for r in valid])) if valid else 0.
        by_category={}
        for category in sorted({r["category"] for r in group}):
            subset=[r for r in group if r["category"]==category]
            scores=[r["geometry"]["fscore_005"] for r in subset if r["valid"]]
            by_category[category]={"count":len(subset),"valid":sum(r["valid"] for r in subset),
                                  "fscore_median":float(np.median(scores)) if scores else None}
        passed=(fraction>=cfg["evaluation"]["vae_valid_min"] and
                median>=cfg["evaluation"]["vae_fscore_median_min"])
        summary["groups"][kind]={"valid_fraction":fraction,"fscore_median":median,
                                "passed":passed,"by_category":by_category}
        summary["passed"] &= passed
    atomic_lines(folder/"metrics.jsonl",records)
    atomic_json(folder/"summary.json",summary)
    if summary["passed"]:
        atomic_json(manifest_root(cfg)/"vae_gate.json",summary)
    return summary


def reviewed_score(folder):
    folder=path(folder); summary=read_json(folder/"complete.json")
    records=read_lines(folder/"metrics.jsonl")
    correct=[r for r in records if r["kind"]=="correct" and
             (summary["mode"]!="monitor" or r["split"]=="val")]
    subject=structure=0
    for item in correct:
        if not item["valid"]: continue
        review=read_json(folder/"reviews"/(item["uid"]+"__correct.json"))
        required=["subject_correct","key_geometry_correct","overall_structure_usable"]
        if not review.get("reviewer") or any(type(review.get(k)) is not bool for k in required):
            raise RuntimeError("Semantic review incomplete: "+item["uid"])
        subject+=review["subject_correct"]
        structure+=review["key_geometry_correct"] and review["overall_structure_usable"]
    score={"structure_fraction":structure/len(correct),"subject_fraction":subject/len(correct),
           "count":len(correct),"valid_fraction":summary["valid_fraction"]}
    atomic_json(folder/"reviewed_score.json",score)
    return score
