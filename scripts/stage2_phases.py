"""Single-GPU decoder phase and GPU-free CPU geometry/render phase."""
from stage2_runtime import *
import argparse,gc,time,traceback
from stage2_schedule import deadline_check


def decoder(cfg,job):
    import torch
    from stage2_evaluation import load_vae,decode
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    cap=job["admission"][0]["limit_bytes"]
    torch.cuda.set_per_process_memory_fraction(min(1,cap/total),0)
    torch.cuda.reset_peak_memory_stats();suite=path(job["suite"])
    protocol=read_json(suite/"suite.json")
    vae=load_vae(cfg)
    for item in protocol["jobs"]:
        label=item["uid"]+"__"+item["kind"];record=suite/"decode_records"/(label+".json")
        if record.exists():
            old=read_json(record)
            if old["suite_contract"]!=protocol["contract"]: raise RuntimeError("decoder contract changed")
            if old["valid"] and digest(old["mesh"])!=old["mesh_sha256"]: raise RuntimeError("decoded mesh changed")
            continue
        deadline_check(30);data=torch.load(suite/"latents"/(label+".pt"),map_location="cpu",weights_only=True)
        if data["suite_contract"]!=protocol["contract"]: raise RuntimeError("latent suite contract changed")
        result={**item,"suite_contract":protocol["contract"],"valid":False}
        try:
            mesh=decode(vae,data["latent"],cfg)
            output=suite/"meshes"/(label+".glb");output.parent.mkdir(parents=True,exist_ok=True)
            mesh.export(output)
            result.update(valid=True,mesh=str(output),mesh_sha256=digest(output))
            del mesh
        except ValueError as exc: result["error"]=repr(exc)
        atomic_json(record,result);del data;gc.collect();torch.cuda.empty_cache()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_reserved();headroom=cap-peak
    atomic_json(suite/"decoder_memory.json",{"reserved_bytes":peak,
        "allocated_bytes":torch.cuda.max_memory_allocated(),"cap_bytes":cap,
        "headroom_bytes":headroom,"passed":headroom>=cfg["fsdp"]["min_headroom_bytes"],**clock()})
    if headroom<cfg["fsdp"]["min_headroom_bytes"]: raise RuntimeError("decoder lacks 2GiB headroom")
    atomic_json(suite/"decode_complete.json",{"jobs":len(protocol["jobs"]),**clock()})


def cpu(cfg,job):
    import numpy as np
    from stage2_evaluation import row_lookup,normalized_mesh,render_cpu
    from stage2_data import load_mesh
    from metrics import compare
    suite=path(job["suite"]);protocol=read_json(suite/"suite.json");lookup=row_lookup(cfg);records=[]
    for item in protocol["jobs"]:
        label=item["uid"]+"__"+item["kind"];file=suite/"records"/(label+".json")
        if file.exists(): records.append(read_json(file));continue
        record=read_json(suite/"decode_records"/(label+".json"));row,_=lookup[item["uid"]]
        if record["valid"]:
            mesh=load_mesh(record["mesh"]);target=normalized_mesh(row)
            record["geometry"]=compare(mesh,target,seed=seed(cfg["seed"],"metrics",item["uid"]),n=cfg["evaluation"]["metric_points"])
            render_cpu(mesh,suite/"views"/label)
            target_dir=suite/"targets"/item["uid"]
            if not target_dir.exists(): render_cpu(target,target_dir)
            del mesh,target
        review=suite/"reviews"/(label+".json")
        if not review.exists():
            atomic_json(review,{**item,"prompt":lookup[item["condition_uid"]][0]["prompt"],
                "target_prompt":row["prompt"],"subject_correct":None,"key_geometry_correct":None,
                "overall_structure_usable":None,"condition_change_correct":None,
                "pair_geometrically_distinguishable":None,"reviewer":None,"notes":None,"status":"pending"})
        record["category"]=row["category"];atomic_json(file,record);records.append(record)
    atomic_lines(suite/"metrics.jsonl",records)
    correct=[r for r in records if r["kind"]=="correct"];valid=[r for r in correct if r["valid"]]
    summary={"correct_count":len(correct),"valid_count":len(valid),
        "valid_fraction":len(valid)/len(correct) if correct else None,
        "semantic_status":"pending_explicit_review","semantic_success":None,
        "geometry":{k:float(np.mean([r["geometry"][k] for r in valid])) for k in ["cd_squared_sum","fscore_005"]} if valid else {},
        "by_category":{}}
    for category in sorted({r["category"] for r in correct}):
        group=[r for r in correct if r["category"]==category];good=[r for r in group if r["valid"]]
        summary["by_category"][category]={"count":len(group),"valid_fraction":len(good)/len(group),
            "fscore_005":float(np.mean([r["geometry"]["fscore_005"] for r in good])) if good else None}
    atomic_json(suite/"complete.json",{**summary,**clock()})


def main():
    p=argparse.ArgumentParser();p.add_argument("--job",required=True);args=p.parse_args()
    job=read_json(args.job);cfg=config(job["config"])
    try:
        if job["phase"]=="decode": decoder(cfg,job)
        elif job["phase"]=="cpu": cpu(cfg,job)
        else: raise ValueError(job["phase"])
    except BaseException as exc:
        atomic_json(path(job["suite"])/(job["phase"]+"_failure.json"),{"error":repr(exc),"traceback":traceback.format_exc(),**clock()})
        raise


if __name__=="__main__": main()

