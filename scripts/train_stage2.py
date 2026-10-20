"""Full DiT Stage 2 training. Formal start requires a measured, approved new GPU budget."""
from stage2_runtime import *
import math
import random
import resource


def preflight(cfg):
    from stage2_cache import cache_base,load_cached
    bases={kind:cache_base(cfg,kind) for kind in ["text","shape"]}
    splits={split:manifests(cfg,split) for split in ["train","val","test"]}
    for split,rows in splits.items():
        if len(rows)!=cfg["dataset"]["counts"][split]:
            raise RuntimeError("Unexpected split size")
        for row in rows: load_cached(cfg,row,split,bases)
    gate=read_json(manifest_root(cfg)/"vae_gate.json")
    if (not gate.get("passed") or gate["shape_contract"]!=bases["shape"][1] or
        gate.get("evaluation_contract")!=fingerprint(cfg["evaluation"])):
        raise RuntimeError("Current posterior VAE gate has not passed")
    if gate["protocol_sha256"]!=digest(manifest_root(cfg)/"evaluation_protocol.json"):
        raise RuntimeError("VAE gate protocol changed")
    return bases,splits


def validate_budget(cfg,run,contract,budget_file):
    approval=read_json(budget_file)
    if approval.get("approved") is not True:
        raise RuntimeError("Budget has not been approved")
    if approval.get("config_fingerprint")!=fingerprint(cfg):
        raise RuntimeError("Budget/config mismatch")
    probe=path(approval["probe_report"])
    if digest(probe)!=approval["probe_sha256"]:
        raise RuntimeError("Budget probe report changed")
    report=read_json(probe)
    if not report.get("reserve_passed") or report["contract"]!=contract:
        raise RuntimeError("Probe/contract mismatch or insufficient memory reserve")
    seconds=approval.get("gpu_budget_seconds",0)
    if not isinstance(seconds,(int,float)) or seconds<=0 or not approval.get("approved_by"):
        raise RuntimeError("Positive GPU budget and named approval required")
    return approval


def consumed_seconds(run):
    file=path(run)/"gpu_usage.jsonl"
    if not file.exists(): return 0
    events=read_lines(file)
    total=sum(r.get("allocated_seconds",0) for r in events if r["event"]=="end")
    starts=sum(r["event"]=="start" for r in events)
    ends=sum(r["event"]=="end" for r in events)
    if starts>ends+1:
        raise RuntimeError("Unclosed prior GPU accounting; reconcile before resume")
    return total


def train(cfg,args,run,bases,splits,gpu):
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from stage2_cache import load_cached
    from stage2_model import load_model,objective,lr_factor
    from stage2_optimizer import build_optimizer,move_optimizer
    from stage2_checkpoint import (checkpoint_contract,save_checkpoint,load_checkpoint,
                                   restore_rng,rng_state)
    from stage2_evaluation import validation,generation_suite
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    contract=checkpoint_contract(cfg,bases)
    budget=None
    if not args.probe_only:
        if not args.budget: raise RuntimeError("Formal training requires --budget approval.json")
        budget=validate_budget(cfg,run,contract,args.budget)
    settings=cfg["training"]; total=math.ceil(len(splits["train"])/settings["accumulation"])*settings["epochs"]
    model=load_model(cfg)
    model.train()
    optimizer=build_optimizer(model,cfg)
    generator=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"training_flow"))
    step=0;start_epoch=0;offset=0;interval={}
    if args.resume:
        state=load_checkpoint(args.resume,contract)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        pos=state["position"]
        step=pos["update"];start_epoch=pos["epoch"];offset=pos["offset"]
        generator.set_state(state["training_generator"])
        restore_rng(state["rng"]);interval=state["stats"].get("interval",{})
        del state
    writer=SummaryWriter(str(run/"tensorboard"),purge_step=step+1 if args.resume else None)
    atomic_json(run/"provenance.json",provenance(cfg))
    if not (run/"improvement_review.json").exists():
        atomic_json(run/"improvement_review.json",{
            "reviewer":None,"train_generation_improved":None,"val_generation_improved":None,
            "train_condition_improved":None,"val_condition_improved":None,"notes":None})
    started=gpu["leased_at_monotonic"];initial_used=consumed_seconds(run)
    interval=interval or {"samples":0,"mse":0.,"aux":0.,"total":0.,"updates":0,
                          "seconds":0.,"gradient_norm":0.}
    budget_deadline=(started+budget["gpu_budget_seconds"]-initial_used) if budget else None
    probe_times=[];longest=None
    if args.probe_only:
        longest=max(splits["train"],key=lambda row: load_cached(
            cfg,row,"train",bases)[0].shape[1])
    def monitor(label):
        state=rng_state()
        move_optimizer(optimizer,"cpu")
        deadline=(started+budget["gpu_budget_seconds"]-initial_used) if budget else None
        summary=generation_suite(model,cfg,bases,run/"evaluation"/label,"monitor",writer,step,
                                 deadline=deadline)
        model.cuda()
        if settings["optimizer"]=="adamw": move_optimizer(optimizer,"cuda")
        model.train()
        restore_rng(state)
        return summary
    try:
        if step==0 and not args.probe_only: monitor("initial")
        elif args.resume and not args.probe_only:
            monitor(f"step_{step:06d}")
        for epoch in range(start_epoch,settings["epochs"]):
            order=splits["train"].copy()
            random.Random(seed(cfg["seed"],"epoch",epoch)).shuffle(order)
            begin=offset if epoch==start_epoch else 0
            for start in range(begin,len(order),settings["accumulation"]):
                if budget and initial_used+time.monotonic()-started>=budget["gpu_budget_seconds"]:
                    atomic_json(run/"status.json",{"state":"budget_exhausted","last_saved_update":
                        step//settings["checkpoint_every"]*settings["checkpoint_every"],
                        "completed_unsaved_update":step,**clock()})
                    return
                batch=order[start:start+settings["accumulation"]]
                if args.probe_only: batch=[longest]*settings["accumulation"]
                update_start=time.monotonic()
                next_step=step+1
                factor=lr_factor(next_step,total,settings["warmup_updates"],
                                 settings["minimum_lr_factor"])
                for group,peak in zip(optimizer.param_groups,
                                      [settings["dit_lr"],settings["connector_lr"]]):
                    group["lr"]=peak*factor
                optimizer.zero_grad(set_to_none=True)
                mse_sum=aux_sum=total_sum=0.
                for row in batch:
                    h,mean,logvar=load_cached(cfg,row,"train",bases)
                    mse,aux,loss=objective(model,mean.cuda(),logvar.cuda(),h.cuda().float(),
                                          cfg["cache"]["scale"],generator,
                                          settings["activation_checkpointing"])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite train objective")
                    (loss/len(batch)).backward()
                    mse_sum+=float(mse.detach());aux_sum+=float(aux.detach());total_sum+=float(loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["clip_norm"],
                                                    error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed=time.monotonic()-update_start
                step=next_step
                interval["samples"]+=len(batch);interval["mse"]+=mse_sum
                interval["aux"]+=aux_sum;interval["total"]+=total_sum
                interval["updates"]+=1;interval["seconds"]+=elapsed
                interval["gradient_norm"]+=float(norm)
                position={"update":step,"epoch":epoch,"offset":start+len(batch)}
                if position["offset"]==len(order):
                    position.update(epoch=epoch+1,offset=0)
                if args.probe_only:
                    probe_times.append(elapsed)
                    append(run/"probe_updates.jsonl",{"update":step,"seconds":elapsed,
                        "flow_mse":mse_sum/len(batch),"moe_aux":aux_sum/len(batch),
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
                    print("PROBE_UPDATE",step,elapsed,flush=True)
                    if len(probe_times)>=settings["probe_updates"]:
                        io=save_checkpoint(run/"probe_checkpoint",model,optimizer,position,contract,
                                           generator,{"probe_only":True})
                        from stage2_evaluation import sample,load_vae,decode
                        torch.cuda.synchronize();validation_start=time.monotonic()
                        validation_probe=validation(model,cfg,splits["val"][:8],bases)
                        torch.cuda.synchronize();validation_seconds=time.monotonic()-validation_start
                        sampling_generator=torch.Generator(device="cuda").manual_seed(
                            seed(cfg["seed"],"probe_sampling",longest["uid"]))
                        hidden,_,_=load_cached(cfg,longest,"train",bases)
                        model.eval();torch.cuda.synchronize();sample_start=time.monotonic()
                        latent=sample(model,hidden.cuda().float(),cfg,sampling_generator)
                        torch.cuda.synchronize();sample_seconds=time.monotonic()-sample_start
                        move_optimizer(optimizer,"cpu");model.cpu();torch.cuda.empty_cache()
                        vae=load_vae(cfg);torch.cuda.synchronize();decode_start=time.monotonic()
                        mesh=decode(vae,latent,cfg)
                        torch.cuda.synchronize();decode_seconds=time.monotonic()-decode_start
                        mesh.export(run/"probe_generation.glb")
                        del vae;torch.cuda.empty_cache()
                        memory=torch.cuda.max_memory_allocated()
                        reserved=torch.cuda.max_memory_reserved()
                        total_memory=torch.cuda.get_device_properties(0).total_memory
                        report={"contract":contract,"gpu":gpu,"updates":len(probe_times),
                          "accumulation":settings["accumulation"],"condition_uid":longest["uid"],
                          "update_seconds":probe_times,"peak_allocated_bytes":memory,
                          "peak_reserved_bytes":reserved,"total_device_bytes":total_memory,
                          "reserve_passed":reserved<=(gpu["process_memory_limit_mib"]*1024**2 if gpu.get("shared") else total_memory*(1-settings["reserve_fraction"])),
                          "cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "checkpoint_io":io,"sampling_seconds_one_long_condition":sample_seconds,
                          "decode_seconds_one_sample":decode_seconds,
                          "validation_probe_count":validation_probe["count"],
                          "validation_probe_seconds":validation_seconds,
                          "estimated_500_object_validation_seconds":validation_seconds/validation_probe["count"]*len(splits["val"]),
                          "estimated_train_compute_seconds":
                            float(np.mean(probe_times[1:] or probe_times))*total,
                          "estimate_excludes":"cache encoding, validation, sampling, decoding, review and checkpoint overhead",
                          **clock()}
                        atomic_json(run/"probe_report.json",report)
                        print(canonical(report),flush=True)
                        return
                    continue
                if step%settings["log_every"]==0 or step==total:
                    count=interval["samples"]
                    stats={"update":step,"epoch":epoch+1,"samples":count,
                           "flow_mse":interval["mse"]/count,"moe_aux":interval["aux"]/count,
                           "total_loss":interval["total"]/count,
                           "gradient_norm":interval["gradient_norm"]/interval["updates"],
                           "dit_lr":optimizer.param_groups[0]["lr"],
                           "connector_lr":optimizer.param_groups[1]["lr"],
                           "samples_per_second":count/interval["seconds"],
                           "interval_seconds":interval["seconds"],
                           "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                           "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                           "cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    append(run/"train.jsonl",stats);print(canonical(stats),flush=True)
                    for key,value in stats.items():
                        if isinstance(value,(int,float)): writer.add_scalar("train/"+key,value,step)
                    writer.flush()
                    interval={"samples":0,"mse":0.,"aux":0.,"total":0.,"updates":0,
                              "seconds":0.,"gradient_norm":0.}
                if position["offset"]==0:
                    value=validation(model,cfg,splits["val"],bases,deadline=budget_deadline)
                    append(run/"validation.jsonl",{"update":step,**value})
                    writer.add_scalar("validation/flow_mse",value["flow_mse"],step)
                    writer.flush()
                save_now=step%settings["checkpoint_every"]==0 or step==total
                if save_now:
                    # Fixed validation needed for candidate tie-break, independent of train randomness.
                    val=(value if position["offset"]==0 else
                         validation(model,cfg,splits["val"],bases,deadline=budget_deadline))
                    atomic_json(run/"evaluation"/f"step_{step:06d}"/"validation.json",val)
                    name=f"step_{step:06d}"+("_final" if step==total else "")
                    io=save_checkpoint(run/"checkpoints"/name,model,optimizer,position,contract,
                                       generator,{"interval":interval,"validation":val})
                    append(run/"checkpoint_io.jsonl",{"update":step,**io})
                    monitor(f"step_{step:06d}")
            offset=0
        atomic_json(run/"status.json",{"state":"training_complete",
                    "updates":step,"semantic_success":"not_yet_evaluated",**clock()})
    finally:
        writer.close()


def main():
    p=parser(__doc__,gpu=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--probe-only",action="store_true")
    p.add_argument("--budget")
    p.add_argument("--resume")
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args();cfg=config(args.config);run=run_path(cfg,args.run_id)
    bases,splits=preflight(cfg)
    if args.preflight_only:
        print("PREFLIGHT_OK");return
    if not args.probe_only:
        if not args.budget: p.error("Formal training requires an approved --budget file")
        from stage2_checkpoint import checkpoint_contract
        validate_budget(cfg,run,checkpoint_contract(cfg,bases),args.budget)
    if not args.resume and (run/"provenance.json").exists():
        raise FileExistsError("Run was already initialized; resume a complete checkpoint or use a new run-id")
    if args.resume and not path(args.resume).is_relative_to(run/"checkpoints"):
        raise ValueError("Resume checkpoint must belong to the same run")
    run.mkdir(parents=True,exist_ok=True)
    try:
        with lock("environment/stage2_train.lock"),gpu_lease(args.gpu,run) as gpu:
            train(cfg,args,run,bases,splits,gpu)
    except BaseException as exc:
        atomic_json(run/"status.json",{"state":"budget_exhausted" if isinstance(exc,TimeoutError)
                    else "failed","error":repr(exc),"no_emergency_checkpoint_written":True,
                    "resume":"latest complete saved checkpoint",**clock()})
        raise

if __name__=="__main__": main()
