"""Stage 2 FSDP parent: bounded verification, restartable phases, shared 16 GPU-hour ledger."""
from stage2_runtime import *
import argparse,contextlib,signal,traceback
from stage2_schedule import Ledger,cpu_sampler_checks


def prepare(cfg,run):
    from stage2_cache import cache_base,load_cached
    from stage2_checkpoint import checkpoint_contract
    bases={k:cache_base(cfg,k) for k in ["text","shape"]}
    splits={s:manifests(cfg,s) for s in ["train","val","test"]}
    if any(len(splits[s])!=cfg["dataset"]["counts"][s] for s in splits): raise RuntimeError("manifest count changed")
    gate=read_json(manifest_root(cfg)/"vae_gate.json")
    if not gate.get("passed") or gate["shape_contract"]!=bases["shape"][1] or gate["evaluation_contract"]!=fingerprint(cfg["evaluation"]):
        raise RuntimeError("VAE gate contract changed")
    if gate["protocol_sha256"]!=digest(manifest_root(cfg)/"evaluation_protocol.json"): raise RuntimeError("VAE protocol changed")
    previous=read_json("results/stage2_nano10k/fsdp_gpu2_gpu5_20261021/prepared.json")
    if previous["validated_objects"]!=11000 or any(previous["bases"][k][1]!=bases[k][1] for k in bases):
        raise RuntimeError("prior full cache verification cannot be reused")
    if previous["contract"]["dataset"]!=read_json(manifest_root(cfg)/"ready.json"):
        raise RuntimeError("dataset differs from prior full preflight")
    cutoff=dt.datetime.fromisoformat(previous["utc"]).timestamp()
    stats=[];changed=[]
    receipt=path("environment/cache/stage2/fsdp_preflight_receipt.json")
    for split,rows in splits.items():
        for row in rows:
            for kind in ["text","shape"]:
                file=path(bases[kind][0]/split/(row["uid"]+".pt"));s=file.stat()
                stats.append([str(file),s.st_size,s.st_mtime_ns])
                if s.st_size<=0: raise RuntimeError("empty cache file")
                if s.st_mtime>cutoff: changed.append((row,split))
    stat_hash=fingerprint(stats)
    cached=read_json(receipt) if receipt.exists() else None
    if cached and cached.get("stat_fingerprint")==stat_hash:
        changed=[]
    else:
        # Only changed files need content loading. Unchanged payloads retain the
        # prior successful 11k preflight evidence, not another encoding/gate.
        for row,split in {r[0]["uid"]:r for r in changed}.values(): load_cached(cfg,row,split,bases)
    report={"stat_fingerprint":stat_hash,"file_count":len(stats),"validated_objects":11000,
            "prior_full_preflight":"results/stage2_nano10k/fsdp_gpu2_gpu5_20261021/prepared.json",
            "changed_objects_checked":len({r[0]["uid"] for r in changed}),
            "cache_fingerprints":{k:bases[k][1] for k in bases},"vae_gate_reused":True,**clock()}
    atomic_json(receipt,report)
    output={"bases":{k:[str(v[0]),v[1]] for k,v in bases.items()},
        "contract":checkpoint_contract(cfg,bases),"longest":previous["longest"],"preflight":report,
        "cpu_sampler":cpu_sampler_checks(cfg),**clock()}
    atomic_json(run/"prepared.json",output);atomic_json(run/"cpu_sampler_checks.json",output["cpu_sampler"])
    return output


def admission(gpus):
    output=[]
    for gpu in gpus:
        line=subprocess.check_output(["nvidia-smi",f"--id={gpu}",
            "--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"],text=True).strip().split(",")
        uid,total,free,used,util=[x.strip() for x in line]
        budget=int(float(free)*1024**2*.9)
        if budget<8*1024**3: raise RuntimeError("insufficient shared-GPU admission")
        output.append({"physical_gpu":gpu,"uuid":uid,"nvidia_total_mib":float(total),
            "initial_free_mib":float(free),"initial_used_mib":float(used),"utilization":float(util),
            "limit_bytes":budget,"reserve_fraction":.1})
    return output


def state(run,phase,**extra):
    extra.setdefault("formal_training_started",(run/"formal_started.json").exists())
    atomic_json(run/"status.json",{"state":phase,**extra,**clock()})



def shutdown_reserve(cards):
    # GPU seconds: 2s polling + 15s graceful exit + kill/reap, with margin.
    return max(60,30*cards)


def archive_incomplete(run,folder):
    run=path(run);folder=path(folder)
    if folder.parent!=run/"checkpoints": raise ValueError("checkpoint must belong to this run")
    archived=[]
    for source in [folder.with_name(folder.name+".pending"),folder]:
        if source.is_symlink(): raise ValueError("checkpoint must not be a symlink")
        if not source.exists(): continue
        if source==folder and (source/"complete.json").exists(): continue
        destination=run/"incomplete"/(source.name+"."+str(time.time_ns()))
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.rename(source,destination);archived.append(str(destination))
    return archived


def rewind_logs(run,completed_update):
    # Preserve every interrupted attempt; the active stream contains committed
    # updates only, so replay does not create duplicate update/TB records.
    run=path(run);archived=[]
    for name in ["updates_rank_0.jsonl","updates_rank_1.jsonl","train.jsonl",
                 "validation.jsonl","checkpoint_io.jsonl"]:
        file=run/name
        if not file.exists(): continue
        raw=file.read_bytes();lines=raw.splitlines(keepends=True);keep=[];changed=False
        for i,line in enumerate(lines):
            try: row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1 or line.endswith(b"\n"): raise
                changed=True;continue
            if row["update"]<=completed_update: keep.append(row)
            else: changed=True
        if changed:
            destination=run/"interrupted_logs"/(name+"."+str(time.time_ns()))
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            atomic_lines(file,keep);archived.append(str(destination))
    return archived


def child(cfg,args,run,ledger,name,job,cards=2,cpu=False):
    job={**job,"config":str(path(args.config)),"prepared":str(run/"prepared.json"),
         "budget_ledger":str(ledger.file)}
    script="scripts/stage2_phases.py" if job["phase"] in ["decode","cpu"] else "scripts/stage2_fsdp_worker.py"
    environment=os.environ.copy()
    environment["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    environment["PYTORCH_CUDA_ALLOC_CONF"]=cfg["fsdp"]["allocator"]
    environment["CUBLAS_WORKSPACE_CONFIG"]=cfg["training"]["cublas_workspace_config"]
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"]="1";environment["NCCL_DEBUG"]="WARN"
    environment["CUDA_VISIBLE_DEVICES"]=""
    if not cpu:
        devices=admission([2,5] if cards==2 else [2]);job["admission"]=devices
        environment["CUDA_VISIBLE_DEVICES"]=",".join(v["uuid"] for v in devices)
        environment["STAGE2_PHASE_DEADLINE_UNIX"]=str(time.time()+(ledger.remaining()-shutdown_reserve(cards))/cards)
    suffix=str(time.time_ns());jobfile=run/"jobs"/(name+"_"+suffix+".json");atomic_json(jobfile,job)
    command=[str(path("environment/venvs/shape/bin/python")),"-B","-u"]
    if not cpu and cards==2:
        command += ["-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=2"]
    command += [str(path(script)),"--job",str(jobfile)]
    log=run/"phases"/(name+"_"+suffix+".log");log.parent.mkdir(parents=True,exist_ok=True)
    item=None;process=None;code=-1
    phase_kind=job["phase"] if job["phase"] in ["sample","decode"] else "train"
    minimum=cfg["orchestration"]["phase_minimum_seconds"].get(phase_kind,120)
    if not cpu and ledger.remaining()<minimum*cards+shutdown_reserve(cards):
        raise TimeoutError("insufficient GPU budget including shutdown reserve")
    if not cpu: item=ledger.start(run,name,cards,minimum)
    try:
        with log.open("a") as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            if item: item["child_pid"]=process.pid;ledger.save()
            if job["phase"]=="train" and not (run/"formal_started.json").exists():
                atomic_json(run/"formal_started.json",{"first_worker_pid":process.pid,**clock()})
            formal_started=(run/"formal_started.json").exists()
            state(run,name,child_pid=process.pid,log=str(log),budget_used_seconds=ledger.used(),
                  formal_training_started=formal_started)
            next_heartbeat=0
            while process.poll() is None:
                if not cpu and ledger.remaining()<=shutdown_reserve(cards):
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
                    raise TimeoutError("hard GPU budget reached")
                if time.monotonic()>=next_heartbeat:
                    state(run,name,child_pid=process.pid,log=str(log),
                          budget_used_seconds=ledger.used(),remaining_gpu_seconds=ledger.remaining(),
                          formal_training_started=formal_started)
                    next_heartbeat=time.monotonic()+10
                time.sleep(2)
            code=process.returncode
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
        if item: ledger.end(item,code)
    if code: raise RuntimeError(f"phase {name} failed ({code}); {log}")
    return log


def suite(cfg,args,run,ledger,prepared,label,checkpoint=None,verify=False):
    from stage2_evaluation import specs
    folder=run/"evaluation"/label
    jobs=specs(cfg,"monitor")
    if verify: jobs=[j for j in jobs if j["kind"]=="correct"][:2]
    contract=fingerprint({"run_contract":prepared["contract"],"label":label,"checkpoint":str(checkpoint),"jobs":jobs})
    description={"contract":contract,"jobs":jobs,"label":label,"checkpoint":str(checkpoint)}
    if (folder/"suite.json").exists():
        if read_json(folder/"suite.json")!=description: raise RuntimeError("suite contract changed")
    else: atomic_json(folder/"suite.json",description)
    if (folder/"complete.json").exists(): return
    if not (folder/"sampling_complete.json").exists():
        child(cfg,args,run,ledger,"sample_"+label,{"phase":"sample","output":str(folder),
            "suite":str(folder),"suite_contract":contract,"checkpoint":str(checkpoint) if checkpoint else None})
    if not (folder/"decode_complete.json").exists():
        child(cfg,args,run,ledger,"decode_"+label,{"phase":"decode","suite":str(folder)},cards=1)
    child(cfg,args,run,ledger,"cpu_"+label,{"phase":"cpu","suite":str(folder)},cpu=True)
    atomic_json(run/"phase_state.json",{"last_completed_evaluation":label,"checkpoint":str(checkpoint),**clock()})


def verify(cfg,args,run,ledger,prepared):
    for tiny,number in [(True,2),(False,cfg["orchestration"]["verify_updates"])]:
        name="tiny" if tiny else "actual3b";folder=run/"verification"/name
        contract=prepared["contract"]
        base={"output":str(folder),"tiny":tiny,"target_update":number+1,"save_update":number}
        if not (folder/"verify_reference_complete.json").exists():
            child(cfg,args,run,ledger,name+"_reference",{**base,"phase":"verify_reference","parity":tiny})
        if not (folder/"verify_resume_complete.json").exists():
            if not tiny:
                # Exercise the full exit -> model-only sample -> decoder -> CPU -> fresh restore chain.
                suite(cfg,args,run,ledger,prepared,"verify_stage_loop",folder/"checkpoint",verify=True)
            child(cfg,args,run,ledger,name+"_resume",{**base,"phase":"verify_resume",
                                                   "checkpoint":str(folder/"checkpoint"),"parity":tiny})
        reports=[read_json(folder/f"verify_resume_rank_{r}.json") for r in [0,1]]
        if not all(r.get("passed") for r in reports): raise RuntimeError("resume comparison not passed")
    files=list((run/"verification").rglob("memory_rank_*.jsonl"))+list((run/"evaluation").rglob("memory_rank_*.jsonl"))
    memory=[r for f in files for r in read_lines(f)]
    decoder_memory=read_json(run/"evaluation/verify_stage_loop/decoder_memory.json")
    if not all(x["passed"] for x in memory) or not decoder_memory["passed"]: raise RuntimeError("phase memory failed")
    reference=read_json(run/"verification/actual3b/verify_reference_complete.json")
    timings=reference["update_seconds"]
    report={"passed":True,"contract":prepared["contract"],"cpu":prepared["cpu_sampler"],
        "new_process_full_comparison":[read_json(run/f"verification/actual3b/verify_resume_rank_{r}.json") for r in [0,1]],
        "actual_update_count":20,"distinct_uid_count":640,"longest_uid":prepared["longest"]["uid"],
        "mean_update_seconds_with_io":sum(timings[1:20])/len(timings[1:20]),
        "all_memory_phases":memory,"decoder_memory":decoder_memory,
        "minimum_headroom_bytes":min([x["headroom_bytes"] for x in memory]+[decoder_memory["headroom_bytes"]]),
        "gpu_seconds_shared_ledger":ledger.used(),"formal_training_started":False,**clock()}
    atomic_json(run/"verification_report.json",report)
    state(run,"verification_complete_waiting_formal_start",formal_training_started=False,budget_used_seconds=ledger.used())


def formal(cfg,args,run,ledger,prepared):
    approval=read_json(args.budget)
    if not approval.get("formal_start_authorized"):
        raise RuntimeError("formal start requires explicit user confirmation after independent verification")
    verification=read_json(approval["verification_report"])
    if not verification.get("passed") or verification["contract"]!=prepared["contract"]:
        raise RuntimeError("verification contract changed")
    latest=None
    checkpoints=run/"checkpoints"
    existing=[]
    for target in cfg["orchestration"]["save_updates"]:
        name=f"step_{target:06d}"+("_final" if target==1565 else "")
        folder=checkpoints/name
        if (folder/"complete.json").exists(): existing.append((target,folder))
    if existing and not args.resume: raise RuntimeError("existing run requires --resume checkpoint or auto")
    if args.resume:
        if args.resume=="auto": latest=existing[-1][1] if existing else None
        else:
            latest=path(args.resume)
            if not existing or latest!=existing[-1][1]:
                raise RuntimeError("resume must use this run's latest committed checkpoint")
    rewind_logs(run,existing[-1][0] if existing else 0)
    if not existing: suite(cfg,args,run,ledger,prepared,"initial")
    # Complete any checkpoint's unfinished monitor before advancing model updates.
    for target,folder in existing: suite(cfg,args,run,ledger,prepared,folder.name,folder)
    for target in cfg["orchestration"]["save_updates"]:
        if existing and target<=existing[-1][0]: continue
        name=f"step_{target:06d}"+("_final" if target==1565 else "")
        archive_incomplete(run,checkpoints/name)
        child(cfg,args,run,ledger,f"train_to_{target}",{"phase":"train","output":str(run),
             "checkpoint":str(latest) if latest else None,"target_update":target})
        name=f"step_{target:06d}"+("_final" if target==1565 else "")
        latest=checkpoints/name
        if not (latest/"complete.json").exists(): raise RuntimeError("worker exited without committed checkpoint")
        suite(cfg,args,run,ledger,prepared,name,latest)
    state(run,"training_complete",formal_training_started=True,completed_updates=1565,
          budget_used_seconds=ledger.used(),semantic_status="pending_explicit_review")


def supervisor_signal(signum,frame):
    raise InterruptedError(f"supervisor received signal {signum}")


def main():
    signal.signal(signal.SIGTERM,supervisor_signal)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/stage2_nano10k_fsdp.yaml")
    p.add_argument("--gpus",default="2,5");p.add_argument("--run-id",required=True)
    p.add_argument("--budget",required=True);p.add_argument("--resume")
    p.add_argument("--verify-only",action="store_true");p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args();cfg=config(args.config)
    if args.gpus!="2,5": p.error("fixed physical GPUs are 2,5")
    if cfg["training"]["accumulation"]!=16 or cfg["training"]["global_batch"]!=32: raise RuntimeError("batch config")
    approval=read_json(args.budget)
    if not approval.get("approved") or approval["gpu_budget_seconds"]!=57600: raise RuntimeError("unapproved shared 16GPUh budget")
    run=run_path(cfg,args.run_id);run.mkdir(parents=True,exist_ok=True)
    try:
        with lock("environment/stage2_train.lock"),lock("environment/stage2_gpu.lock"),\
             lock("environment/gpu_2.lock"),lock("environment/gpu_5.lock"),\
             lock("environment/stage2_fsdp_budget.lock"):
            ledger=Ledger(approval["ledger"],approval["gpu_budget_seconds"])
            if (run/"provenance.json").exists():
                old=read_json(run/"prepared.json")
                from stage2_checkpoint import checkpoint_contract
                from stage2_cache import cache_base
                current=checkpoint_contract(cfg,{k:cache_base(cfg,k) for k in ["text","shape"]})
                if old["contract"]!=current and any(x["run"]==str(run) for x in ledger.data["phases"]):
                    raise RuntimeError("run contract changed after GPU execution: new run-id required")
            prepared=prepare(cfg,run)
            if not (run/"provenance.json").exists(): atomic_json(run/"provenance.json",provenance(cfg))
            if args.preflight_only:
                state(run,"cpu_preflight_complete",formal_training_started=False);return
            if args.verify_only: verify(cfg,args,run,ledger,prepared)
            else: formal(cfg,args,run,ledger,prepared)
    except BaseException as exc:
        state(run,"budget_exhausted" if isinstance(exc,TimeoutError) else "failed",error=repr(exc),
              traceback=traceback.format_exc(),resume="latest complete checkpoint; finish pending evaluation first",
              no_emergency_checkpoint=True)
        raise


if __name__=="__main__": main()

