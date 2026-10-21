"""Isolated two-rank worker: training/verification/model-only sampling; no VAE."""
from stage2_runtime import *
import datetime,gc,math,random,resource
import numpy as np
import torch
import torch.distributed as dist
from stage2_model import objective,lr_factor
from stage2_schedule import Prefetch,batches,next_position,deadline_check
import stage2_fsdp_io as io


def setup(cfg,job):
    rank=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(rank)
    a=job["admission"][rank]
    actual_total=torch.cuda.get_device_properties(rank).total_memory
    fraction=min(1.,a["limit_bytes"]/actual_total)
    torch.cuda.set_per_process_memory_fraction(fraction,rank)
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=10))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(cfg["training"]["deterministic_algorithms"])
    value=seed(cfg["seed"],"rank_global",rank)
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    # manual_seed only seeds the already current device; never get_rng_state_all.
    torch.cuda.manual_seed(value)
    return rank,int(actual_total*fraction)


class Memory:
    def __init__(self,folder,rank,cap,minimum):
        self.folder=folder;self.rank=rank;self.cap=cap;self.minimum=minimum
    def begin(self):
        io.trim();torch.cuda.reset_peak_memory_stats();self.started=time.monotonic()
    def end(self,phase):
        torch.cuda.synchronize()
        record={"phase":phase,"rank":self.rank,"seconds":time.monotonic()-self.started,
                "allocated_bytes":torch.cuda.max_memory_allocated(),
                "reserved_bytes":torch.cuda.max_memory_reserved(),"cap_bytes":self.cap,
                "cuda_total_bytes":torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                "cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        record["headroom_bytes"]=self.cap-record["reserved_bytes"]
        record["passed"]=record["headroom_bytes"]>=self.minimum
        append(self.folder/f"memory_rank_{self.rank}.jsonl",record)
        self.reports=[None]*2;dist.all_gather_object(self.reports,record)
        passed=torch.tensor(int(record["passed"]),device="cuda")
        dist.all_reduce(passed,op=dist.ReduceOp.MIN)
        if not passed.item(): raise RuntimeError("phase lacks required 2GiB allocator headroom: "+phase)
        return record


def build(cfg,job,rank):
    from stage2_fsdp import wrap
    from stage2_model import load_model,Stage2Generator
    if job.get("tiny"):
        from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed(cfg["seed"],"tiny_init"))
            dit=HunYuanDiTPlain(input_size=16,in_channels=4,hidden_size=32,context_dim=16,
                depth=4,num_heads=4,use_attention_pooling=False,num_moe_layers=2,num_experts=2,moe_top_k=2)
            raw=Stage2Generator(dit,12)
    else: raw=load_model(cfg,device="cpu")
    return wrap(raw,rank)


def cache_loader(cfg,prepared,tiny=False):
    if tiny:
        def synthetic(row):
            g=torch.Generator().manual_seed(seed(cfg["seed"],"tiny_cache",row["uid"]))
            return (torch.randn(1,7+int(row["uid"])%8,12,generator=g),
                    torch.randn(1,16,4,generator=g),torch.full((1,16,4),-2.))
        return synthetic
    from stage2_cache import load_cached
    bases={k:(path(v[0]),v[1]) for k,v in prepared["bases"].items()}
    return lambda row:load_cached(cfg,row,row["split"],bases)


def update(model,opt,cfg,loaded,generator,number,total):
    settings=cfg["training"];opt.zero_grad(set_to_none=True)
    factor=lr_factor(number,total,settings["warmup_updates"],settings["minimum_lr_factor"])
    for group,peak in zip(opt.param_groups,[settings["dit_lr"],settings["connector_lr"]]):
        group["lr"]=peak*factor
    sums=torch.zeros(3,device="cuda",dtype=torch.float64)
    start=time.monotonic();uids=[];local_count=len(loaded)
    model.train()
    for row,(hidden,mean,logvar) in loaded:
        deadline_check(30)
        mse,aux,loss=objective(model,mean.cuda(),logvar.cuda(),hidden.cuda().float(),
                              cfg["cache"]["scale"],generator,True)
        finite=torch.tensor(int(bool(torch.isfinite(loss))),device="cuda")
        dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise RuntimeError("nonfinite objective")
        (loss/local_count).backward()
        sums+=torch.stack([mse.detach(),aux.detach(),loss.detach()]).double()
        uids.append(row["uid"])
        del mse,aux,loss
    norm=model.clip_grad_norm_(settings["clip_norm"])
    if not torch.isfinite(norm): raise RuntimeError("nonfinite global gradient")
    opt.step();opt.zero_grad(set_to_none=True)
    dist.all_reduce(sums)
    count=torch.tensor(local_count,device="cuda",dtype=torch.int64);dist.all_reduce(count)
    torch.cuda.synchronize()
    elapsed=torch.tensor(time.monotonic()-start,device="cuda",dtype=torch.float64)
    dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    vals=(sums/count).cpu().tolist()
    return {"update":number,"flow_mse":vals[0],"moe_aux":vals[1],"total_loss":vals[2],
            "gradient_norm":float(norm),"seconds":float(elapsed),"global_samples":int(count),
            "uids":uids,"dit_lr":opt.param_groups[0]["lr"],"connector_lr":opt.param_groups[1]["lr"]}


def validation(model,cfg,rows,load,memory):
    memory.begin();model.eval();sums=torch.zeros(2,device="cuda",dtype=torch.float64)
    with torch.no_grad():
        for row in rows[dist.get_rank()::2]:
            deadline_check(15)
            h,m,l=load(row)
            g=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"validation",row["uid"]))
            mse,aux,loss=objective(model,m.cuda(),l.cuda(),h.cuda().float(),cfg["cache"]["scale"],g,False)
            if not torch.isfinite(mse): raise RuntimeError("nonfinite validation")
            sums[0]+=mse;sums[1]+=1
            del h,m,l,mse,aux,loss
    dist.all_reduce(sums);value={"flow_mse":float(sums[0]/sums[1]),"count":int(sums[1]),
                               "fixed_randomness":True,"moe_aux":"disabled_in_eval"}
    del sums;memory.end("validation");model.train();return value


def summary_rng(record,position,generator):
    return {**record,"position":position,"rng_sha256":io.rng_digest(io.local_rng()),
            "training_rng_sha256":io.rng_digest({k:v.get_state() for k,v in generator.items()})}


def save_point(folder,model,opt,cfg,contract,position,generator,interval,memory):
    scheduler={"last_completed_update":position["update"],"total_updates":1565,
               "warmup_updates":80,"minimum_lr_factor":.1,
               "next_update":position["update"]+1}
    state={"position":position,"scheduler":scheduler,"training_generator":{k:v.get_state() for k,v in generator.items()},
           "interval":interval}
    memory.begin();value=io.save(folder,model,opt,contract,state);memory.end("save")
    return value


def execute_training(cfg,job,rank,model,memory):
    from stage2_optimizer import build_optimizer
    folder=path(job["output"]);prepared=read_json(job["prepared"]);contract=prepared["contract"]
    tiny=job.get("tiny",False);load=cache_loader(cfg,prepared,tiny)
    rows=([{"uid":str(i),"split":"train"} for i in range(96)] if tiny else manifests(cfg,"train"))
    val=([{"uid":str(i),"split":"val"} for i in range(8)] if tiny else manifests(cfg,"val"))
    if job["phase"].startswith("verify") and not tiny:
        # Preserve unique UIDs while putting the known longest condition in the first update.
        rows=epoch_verify_rows(rows,cfg["seed"],prepared["longest"]["uid"])
    opt=build_optimizer(model.module,cfg)
    g={name:torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"training",name,rank))
       for name in ["posterior","noise","time"]}
    position={"epoch":0,"offset":0,"update":0}
    interval={"samples":0,"flow_sum":0.,"aux_sum":0.,"total_sum":0.,"updates":0,
              "seconds":0.,"gradient_sum":0.}
    if job.get("checkpoint"):
        memory.begin();state=io.load(job["checkpoint"],model,opt,contract);memory.end("restore")
        position=state["position"];interval=state["interval"]
        if state["scheduler"]["next_update"]!=position["update"]+1: raise RuntimeError("scheduler/update mismatch")
        for name,stream in g.items(): stream.set_state(state["training_generator"][name])
        del state
    target=job["target_update"];verify=job["phase"].startswith("verify")
    writer=None
    if not verify and rank==0:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(folder/"tensorboard"),purge_step=position["update"]+1)
    timings=[]
    try:
        for epoch in range(position["epoch"],5):
            if position["update"]>=target: break
            # Verification uses one fixed unique order; formal training uses the prescribed global shuffle.
            stream=verify_batches(rows,position["offset"]) if verify else batches(rows,cfg["seed"],epoch,position["offset"])
            local=((start,group[rank::2]) for start,group in stream)
            memory.begin()
            with Prefetch(local,load) as queue:
                wall_start=time.monotonic()
                for start,loaded in queue:
                    if position["update"]>=target: break
                    record=update(model,opt,cfg,loaded,g,position["update"]+1,1565)
                    wall=torch.tensor(time.monotonic()-wall_start,device="cuda",dtype=torch.float64)
                    dist.all_reduce(wall,op=dist.ReduceOp.MAX)
                    record["compute_seconds"]=record["seconds"];record["seconds"]=float(wall)
                    del wall
                    position=next_position(epoch,start,len(loaded)*2,len(rows),record["update"])
                    record["position"]=position.copy();timings.append(record["seconds"])
                    n=record["global_samples"]
                    interval["samples"]+=n;interval["updates"]+=1;interval["seconds"]+=record["seconds"]
                    interval["flow_sum"]+=record["flow_mse"]*n
                    interval["aux_sum"]+=record["moe_aux"]*n
                    interval["total_sum"]+=record["total_loss"]*n
                    interval["gradient_sum"]+=record["gradient_norm"]*n
                    append(folder/f"updates_rank_{rank}.jsonl",record)
                    if rank==0:
                        atomic_json(folder/"worker_progress.json",{"phase":job["phase"],**record,**clock()})
                        print("UPDATE",canonical(record),flush=True)
                    if writer and (record["update"]%100==0 or record["update"]==1565):
                        memory.end("train_interval")
                        usage=read_json(job["budget_ledger"])
                        used=sum(x.get("gpu_seconds",max(0,time.time()-x["started_unix"])*x["gpu_count"])
                                 for x in usage["phases"])
                        logged={"update":record["update"],"samples":interval["samples"],
                            "flow_mse":interval["flow_sum"]/interval["samples"],
                            "moe_aux":interval["aux_sum"]/interval["samples"],
                            "total_loss":interval["total_sum"]/interval["samples"],
                            "gradient_norm":interval["gradient_sum"]/interval["samples"],
                            "samples_per_second":interval["samples"]/interval["seconds"],
                            "interval_seconds":interval["seconds"],"gpu_seconds":used,
                            "dit_lr":record["dit_lr"],"connector_lr":record["connector_lr"],
                            "rank_memory":memory.reports}
                        append(folder/"train.jsonl",logged)
                        for key,v in logged.items():
                            if isinstance(v,(int,float)): writer.add_scalar("train/"+key,v,record["update"])
                        writer.flush();interval={k:0 for k in interval};memory.begin()
                    # All ranks enter memory collectives, even though only rank0 owns writer.
                    elif not writer and not verify and (record["update"]%100==0 or record["update"]==1565):
                        memory.end("train_interval");interval={k:0 for k in interval};memory.begin()
                    if verify and record["update"]==job.get("save_update"):
                        memory.end("train")
                        io_value=save_point(folder/"checkpoint",model,opt,cfg,contract,position,g,interval,memory)
                        if rank==0: atomic_json(folder/"checkpoint_io.json",io_value)
                        memory.begin()
                    if not verify and (position["offset"]==0 or record["update"]==target):
                        memory.end("train")
                        value=validation(model,cfg,val,load,memory)
                        if rank==0:
                            append(folder/"validation.jsonl",{"update":record["update"],**value})
                            writer.add_scalar("validation/flow_mse",value["flow_mse"],record["update"]);writer.flush()
                        memory.begin()
                    wall_start=time.monotonic()
            memory.end("train")
            if position["update"]>=target: break
            position["offset"]=0
        if position["update"]!=target: raise RuntimeError("training stopped before phase target")
        if verify:
            summary=summary_rng(record,position,g)
            if job["phase"]=="verify_reference":
                report=io.snapshot(folder/f"reference_rank_{rank}.bin",model,opt,summary)
            else:
                report=io.compare_snapshot(folder/f"reference_rank_{rank}.bin",model,opt,summary)
            atomic_json(folder/f"{job['phase']}_rank_{rank}.json",report)
        else:
            name=f"step_{target:06d}"+("_final" if target==1565 else "")
            value=save_point(folder/"checkpoints"/name,model,opt,cfg,contract,position,g,interval,memory)
            if rank==0: append(folder/"checkpoint_io.jsonl",{"update":target,**value})
        if verify:
            value=validation(model,cfg,val,load,memory)
            if rank==0: atomic_json(folder/(job["phase"]+"_validation.json"),value)
        if rank==0: atomic_json(folder/(job["phase"]+"_complete.json"),{
            "position":position,"update_seconds":timings,"tiny":tiny,**clock()})
    finally:
        if writer: writer.close()
        del opt;io.trim()


def verify_batches(rows,offset):
    for start in range(offset,len(rows),32): yield start,rows[start:start+32]


def epoch_verify_rows(rows,master,longest):
    from stage2_schedule import epoch_order
    order=epoch_order(rows,master,0);index=next(i for i,r in enumerate(order) if r["uid"]==longest)
    order[1],order[index]=order[index],order[1]
    return order


def sample_jobs(cfg,job,rank,model,memory):
    from stage2_fsdp import sampling
    prepared=read_json(job["prepared"]);contract=prepared["contract"]
    if job.get("checkpoint"):
        memory.begin();io.load(job["checkpoint"],model,None,contract,restore_random=False);memory.end("model_only_restore")
    model.eval();folder=path(job["suite"]);jobs=read_json(folder/"suite.json")["jobs"]
    load=cache_loader(cfg,prepared);lookup={r["uid"]:r for s in ["train","val"] for r in manifests(cfg,s)}
    memory.begin()
    for offset in range(0,len(jobs),2):
        deadline_check(20)
        pair=jobs[offset:offset+2]
        if len(pair)!=2: raise RuntimeError("sampling jobs must be paired across ranks")
        item=pair[rank];label=item["uid"]+"__"+item["kind"]
        out=folder/"latents"/(label+".pt")
        present=torch.tensor(int(out.exists()),device="cuda");dist.all_reduce(present)
        if int(present)==2: continue
        h=load(lookup[item["condition_uid"]])[0]
        g=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"generation",item["uid"]))
        latent=sampling(model,h.cuda().float(),cfg,g)
        if not out.exists(): save_tensor(out,{"latent":latent,"spec":item,"suite_contract":job["suite_contract"]})
        else:
            old=torch.load(out,map_location="cpu",weights_only=True)
            if old["suite_contract"]!=job["suite_contract"]: raise RuntimeError("latent contract mismatch")
        del h,latent
    memory.end("sample");dist.barrier()
    if rank==0: atomic_json(folder/"sampling_complete.json",{"jobs":len(jobs),**clock()})


def tiny_parity(cfg,rank):
    # Existing actual-MoE+checkpoint parity logic, rerun because new worker/IO is introduced.
    import copy
    from stage2_fsdp import wrap
    from stage2_model import Stage2Generator
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
    torch.manual_seed(123)
    raw=Stage2Generator(HunYuanDiTPlain(input_size=16,in_channels=4,hidden_size=32,context_dim=16,
        depth=4,num_heads=4,use_attention_pooling=False,num_moe_layers=2,num_experts=2,moe_top_k=2),12).cuda()
    ref=copy.deepcopy(raw);model=wrap(raw,rank)
    opts=[torch.optim.AdamW(m.parameters(),lr=1e-3,betas=(.9,.99),eps=1e-6,foreach=False) for m in [ref,model]]
    g=torch.Generator(device="cuda").manual_seed(444)
    data=[(torch.randn(1,16,4,device="cuda",generator=g),torch.rand(1,device="cuda",generator=g),
           torch.randn(1,7+i,12,device="cuda",generator=g),torch.randn(1,16,4,device="cuda",generator=g)) for i in range(4)]
    for m,opt,selected in [(ref,opts[0],data),(model,opts[1],data[rank::2])]:
        opt.zero_grad(set_to_none=True)
        for x,t,h,y in selected:
            pred,aux=m(x,t,h,checkpoint_blocks=True);loss=((pred-y).square().mean()+aux)/len(selected);loss.backward()
        if m is model: m.clip_grad_norm_(1.)
        else: torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step();opt.zero_grad(set_to_none=True)
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=False,rank0_only=False)):
        actual=model.state_dict()
        error=max(float((v-ref.state_dict()[k.replace(".block.",".")]).abs().max()) for k,v in actual.items())
    if error>2e-5: raise RuntimeError("tiny actual-MoE gradient parity failure")
    del actual,ref,raw,model,opts;io.trim()
    return {"passed":True,"max_parameter_error":error,"actual_upstream_moe":True,"activation_checkpoint":True}


def main():
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument("--job",required=True);args=p.parse_args()
    job=read_json(args.job);cfg=config(job["config"]);folder=path(job["output"]);folder.mkdir(parents=True,exist_ok=True)
    rank,cap=setup(cfg,job);memory=Memory(folder,rank,cap,cfg["fsdp"]["min_headroom_bytes"])
    try:
        if job.get("parity"):
            report=tiny_parity(cfg,rank);atomic_json(folder/f"tiny_parity_rank_{rank}.json",report)
        memory.begin();model=build(cfg,job,rank);memory.end("load")
        if job["phase"]=="sample": sample_jobs(cfg,job,rank,model,memory)
        else: execute_training(cfg,job,rank,model,memory)
        del model;io.trim();dist.barrier()
    except BaseException as exc:
        atomic_json(folder/f"failure_rank_{rank}_{job['phase']}.json",{"error":repr(exc),"traceback":traceback.format_exc(),**clock()})
        raise
    finally: dist.destroy_process_group()


if __name__=="__main__": main()

