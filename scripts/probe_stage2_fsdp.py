"""Finite two-rank GPU-AdamW FULL_SHARD probe; cannot start formal training."""
from stage2_runtime import *
import datetime,resource,gc,copy
import torch
import torch.distributed as dist
from stage2_fsdp import wrap,save_sharded,load_sharded,sampling
from stage2_model import Stage2Generator,load_model,objective,lr_factor
from stage2_optimizer import build_optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig

def setup():
    rank=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(rank)
    admission=read_json(os.environ["STAGE2_FSDP_ADMISSION"])[rank]
    torch.cuda.set_per_process_memory_fraction(admission["limit_mib"]/admission["total_mib"],rank)
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=30))
    torch.set_num_threads(1);torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32=False
    return rank,admission

def adam(model):
    return torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(.9,.99),eps=1e-6,
                             weight_decay=.01,foreach=False)

def tiny(folder,rank):
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
    dit=HunYuanDiTPlain(input_size=16,in_channels=4,hidden_size=32,context_dim=16,
                       depth=4,num_heads=4,use_attention_pooling=False,
                       num_moe_layers=2,num_experts=2,moe_top_k=2)
    raw=Stage2Generator(dit,hidden_dim=12).cuda()
    ref=copy.deepcopy(raw);ref_optimizer=adam(ref)
    model=wrap(raw,torch.cuda.current_device());optimizer=adam(model)
    g=torch.Generator(device="cuda").manual_seed(444)
    batches=[[(torch.randn(1,16,4,device="cuda",generator=g),
               torch.rand(1,device="cuda",generator=g),
               torch.randn(1,7+i,12,device="cuda",generator=g),
               torch.randn(1,16,4,device="cuda",generator=g)) for i in range(4)]
             for _ in range(2)]
    def update(m,o,b,distributed):
        m.train();o.zero_grad(set_to_none=True)
        selected=b[rank::2] if distributed else b
        for x,t,h,target in selected:
            pred,aux=m(x,t,h,checkpoint_blocks=True)
            loss=(pred-target).square().mean()+aux
            (loss/len(selected)).backward()
        norm=m.clip_grad_norm_(1.) if distributed else torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        o.step();o.zero_grad(set_to_none=True)
        return float(norm)
    def full():
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                 FullStateDictConfig(offload_to_cpu=False,rank0_only=False)):
            return {k:v.detach().clone() for k,v in model.state_dict().items()}
    def canonical(key): return key.replace(".block.",".")
    def compare_ref():
        state=full();expected=ref.state_dict()
        error=max(float((v-expected[canonical(k)]).abs().max()) for k,v in state.items())
        if error>2e-5: raise RuntimeError(f"FSDP/reference parameter error {error}")
        return error
    update(ref,ref_optimizer,batches[0],False)
    update(model,optimizer,batches[0],True)
    error1=compare_ref()
    metadata={"kind":"tiny_actual_dit_moe_parity","update":1,"world_size":2}
    io=save_sharded(folder/"checkpoint",model,optimizer,metadata)
    update(ref,ref_optimizer,batches[1],False)
    update(model,optimizer,batches[1],True)
    error2=compare_ref();expected=full()
    load_sharded(folder/"checkpoint",model,optimizer,metadata)
    update(model,optimizer,batches[1],True)
    actual=full()
    resume_error=max(float((actual[k]-v).abs().max()) for k,v in expected.items())
    if resume_error>1e-7: raise RuntimeError(f"FSDP resume error {resume_error}")
    result={"passed":True,"first_update_max_error":error1,"second_update_max_error":error2,
            "resume_next_update_max_error":resume_error,"checkpoint_io":io,
            "actual_upstream_moe":True,"activation_checkpointing":True,**clock()}
    if rank==0: atomic_json(folder/"tiny_report.json",result);print("TINY_PASSED",canonical_json(result),flush=True)
    del model,optimizer,raw,ref,ref_optimizer
    gc.collect();torch.cuda.empty_cache();dist.barrier()

def canonical_json(v): return canonical(v)

def snapshots(model,optimizer):
    result=[]
    for p in model.parameters():
        if p.numel(): result.append(p.detach().flatten()[:4].cpu().clone())
    for state in optimizer.state.values():
        for k in ["step","exp_avg","exp_avg_sq"]:
            if k in state and torch.is_tensor(state[k]) and state[k].numel():
                result.append(state[k].detach().flatten()[:4].cpu().clone())
    return result

def actual_probe(cfg,run,prepared,rank,admission):
    from stage2_cache import load_cached
    from stage2_checkpoint import checkpoint_contract
    started=time.monotonic();bases={k:(path(v[0]),v[1]) for k,v in prepared["bases"].items()}
    contract=checkpoint_contract(cfg,bases)
    if contract!=prepared["contract"]: raise RuntimeError("Prepared probe contract changed")
    longest=prepared["longest"]
    h,mean,logvar=load_cached(cfg,longest,"train",bases)
    print("RANK",rank,"LOAD_PRETRAINED",flush=True)
    model=wrap(load_model(cfg,device="cpu"),torch.cuda.current_device())
    print("RANK",rank,"FSDP_MODEL_READY",flush=True)
    optimizer=build_optimizer(model.module,cfg)
    generator=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"fsdp_probe",rank))
    train=cfg["training"];accum=train["accumulation"]//dist.get_world_size()
    if accum*dist.get_world_size()!=32: raise RuntimeError("Global batch must stay 32")
    total=1565;times=[]
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for update in range(1,6):
        dist.barrier();torch.cuda.synchronize();start=time.monotonic()
        factor=lr_factor(update,total,train["warmup_updates"],train["minimum_lr_factor"])
        for group,peak in zip(optimizer.param_groups,[train["dit_lr"],train["connector_lr"]]):
            group["lr"]=peak*factor
        optimizer.zero_grad(set_to_none=True);stats=torch.zeros(3,device="cuda")
        for micro in range(accum):
            mse,aux,loss=objective(model,mean.cuda(),logvar.cuda(),h.cuda().float(),
                                  cfg["cache"]["scale"],generator,True)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite FSDP loss")
            (loss/accum).backward()
            stats+=torch.stack([mse.detach(),aux.detach(),loss.detach()])
        norm=model.clip_grad_norm_(train["clip_norm"])
        if not torch.isfinite(norm): raise RuntimeError("Nonfinite global FSDP gradient")
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(stats);stats/=32
        torch.cuda.synchronize()
        seconds=torch.tensor(time.monotonic()-start,device="cuda")
        dist.all_reduce(seconds,op=dist.ReduceOp.MAX);elapsed=float(seconds);times.append(elapsed)
        record={"update":update,"seconds":elapsed,"flow_mse":float(stats[0]),
                "moe_aux":float(stats[1]),"total_loss":float(stats[2]),"global_gradient_norm":float(norm),
                "global_batch":32,"local_accumulation":accum,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        append(run/f"rank_{rank}_updates.jsonl",record)
        if rank==0:
            atomic_json(run/"status.json",{"state":"fsdp_probe_updates","completed_updates":update,
                        "formal_training_started":False,**clock()})
            print("FSDP_UPDATE",canonical(record),flush=True)
    train_memory={"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
    metadata={"contract":contract,"update":5,"world_size":2,"global_batch":32}
    saved=snapshots(model,optimizer)
    io=save_sharded(run/"probe_checkpoint",model,optimizer,metadata,
                    {"training_generator":generator.get_state()})
    with torch.no_grad():
        for p in model.parameters():
            if p.numel(): p.flatten()[0].add_(.25);break
    restore_start=time.monotonic()
    restored=load_sharded(run/"probe_checkpoint",model,optimizer,metadata)
    generator.set_state(restored["training_generator"])
    actual=snapshots(model,optimizer)
    if len(saved)!=len(actual) or any(not torch.equal(a,b) for a,b in zip(saved,actual)):
        raise RuntimeError("Actual model/optimizer checkpoint restore mismatch")
    restore_seconds=time.monotonic()-restore_start
    model.eval();dist.barrier();val_start=time.monotonic()
    val_mse=torch.zeros((),device="cuda")
    with torch.no_grad():
        for row in prepared["validation"]:
            vh,vm,vl=load_cached(cfg,row,"val",bases)
            # Same fixed validation stream across ranks; equal collective counts.
            vg=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"validation",row["uid"]))
            mse,aux,loss=objective(model,vm.cuda(),vl.cuda(),vh.cuda().float(),
                                  cfg["cache"]["scale"],vg,False)
            val_mse+=mse
    torch.cuda.synchronize();validation_seconds=time.monotonic()-val_start
    sg=torch.Generator(device="cuda").manual_seed(seed(cfg["seed"],"probe_sampling",longest["uid"]))
    sample_start=time.monotonic();latent=sampling(model,h.cuda().float(),cfg,sg)
    torch.cuda.synchronize();sample_seconds=time.monotonic()-sample_start
    all_memory={"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
    del model,optimizer;gc.collect();torch.cuda.empty_cache();dist.barrier()
    decode_report=None
    if rank==0:
        from stage2_evaluation import load_vae,decode
        atomic_json(run/"status.json",{"state":"fsdp_probe_decode","formal_training_started":False,**clock()})
        vae=load_vae(cfg);torch.cuda.synchronize();decode_start=time.monotonic()
        try:
            mesh=decode(vae,latent,cfg);mesh.export(run/"probe_generation.glb")
            decode_report={"valid":True,"vertices":len(mesh.vertices),"faces":len(mesh.faces)}
        except ValueError as exc:
            decode_report={"valid":False,"error":repr(exc)}
        torch.cuda.synchronize();decode_report["seconds"]=time.monotonic()-decode_start
        del vae;torch.cuda.empty_cache()
    dist.barrier()
    all_memory={"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
    local={"rank":rank,"admission":admission,"train_peak":train_memory,"all_peak":all_memory,
           "cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "update_seconds":times,"checkpoint_io":io,"checkpoint_restore_seconds":restore_seconds,
           "checkpoint_restore_passed":True,"validation_count":8,
           "validation_flow_mse":float(val_mse)/8,"validation_seconds":validation_seconds,
           "sampling_seconds":sample_seconds,"elapsed_seconds":time.monotonic()-started}
    reports=[None]*2;dist.all_gather_object(reports,local)
    if rank==0:
        atomic_json(run/"probe_report.json",{"kind":"two_gpu_fsdp_full_shard","contract":contract,
                    "rank_reports":reports,"global_batch":32,"updates":5,
                    "reserve_passed":all(x["all_peak"]["reserved"]<=x["admission"]["limit_mib"]*1024**2 for x in reports),
                    "gradient_sync":"reduce-scatter every microbatch; no no_sync",
                    "parameter_dtype":"float32","autocast":"bfloat16","optimizer":"gpu_adamw",
                    "checkpoint_restore_passed":True,"decode":decode_report,
                    "semantic_success":"not_evaluated","formal_training_started":False,**clock()})
        atomic_json(run/"status.json",{"state":"probe_complete_waiting_budget",
                    "formal_training_started":False,**clock()})
        print("FSDP_PROBE_COMPLETE",flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--prepared",required=True)
    p.add_argument("--run-id",required=True);p.add_argument("--tiny-only",action="store_true")
    args=p.parse_args();rank,admission=setup()
    cfg=config("configs/stage2_nano10k.yaml");run=run_path(cfg,args.run_id)
    if rank==0:
        run.mkdir(parents=True,exist_ok=True)
        atomic_json(run/"provenance.json",{**provenance(cfg),"world_size":2,"physical_gpus":[2,5]})
    dist.barrier()
    try:
        tiny(run/"tiny",rank)
        if not args.tiny_only: actual_probe(cfg,run,read_json(args.prepared),rank,admission)
    except BaseException as exc:
        atomic_json(run/f"rank_{rank}_failure.json",{"error":repr(exc),**clock()})
        raise
    finally: dist.destroy_process_group()

if __name__=="__main__": main()
