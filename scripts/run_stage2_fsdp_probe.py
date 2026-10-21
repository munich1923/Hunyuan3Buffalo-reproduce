"""Wait for CPU probe, admit GPU2+5, run finite correctness + full resource probes."""
from stage2_runtime import *
import contextlib

def alive(pid):
    try: os.kill(pid,0);return True
    except ProcessLookupError: return False

def main():
    cfg=config("configs/stage2_nano10k.yaml")
    folder=path("results/stage2_nano10k/fsdp_gpu2_gpu5_20261021")
    folder.mkdir(parents=True,exist_ok=True)
    with lock("environment/stage2_fsdp_probe_supervisor.lock"):
        if (folder/"process.json").exists(): raise FileExistsError("Inspect prior probe before rerun")
        atomic_json(folder/"process.json",{"pid":os.getpid(),"state":"cpu_preflight",**clock()})
        from train_stage2 import preflight
        from stage2_checkpoint import checkpoint_contract
        lengths={};bases,splits=preflight(cfg,lengths)
        longest=next(row for row in splits["train"] if row["uid"]==max(lengths,key=lengths.get))
        prepared={"bases":{k:[str(v[0]),v[1]] for k,v in bases.items()},
                  "contract":checkpoint_contract(cfg,bases),"longest":longest,
                  "validation":splits["val"][:8],"validated_objects":11000,**clock()}
        atomic_json(folder/"prepared.json",prepared)
        atomic_json(folder/"status.json",{"state":"waiting_cpu_probe_exit",**clock()})
        # Never overlap GPU probes; ownership is additionally protected by stage2 locks.
        while alive(70232) or alive(70224): time.sleep(5)
        cpu=read_json("results/stage2_nano10k/gpu2_cpuoffload_20261021/status.json")
        atomic_json(folder/"cpu_predecessor.json",cpu)
        with contextlib.ExitStack() as stack:
            for name in ["environment/stage2_train.lock","environment/stage2_gpu.lock",
                         "environment/gpu_2.lock","environment/gpu_5.lock"]:
                stack.enter_context(lock(name))
            admission=[]
            for index in [2,5]:
                values=subprocess.check_output(["nvidia-smi",f"--id={index}",
                  "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
                  "--format=csv,noheader,nounits"],text=True).strip().split(",")
                uuid,total,free,util=[x.strip() for x in values]
                limit=shared_memory_limit(float(total),float(free))
                if limit<24000: raise RuntimeError(f"GPU{index} budget under 24000 MiB; not starting")
                admission.append({"physical_gpu":index,"uuid":uuid,"total_mib":float(total),
                                  "initial_free_mib":float(free),"initial_utilization":float(util),
                                  "limit_mib":limit,"reserve_fraction":.1})
            atomic_json(folder/"admission.json",admission)
            command=[str(path("environment/venvs/shape/bin/python")),"-B","-u","-m",
                     "torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=2",
                     str(path("scripts/probe_stage2_fsdp.py")),"--prepared",str(folder/"prepared.json"),
                     "--run-id","probe_fsdp_gpu2_gpu5_20261021"]
            env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=",".join(a["uuid"] for a in admission)
            env["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";env["STAGE2_FSDP_ADMISSION"]=str(folder/"admission.json")
            env["TORCH_NCCL_ASYNC_ERROR_HANDLING"]="1";env["NCCL_DEBUG"]="WARN"
            atomic_json(folder/"provenance.json",{**provenance(cfg),"command":command,"admission":admission})
            started=time.monotonic()
            with (folder/"probe.log").open("a") as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
                atomic_json(folder/"status.json",{"state":"two_rank_probe_running","child_pid":child.pid,
                            "formal_training_started":False,**clock()})
                code=child.wait()
            run=run_path(cfg,"probe_fsdp_gpu2_gpu5_20261021")
            atomic_json(folder/"status.json",{"state":"complete_waiting_budget" if code==0 and
                        (run/"probe_report.json").exists() else "failed","exit_code":code,
                        "allocated_gpu_seconds":2*(time.monotonic()-started),
                        "formal_training_started":False,**clock()})
            raise SystemExit(code)

if __name__=="__main__": main()
