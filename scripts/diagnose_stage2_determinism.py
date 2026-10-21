"""Finite BF16 SDPA backward determinism diagnostic; supervisor bills GPU2."""
from stage2_runtime import *
import argparse


def worker(output):
    import torch
    torch.cuda.set_device(0);torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    g=torch.Generator(device="cuda").manual_seed(123)
    shape=(1,16,4097,64)
    inputs=[torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=g) for _ in range(3)]
    upstream=torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=g)
    result=[]
    for deterministic in [False,True]:
        torch.use_deterministic_algorithms(deterministic)
        repeated=[]
        for repeat in range(3):
            q,k,v=[x.detach().clone().requires_grad_(True) for x in inputs]
            with torch.backends.cuda.sdp_kernel(enable_flash=True,enable_math=False,enable_mem_efficient=True):
                y=torch.nn.functional.scaled_dot_product_attention(q,k,v)
            y.backward(upstream)
            repeated.append([x.grad.detach().cpu().clone() for x in [q,k,v]])
            del q,k,v,y
        errors=[max(float((a.float()-b.float()).abs().max()) for a,b in zip(repeated[0],other))
                for other in repeated[1:]]
        result.append({"deterministic_algorithms":deterministic,"gradient_max_errors":errors,
                       "exact":all(x==0 for x in errors)})
    atomic_json(output,{"attention_shape":list(shape),"dtype":"bfloat16","results":result,**clock()})
    print(canonical(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true");p.add_argument("--output",required=True);args=p.parse_args()
    if args.worker: worker(args.output);return
    from stage2_schedule import Ledger
    import contextlib,signal
    output=path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with lock("environment/stage2_train.lock"),lock("environment/stage2_gpu.lock"),\
         lock("environment/gpu_2.lock"),lock("environment/stage2_fsdp_budget.lock"):
        ledger=Ledger("results/stage2_nano10k/stage2_fsdp_budget_20261021.json",57600)
        uid=subprocess.check_output(["nvidia-smi","--id=2","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=uid;env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        item=ledger.start(output.parent,"bf16_sdpa_diagnostic",1,120);code=-1
        try:
            cmd=[sys.executable,"-B","-u",str(Path(__file__)),"--worker","--output",str(output)]
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True)
            item["child_pid"]=process.pid;ledger.save();code=process.wait(timeout=180)
        finally:
            if code==-1:
                os.killpg(process.pid,signal.SIGTERM);process.wait()
            ledger.end(item,code)
        if code: raise SystemExit(code)


if __name__=="__main__": main()

