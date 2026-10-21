"""Finite GPU2 CPU-AdamW resource probe; no formal training or automatic restart."""
from stage2_runtime import *

def main():
    folder=path("results/stage2_nano10k/gpu2_cpuoffload_20261021")
    run=path("results/stage2_nano10k/probe_gpu2_cpuoffload_20261021")
    folder.mkdir(parents=True,exist_ok=True)
    command=["bash",str(path("scripts/stage2_launch.sh")),"train_stage2.py",
             "--config","configs/stage2_nano10k_cpuoffload.yaml","--probe-only",
             "--run-id","probe_gpu2_cpuoffload_20261021","--gpu","2"]
    env=os.environ.copy();env["STAGE2_ALLOW_SHARED_GPU"]="2"
    with lock("environment/stage2_cpuoffload_probe.lock"):
        if (folder/"process.json").exists():
            raise RuntimeError("New probe already launched; inspect existing logs before rerunning")
        atomic_json(folder/"provenance.json",provenance(config("configs/stage2_nano10k_cpuoffload.yaml")))
        with (folder/"probe.log").open("a") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            state={"state":"preflight_or_probe_running","pid":os.getpid(),"child_pid":child.pid,
                   "command":command,"formal_training_started":False,**clock()}
            atomic_json(folder/"process.json",state)
            atomic_json(folder/"status.json",state)
            code=child.wait()
        report=run/"probe_report.json"
        atomic_json(folder/"status.json",{**state,**clock(),"exit_code":code,
            "state":"probe_complete_waiting_budget" if code==0 and report.exists() else "failed"})
        raise SystemExit(code)

if __name__=="__main__": main()
