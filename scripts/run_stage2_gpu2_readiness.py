"""Authorized GPU2 preparation and finite resource probe; never starts formal training."""
from stage2_runtime import *

def main():
    folder=path("results/stage2_nano10k/gpu2_readiness_20261021")
    folder.mkdir(parents=True,exist_ok=True)
    os.environ["STAGE2_ALLOW_SHARED_GPU"]="2"
    commands=[
      ["cache_stage2_text.py","--gpu","2"],
      ["cache_stage2_shape.py","--gpu","2"],
      ["evaluate_stage2.py","--mode","vae","--gpu","2"],
      ["train_stage2.py","--probe-only","--run-id","probe_gpu2_20261021","--gpu","2"]]
    with lock("environment/stage2_gpu2_readiness.lock"):
        atomic_json(folder/"provenance.json",{**provenance(config("configs/stage2_nano10k.yaml")),
                    "commands":commands,"formal_training_authorized":False})
        for i,command in enumerate(commands):
            state={"state":"running","stage":command[0],"stage_index":i,"pid":os.getpid(),**clock()}
            atomic_json(folder/"status.json",state)
            with (folder/(command[0]+".log")).open("a") as log:
                process=subprocess.Popen(["bash",str(path("scripts/stage2_launch.sh")),*command],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                state["child_pid"]=process.pid;atomic_json(folder/"status.json",state)
                code=process.wait()
            append(folder/"stages.jsonl",{"command":command,"exit_code":code})
            if code:
                atomic_json(folder/"status.json",{**state,**clock(),"state":"failed","exit_code":code})
                raise SystemExit(code)
        atomic_json(folder/"status.json",{"state":"probe_complete_waiting_budget",**clock(),
                    "formal_training_started":False})

if __name__=="__main__": main()
