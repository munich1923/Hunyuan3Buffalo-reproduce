"""CPU-only preparation supervisor; records completion/failure and never starts GPU work."""
from stage2_runtime import *


def main():
    p=parser(__doc__);args=p.parse_args();cfg=config(args.config)
    folder=path(cfg["output_root"])/"preparation"
    with lock("environment/stage2_prepare_runner.lock"):
        started=time.monotonic()
        command=[sys.executable,"-B","-u",str(ROOT/"scripts/prepare_stage2.py"),
                 "--config",str(path(args.config))]
        atomic_json(folder/"dataset_status.json",{"state":"running","pid":os.getpid(),
                    "command":command,**clock()})
        with (folder/"dataset.log").open("a") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                      env={**os.environ,"CUDA_VISIBLE_DEVICES":""})
            code=process.wait()
        atomic_json(folder/"dataset_status.json",{"state":"ready" if code==0 else "failed",
            "pid":os.getpid(),"exit_code":code,"elapsed_seconds":time.monotonic()-started,
            "next":"explicit GPU assignment required; no automatic GPU stage",**clock()})

if __name__=="__main__":main()
