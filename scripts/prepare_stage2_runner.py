"""Persistent CPU preparation supervisor; no scheduler, GPU work or duplicate workers."""
from stage2_runtime import *


def supervise(folder, command):
    started=time.monotonic()
    attempt=0
    while True:
        attempt+=1
        atomic_json(folder/"dataset_status.json",{"state":"running","pid":os.getpid(),
                    "attempt":attempt,"command":command,**clock()})
        with (folder/"dataset.log").open("a") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                      env={**os.environ,"CUDA_VISIBLE_DEVICES":""})
            code=process.wait()
        # Network failures retry indefinitely; integrity/QC/config errors stay visible.
        if code==75:
            delay=min(60*2**min(attempt-1,3),300)
            atomic_json(folder/"dataset_status.json",{"state":"retry_wait","pid":os.getpid(),
                "attempt":attempt,"exit_code":code,"retry_seconds":delay,
                "elapsed_seconds":time.monotonic()-started,**clock()})
            append(folder/"supervisor_events.jsonl",{"event":"network_retry", "attempt":attempt,
                   "exit_code":code,"retry_seconds":delay})
            time.sleep(delay)
            continue
        atomic_json(folder/"dataset_status.json",{"state":"ready" if code==0 else "failed",
            "pid":os.getpid(),"attempt":attempt,"exit_code":code,
            "elapsed_seconds":time.monotonic()-started,
            "next":"explicit GPU assignment required; no automatic GPU stage",**clock()})
        return code


def main():
    p=parser(__doc__)
    p.add_argument("--wait-for-pid",type=int,help="Adopt after an existing project supervisor exits")
    args=p.parse_args();cfg=config(args.config)
    folder=path(cfg["output_root"])/"preparation"
    folder.mkdir(parents=True,exist_ok=True)
    if args.wait_for_pid:
        proc=Path(f"/proc/{args.wait_for_pid}/cmdline")
        identity=proc.read_bytes() if proc.exists() else None
        if identity and str(ROOT/"scripts/prepare_stage2_runner.py").encode() not in identity:
            raise RuntimeError("Refusing to wait for unrelated process")
        atomic_json(folder/"guardian_status.json",{"state":"waiting_for_current_runner",
                    "pid":os.getpid(),"adopting_pid":args.wait_for_pid,**clock()})
        while identity and proc.exists() and proc.read_bytes()==identity:
            time.sleep(10)
        atomic_json(folder/"guardian_status.json",{"state":"adopting", "pid":os.getpid(),**clock()})
    with lock("environment/stage2_prepare_runner.lock"):
        command=[sys.executable,"-B","-u",str(ROOT/"scripts/prepare_stage2.py"),
                 "--config",str(path(args.config))]
        # Legacy in-flight worker reports network exhaustion as 1; a fresh worker
        # classifies it as 75. One adoption rerun preserves all receipts/partials.
        return supervise(folder,command)

if __name__=="__main__":raise SystemExit(main())
