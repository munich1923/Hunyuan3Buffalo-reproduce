"""Run optional8-pair geometry diagnostics through the same lock/GPU budget."""
import fcntl,json,sys
from remote_runtime import ROOT
from run_pilot import run,readstate,atomic,utc
def main():
    lock=(ROOT/"environment/pilot.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=readstate()
    if state["status"]!="completed":raise RuntimeError("Finish the main pilot before running geometry diagnostics")
    try:
        for checkpoint in ["overfit","lora"]:
            run(state,"train_geometry_"+checkpoint,ROOT/"environment/venvs/shape/bin/python","evaluate_training_fit.py",["--checkpoint",checkpoint,"--seconds","1500"],gpu=True,cap=1800)
        state.update(status="completed",training_geometry_diagnostics=True,updated_utc=utc());atomic(state)
    except Exception as e:
        state.update(status="stopped",error=repr(e),updated_utc=utc());atomic(state);raise
if __name__=="__main__":main()
