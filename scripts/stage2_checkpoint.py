"""Full immutable checkpoints. Model parameters + Adam + data/RNG position."""
from stage2_runtime import *
import random
import numpy as np
import torch


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]: torch.cuda.set_rng_state_all(state["cuda"])


def code_contract():
    paths=sorted((ROOT/"scripts").glob("*stage2*.py"))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def checkpoint_contract(cfg,bases):
    ready=read_json(manifest_root(cfg)/"ready.json")
    return {"config":fingerprint(cfg),"dataset":ready,
            "text_contract":bases["text"][1],"shape_contract":bases["shape"][1],
            "code":code_contract()}


def save_checkpoint(folder,model,optimizer,position,contract,generator,stats):
    folder=path(folder)
    if folder.exists(): raise FileExistsError("Checkpoint never overwritten: "+str(folder))
    temporary=folder.with_name(folder.name+f".{os.getpid()}.tmp")
    temporary.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
           "position":position,"contract":contract,"rng":rng_state(),
           "training_generator":generator.get_state(),"stats":stats}
    with (temporary/"state.pt").open("wb") as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    checksum=digest(temporary/"state.pt")
    atomic_json(temporary/"complete.json",{"sha256":checksum,
                "bytes":(temporary/"state.pt").stat().st_size,
                "position":position,"contract":contract,**clock()})
    folder.parent.mkdir(parents=True,exist_ok=True)
    os.rename(temporary,folder)
    return {"seconds":time.monotonic()-started,"bytes":(folder/"state.pt").stat().st_size}


def load_checkpoint(folder,contract=None):
    folder=path(folder)
    complete=read_json(folder/"complete.json")
    if digest(folder/"state.pt")!=complete["sha256"]:
        raise RuntimeError("Checkpoint checksum mismatch")
    if contract is not None and complete["contract"]!=contract:
        raise RuntimeError("Resume contract changed; create a new run")
    # Only this project's locally generated checkpoint, whose checksum was verified.
    state=torch.load(folder/"state.pt",map_location="cpu",weights_only=False)
    if state["contract"]!=complete["contract"] or state["position"]!=complete["position"]:
        raise RuntimeError("Checkpoint metadata mismatch")
    return state
