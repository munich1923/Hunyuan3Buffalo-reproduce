"""CPU-offloaded immutable FSDP checkpoints and streaming full-shard comparisons."""
from stage2_runtime import *
import gc,random,struct,pickle
import numpy as np
import torch
import torch.distributed as dist
from stage2_schedule import deadline_check


def trim():
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()


def local_rng():
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]);torch.cuda.set_rng_state(state["cuda"])


def options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    return StateDictOptions(full_state_dict=False,cpu_offload=True,strict=True)


def save(folder,model,optimizer,contract,state):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    rank=dist.get_rank();folder=path(folder);pending=folder.with_name(folder.name+".pending")
    if rank==0:
        if folder.exists() or pending.exists(): raise FileExistsError("immutable checkpoint exists: "+str(folder))
        pending.mkdir(parents=True)
    dist.barrier();start=time.monotonic()
    rng=local_rng();optimizer.zero_grad(set_to_none=True);trim()
    model_state,optim_state=get_state_dict(model,optimizer,options=options())
    dcp.save({"model":model_state,"optimizer":optim_state},checkpoint_id=pending/"distributed")
    del model_state,optim_state;trim()
    save_tensor(pending/f"rank_{rank}.pt",{"contract":contract,"state":state,"rng":rng})
    dist.barrier()
    names=sorted(p for p in pending.rglob("*") if p.is_file())
    payload={str(p.relative_to(pending)):{"bytes":p.stat().st_size,"sha256":digest(p)}
             for i,p in enumerate(names) if i%dist.get_world_size()==rank}
    records=[None]*dist.get_world_size();dist.all_gather_object(records,payload)
    if rank==0:
        files={k:v for record in records for k,v in record.items()}
        atomic_json(pending/"complete.json",{"contract":contract,"world_size":2,
                    "position":state["position"],"scheduler":state["scheduler"],"files":files,**clock()})
        os.rename(pending,folder)
        fd=os.open(str(folder.parent),os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    dist.barrier();restore_rng(rng);trim()
    return {"seconds":time.monotonic()-start,
            "bytes":sum(v["bytes"] for v in read_json(folder/"complete.json")["files"].values())}


def verify_payload(folder,contract):
    folder=path(folder);complete=read_json(folder/"complete.json")
    if complete["contract"]!=contract or complete["world_size"]!=2:
        raise RuntimeError("DCP contract or world-size changed")
    error=None
    for i,(name,record) in enumerate(sorted(complete["files"].items())):
        if i%dist.get_world_size()==dist.get_rank():
            p=path(folder/name)
            if p.stat().st_size!=record["bytes"] or digest(p)!=record["sha256"]:
                error="checkpoint checksum mismatch: "+name;break
    failures=[None]*2;dist.all_gather_object(failures,error)
    if any(failures): raise RuntimeError(str(failures))
    return complete


def load(folder,model,optimizer,contract,restore_random=True):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict,get_model_state_dict,set_model_state_dict
    folder=path(folder);verify_payload(folder,contract);trim()
    if optimizer is None:
        state={"model":get_model_state_dict(model,options=options())}
        dcp.load(state,checkpoint_id=folder/"distributed")
        result=set_model_state_dict(model,state["model"],options=options())
    else:
        m,o=get_state_dict(model,optimizer,options=options())
        state={"model":m,"optimizer":o};del m,o
        dcp.load(state,checkpoint_id=folder/"distributed")
        result=set_state_dict(model,optimizer,model_state_dict=state["model"],
                              optim_state_dict=state["optimizer"],options=options())
    if result.missing_keys or result.unexpected_keys: raise RuntimeError(str(result))
    del state;trim()
    local=torch.load(folder/f"rank_{dist.get_rank()}.pt",map_location="cpu",weights_only=False)
    if local["contract"]!=contract: raise RuntimeError("rank contract changed")
    if restore_random: restore_rng(local["rng"])
    dist.barrier()
    return local["state"]


def tensors(model,optimizer):
    # Original parameters are local shards with use_orig_params=True. Includes
    # every element and all Adam tensors, including empty shards and step.
    for name,p in model.named_parameters():
        yield "parameter/"+name,p.detach()
        if optimizer is not None:
            for key,value in sorted(optimizer.state.get(p,{}).items()):
                if torch.is_tensor(value): yield "adam/"+name+"/"+key,value.detach()
                else: raise TypeError("unexpected non-tensor Adam state")
    for name,value in model.named_buffers(): yield "buffer/"+name,value.detach()


def write_record(stream,value):
    import io
    buffer=io.BytesIO();torch.save(value,buffer)
    view=buffer.getbuffer();stream.write(struct.pack("<Q",len(view)));stream.write(view)


def read_record(stream):
    import io
    header=stream.read(8)
    if len(header)!=8: raise RuntimeError("incomplete snapshot frame header")
    size=struct.unpack("<Q",header)[0]
    if size>2*1024**3: raise RuntimeError("unexpected snapshot tensor size")
    value=stream.read(size)
    if len(value)!=size: raise RuntimeError("incomplete snapshot frame")
    return torch.load(io.BytesIO(value),map_location="cpu",weights_only=False)


def snapshot(file,model,optimizer,summary):
    file=path(file);file.parent.mkdir(parents=True,exist_ok=True)
    count=elements=0
    with file.open("wb") as stream:
        for name,tensor in tensors(model,optimizer):
            write_record(stream,{"name":name,"tensor":tensor.cpu()})
            count+=1;elements+=tensor.numel()
        write_record(stream,{"end":True,"count":count,"elements":elements,"summary":summary})
        stream.flush();os.fsync(stream.fileno())
    return {"tensors":count,"elements":elements,"bytes":file.stat().st_size}


def compare_snapshot(file,model,optimizer,summary,atol=1e-6,rtol=1e-5):
    max_error=0.;count=elements=0
    with path(file).open("rb") as stream:
        for name,tensor in tensors(model,optimizer):
            record=read_record(stream)
            ref=record.get("tensor");actual=tensor.cpu()
            if record.get("name")!=name or ref.shape!=actual.shape or ref.dtype!=actual.dtype:
                raise RuntimeError("local shard schema mismatch: "+name)
            if not torch.isfinite(actual).all() or not torch.allclose(actual,ref,atol=atol,rtol=rtol):
                raise RuntimeError("full local shard comparison failed: "+name)
            if actual.numel(): max_error=max(max_error,float((actual-ref).abs().max()))
            count+=1;elements+=actual.numel()
        end=read_record(stream)
        if not end.get("end") or end["count"]!=count or end["elements"]!=elements:
            raise RuntimeError("local shard stream incomplete")
        expected=end["summary"]
        for key in ["uids","position","rng_sha256","training_rng_sha256"]:
            if expected[key]!=summary[key]: raise RuntimeError("resume mismatch: "+key)
        for key in ["flow_mse","moe_aux","total_loss","gradient_norm"]:
            if not np.isclose(expected[key],summary[key],atol=atol,rtol=rtol):
                raise RuntimeError("next-update scalar mismatch: "+key)
    return {"passed":True,"tensors":count,"elements":elements,"max_absolute_error":max_error,
            "atol":atol,"rtol":rtol,"data_rng_loss_checked":True}


def rng_digest(value):
    import io,hashlib
    stream=io.BytesIO();torch.save(value,stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()

