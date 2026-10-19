"""Versioned cache contracts; cached geometry is supervision, never a model condition."""
from stage2_runtime import *
import torch


def text_contract(cfg,identity):
    return {"kind":"text","version":cfg["cache"]["version"],
            "selection":cfg["cache"]["selection"],"model":identity,
            "layer":-1,"dtype":"bfloat16","hidden_dim":cfg["cache"]["hidden_dim"],
            "no_images":True,"no_answer_generation":True}


def shape_contract(cfg,identity):
    return {"kind":"shape","version":cfg["cache"]["version"],
            "model":identity,"dtype":"float32","encoder_autocast":"float16",
            "normalization":cfg["dataset"]["geometry_version"],
            "extent":cfg["dataset"]["normalized_extent"],
            "sample_points":cfg["dataset"]["sample_points"],
            "latent_shape":cfg["cache"]["latent_shape"],"scale":cfg["cache"]["scale"],
            "logvar_clamp":[-30.0,20.0]}


def cache_base(cfg,kind,create=False):
    identity=model_identity(cfg,"vlm" if kind=="text" else "shape")
    contract=(text_contract if kind=="text" else shape_contract)(cfg,identity)
    key=fingerprint(contract)
    folder=path(cfg["cache"]["root"])/kind/key
    if create:
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/"contract.json",contract)
    elif not (folder/"contract.json").exists() or read_json(folder/"contract.json")!=contract:
        raise RuntimeError(f"Missing/current {kind} cache; run cache stage first")
    return folder,key


def text_key(row,contract):
    return fingerprint({"contract":contract,"prompt":row["prompt"]})


def shape_key(row,contract):
    return fingerprint({"contract":contract,"mesh":row["mesh_sha256"],
                        "normalization":row["normalization"],"seed":row["surface_seed"]})


def load_cached(cfg,row,split,bases=None):
    bases=bases or {k:cache_base(cfg,k) for k in ["text","shape"]}
    text=torch.load(bases["text"][0]/split/(row["uid"]+".pt"),
                    map_location="cpu",weights_only=True)
    shape=torch.load(bases["shape"][0]/split/(row["uid"]+".pt"),
                     map_location="cpu",weights_only=True)
    if text["fingerprint"]!=text_key(row,bases["text"][1]):
        raise RuntimeError("Stale text cache: "+row["uid"])
    if shape["fingerprint"]!=shape_key(row,bases["shape"][1]):
        raise RuntimeError("Stale posterior cache: "+row["uid"])
    hidden=text["hidden"]; mean=shape["mean"]; logvar=shape["logvar"]
    if hidden.ndim!=2 or hidden.shape[1]!=cfg["cache"]["hidden_dim"] or hidden.shape[0]<1:
        raise ValueError("Invalid hidden shape")
    if list(mean.shape)!=cfg["cache"]["latent_shape"] or logvar.shape!=mean.shape:
        raise ValueError("Invalid posterior shape")
    if not all(torch.isfinite(x).all() for x in [hidden,mean,logvar]):
        raise ValueError("Nonfinite cache")
    return hidden[None],mean[None],logvar[None]


def tokenize_condition(tokenizer,prompt):
    messages=[{"role":"user","content":[{"type":"text","text":prompt}]}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    marker="<|im_start|>user\n"
    if rendered.count(marker)!=1:
        raise ValueError("Ambiguous prompt/template boundary")
    start=rendered.index(marker)+len(marker); end=start+len(prompt)
    if rendered[start:end]!=prompt or not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("Unexpected Qwen chat template")
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True,
                      return_tensors="pt")
    offsets=encoded.pop("offset_mapping")[0].tolist()
    selected=[i for i,(a,b) in enumerate(offsets) if b>start and a<end and b>a]
    selected.append(len(offsets)-1)
    selected=sorted(set(selected))
    if not selected: raise ValueError("No condition tokens")
    return encoded,selected,rendered
