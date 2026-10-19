"""VAE gate, pure-noise generation and reviewed checkpoint selection."""
from stage2_runtime import *


def checkpoint_folder(run,step):
    matches=list((run/"checkpoints").glob(f"step_{step:06d}*"))
    if len(matches)!=1: raise RuntimeError("Checkpoint not uniquely found")
    return matches[0]


def choose_candidates(run):
    from stage2_evaluation import reviewed_score
    candidates=[]
    for checkpoint in sorted((run/"checkpoints").glob("step_*")):
        meta=read_json(checkpoint/"complete.json")
        step=meta["position"]["update"]
        folder=run/"evaluation"/f"step_{step:06d}"
        score=reviewed_score(folder)
        val=read_json(folder/"validation.json")["flow_mse"]
        candidates.append({"checkpoint":str(checkpoint),"step":step,"score":score,
                           "validation_flow_mse":val})
    if not candidates: raise RuntimeError("No saved, reviewed checkpoints")
    candidates.sort(key=lambda r:(-r["score"]["structure_fraction"],
                    -r["score"]["subject_fraction"],r["validation_flow_mse"],r["step"]))
    selection={"candidates":candidates[:2],"criterion":
       "structure fraction, subject fraction, fixed validation MSE, earliest update"}
    atomic_json(run/"candidate_checkpoints.json",selection)
    return selection


def lock_selection(run):
    from stage2_evaluation import reviewed_score
    candidates=read_json(run/"candidate_checkpoints.json")["candidates"]
    reviewed=[]
    for candidate in candidates:
        folder=run/"evaluation"/f"heldout_val_{candidate['step']:06d}"
        score=reviewed_score(folder)
        reviewed.append({**candidate,"heldout_score":score})
    reviewed.sort(key=lambda r:(-r["heldout_score"]["structure_fraction"],
       -r["heldout_score"]["subject_fraction"],r["validation_flow_mse"],r["step"]))
    result={**reviewed[0],"selection_locked":True,**clock()}
    target=run/"selected_checkpoint.json"
    if target.exists() and read_json(target)["checkpoint"]!=result["checkpoint"]:
        raise RuntimeError("Selection already locked; do not change after test access")
    atomic_json(target,result)
    return result


def acceptance(cfg,run):
    from stage2_evaluation import reviewed_score,evaluation_protocol
    selected=read_json(run/"selected_checkpoint.json")
    step=selected["step"]
    result={"checkpoint":selected["checkpoint"],"passed":False}
    p=evaluation_protocol(cfg)
    for split in ["val","test"]:
        folder=run/"evaluation"/f"heldout_{split}_{step:06d}"
        score=reviewed_score(folder)
        pair_passed=0
        for i,pair in enumerate(p[split+"_pairs"]):
            f=folder/"reviews"/(pair["uid"]+f"__wrong_{pair['kind']}_{i:02d}.json")
            review=read_json(f)
            # Reviewer confirms distinguishability before judging the generated contrast.
            if (not review.get("reviewer") or
                review.get("pair_geometrically_distinguishable") is not True or
                type(review.get("condition_change_correct")) is not bool):
                raise RuntimeError("Condition pair review incomplete: "+str(f))
            pair_passed+=review["condition_change_correct"]
        result[split]={**score,"condition_pairs_passed":pair_passed}
    initial=reviewed_score(run/"evaluation"/"initial")
    chosen_monitor=reviewed_score(run/"evaluation"/f"step_{step:06d}")
    # Explicit train-monitor improvement review also needed; do not infer from training loss.
    improvement=read_json(run/"improvement_review.json")
    if not improvement.get("reviewer") or any(
        type(improvement.get(k)) is not bool for k in
        ["train_generation_improved","val_generation_improved",
         "train_condition_improved","val_condition_improved"]):
        raise RuntimeError("Recorded initial-versus-trained improvement review is required")
    e=cfg["evaluation"]
    result["passed"]=(all(result[s]["valid_fraction"]>=e["generated_valid_min"] and
       result[s]["condition_pairs_passed"]>=e["condition_pairs_min"] for s in ["val","test"])
       and result["test"]["subject_fraction"]>=e["subject_min"]
       and result["test"]["structure_fraction"]>=e["structure_min"]
       and all(improvement[k] for k in ["train_generation_improved","val_generation_improved",
                                      "train_condition_improved","val_condition_improved"]))
    result["initial_monitor"]=initial;result["trained_monitor"]=chosen_monitor
    result["improvement_review"]=improvement
    atomic_json(run/"acceptance.json",result)
    return result


def main():
    p=parser(__doc__)
    p.add_argument("--mode",choices=["vae","monitor","val","test","candidates",
                                   "lock-selection","acceptance"],required=True)
    p.add_argument("--run-id")
    p.add_argument("--checkpoint")
    p.add_argument("--gpu",type=int)
    args=p.parse_args();cfg=config(args.config)
    from stage2_evaluation import vae_gate,generation_suite
    if args.mode!="vae" and not args.run_id: p.error("--run-id is required")
    run=run_path(cfg,args.run_id) if args.run_id else path(cfg["output_root"])/"preparation"
    if args.mode in ["candidates","lock-selection","acceptance"]:
        function={"candidates":lambda:choose_candidates(run),
                  "lock-selection":lambda:lock_selection(run),
                  "acceptance":lambda:acceptance(cfg,run)}[args.mode]
        print(canonical(function()));return
    if args.gpu is None: p.error("GPU work requires explicit --gpu; no automatic device selection")
    with lock("environment/stage2_evaluate.lock"),gpu_lease(args.gpu,run):
        if args.mode=="vae":
            report=vae_gate(cfg,run/"vae_gate")
            print(canonical(report))
            if not report["passed"]: raise RuntimeError("VAE gate failed")
            return
        from stage2_cache import cache_base
        from stage2_model import load_model
        from stage2_checkpoint import load_checkpoint,checkpoint_contract
        bases={k:cache_base(cfg,k) for k in ["text","shape"]}
        checkpoint=args.checkpoint
        if args.mode=="test":
            selected=read_json(run/"selected_checkpoint.json")
            if not selected.get("selection_locked"): raise RuntimeError("Selection not locked")
            if checkpoint and path(checkpoint)!=path(selected["checkpoint"]):
                raise ValueError("Test checkpoint differs from locked selection")
            checkpoint=selected["checkpoint"]
        if not checkpoint: p.error("--checkpoint is required")
        checkpoint=path(checkpoint)
        if not checkpoint.is_relative_to(run/"checkpoints"):
            raise ValueError("Checkpoint must belong to this run")
        if args.mode in ["val","test"]:
            from stage2_evaluation import evaluation_protocol
            protocol=evaluation_protocol(cfg)
            review=read_json(manifest_root(cfg)/"condition_pair_review.json")
            if (review.get("protocol_fingerprint")!=protocol["fingerprint"] or
                not review.get("reviewer")):
                raise RuntimeError("Pre-generation geometric condition-pair review required")
            pairs=review[args.mode]
            expected_pairs=protocol[args.mode+"_pairs"]
            if len(pairs)!=len(expected_pairs) or any(
                pair.get("geometrically_distinguishable") is not True or
                any(pair.get(k)!=expected.get(k) for k in ["uid","wrong_uid","kind"])
                for pair,expected in zip(pairs,expected_pairs)):
                raise RuntimeError("Condition pairs not confirmed or differ from fixed protocol")
        if args.mode=="val":
            candidates=read_json(run/"candidate_checkpoints.json")["candidates"]
            if str(checkpoint) not in [c["checkpoint"] for c in candidates]:
                raise ValueError("Heldout validation limited to preselected candidates")
        state=load_checkpoint(checkpoint,checkpoint_contract(cfg,bases))
        model=load_model(cfg);model.load_state_dict(state["model"]);step=state["position"]["update"]
        del state
        folder=(run/"evaluation"/f"step_{step:06d}" if args.mode=="monitor" else
                run/"evaluation"/f"heldout_{args.mode}_{step:06d}")
        print(canonical(generation_suite(model,cfg,bases,folder,args.mode,step=step)))

if __name__=="__main__": main()
