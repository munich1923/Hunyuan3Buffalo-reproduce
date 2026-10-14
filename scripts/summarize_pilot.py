"""Summarize durable evidence; never promote unfinished stages to success."""
import datetime,hashlib,json
from remote_runtime import ROOT
def read(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else None
def main():
    state=read("environment/run_state.json")
    evidence={name:read("environment/"+name+".json") for name in ["vae_gate","training_overfit","training_connector","training_lora","data_review"]}
    evaluation={p.parent.name:json.loads(p.read_text()) for p in (ROOT/"results/evaluation").glob("*/summary.json")}
    summary={"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":state["status"],"current_stage":state.get("current",{}).get("name"),"stop_reason":state.get("error"),"gpu_allocated_seconds_completed_stages":state["gpu_seconds"],"gpu_budget_seconds":86400,"source_commit":"82920d643c0dc2f7bfd7255f45f62d386edfe60c","evidence":evidence,"evaluation":evaluation,"conclusions":{"interface":"Official-weight VAE/Qwen/MLP/source-latent DiT forward and backward passed; see interface_diagnostics.json.","training_fit":"Overfit report absent: pending." if evidence["training_overfit"] is None else ("8-pair deterministic velocity-loss fitting gate passed." if evidence["training_overfit"]["passed"] else "8-pair deterministic velocity-loss fitting gate failed; no scale-up permitted."),"unseen_editing":"No complete controlled evaluation: effectiveness is unproven." if len(evaluation)<9 or not all(x.get("complete",False) for x in evaluation.values()) else "Controlled metrics available; compare geometry and instruction controls before concluding."},"paths":{"logs":"logs/","results":"results/","weight_receipts":"logs/downloads/","manifests":"results/summaries/*_manifest.jsonl"}}
    summary["training_geometry"]={stage:read("results/summaries/training_geometry_"+stage+".json") for stage in ["overfit","lora"]}
    comparison=read("results/summaries/controlled_comparisons.json")
    sensitivity=read("results/summaries/instruction_sensitivity.json")
    summary["instruction_sensitivity"]=sensitivity
    if comparison:
        paired=next((r for r in comparison["paired_comparisons"] if r["first"]=="lora" and r["second"]=="source"),{})
        if paired.get("mean_delta_fscore_005",1)<0 and paired.get("mean_delta_cd_squared_sum",-1)>0:
            summary["conclusions"]["unseen_editing"]="LoRA is worse than unchanged source against edited targets; this approximation does not establish reliable instruction editing."
    for stage,data in summary["training_geometry"].items():
        if data and data["complete"]:
            summary["conclusions"]["geometry_fit_"+stage]="Generated geometry does not beat unchanged source F-score." if data.get("generated_fscore_005",0)<=data.get("unchanged_source_fscore_005",0) else "Generated target F-score exceeds unchanged source; inspect edit regions separately."
    folder=ROOT/"results/summaries"
    (folder/"experiment_status.json").write_text(json.dumps(summary,indent=2)+"\n")
    for key,data in evidence.items():
        if data is not None:(folder/(key+".json")).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["evidence","evaluation"]},indent=2))
if __name__=="__main__":main()
