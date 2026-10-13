"""CPU-only paired output differences for instruction/source ablations."""
import json,numpy as np,trimesh
from remote_runtime import ROOT
from metrics import compare
from compare_evaluations import bootstrap_delta
def main():
    modes=["lora","noop","shuffled","no_source"]
    records={}
    for mode in modes:
        path=ROOT/"results/evaluation"/mode/"metrics.jsonl"
        if path.exists():records[mode]={(x["id"],x["seed"]):x for x in map(json.loads,path.read_text().splitlines())}
    rows=[]
    for other in modes[1:]:
        if "lora" not in records or other not in records:continue
        for k in sorted(set(records["lora"])&set(records[other])):
            a,b=records["lora"][k],records[other][k]
            r={"id":k[0],"seed":k[1],"comparison":"lora_vs_"+other,"operation":a["operation"],"both_valid":a["valid"] and b["valid"]}
            if r["both_valid"]:
                r["output_metrics"]=compare(trimesh.load(a["path"],force="mesh"),trimesh.load(b["path"],force="mesh"),k[1])
            rows.append(r)
    folder=ROOT/"results/summaries"
    (folder/"instruction_sensitivity_records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary={"note":"Differences between generated meshes at the same seed measure sensitivity, not instruction correctness; nonzero CD occurs even for identical meshes due to surface sampling.","comparisons":{}}
    for other in modes[1:]:
        selected=[r for r in rows if r["comparison"]=="lora_vs_"+other]
        summary["comparisons"][other]={}
        for op in ["add","remove","all"]:
            valid=[r for r in selected if r["both_valid"] and (op=="all" or r["operation"]==op)]
            groups={}
            for r in valid:
                groups.setdefault(r["id"],[]).append(r["output_metrics"])
            result={"attempted":sum(op=="all" or r["operation"]==op for r in selected),"both_valid":len(valid),"paired_objects":len(groups)}
            for m in ["cd_squared_sum","fscore_005"]:
                values=[np.mean([v[m] for v in g]) for g in groups.values()]
                if values:result[m]=float(np.mean(values))
            summary["comparisons"][other][op]=result
    (folder/"instruction_sensitivity.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
