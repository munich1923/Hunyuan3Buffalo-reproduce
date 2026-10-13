"""Controlled comparisons on shared attempted samples, retaining validity/completeness."""
import json,numpy as np
from remote_runtime import ROOT
MODES=["source","vae","native","random","connector","lora","no_source","noop","shuffled"]
def main():
    records={}
    for mode in MODES:
        path=ROOT/"results/evaluation"/mode/"metrics.jsonl"
        if path.exists():records[mode]={(x["id"],x["seed"]):x for x in map(json.loads,path.read_text().splitlines())}
    report={"expected_modes":MODES,"available_modes":list(records),"note":"Use shared attempted samples for partial runs. Failed meshes remain invalid and are never assigned zero distance."}
    shared=set.intersection(*(set(x) for x in records.values())) if len(records)==len(MODES) else set()
    report["shared_attempted_samples_all_modes"]=len(shared)
    report["shared_all_modes"]={}
    for mode,rows in records.items():
        values=[rows[key] for key in shared];valid=[x for x in values if x["valid"]]
        result={"attempted":len(values),"valid":len(valid),"valid_rate":len(valid)/max(1,len(values))}
        if valid:
            result["target_cd_squared_sum"]=float(np.mean([x["target_metrics"]["cd_squared_sum"] for x in valid]))
            result["target_fscore_005"]=float(np.mean([x["target_metrics"]["fscore_005"] for x in valid]))
            result["source_fscore_005"]=float(np.mean([x["source_metrics"]["fscore_005"] for x in valid]))
        report["shared_all_modes"][mode]=result
    report["paired_comparisons"]=[]
    for first,second in [("connector","random"),("lora","connector"),("lora","native"),("lora","source"),("lora","no_source"),("lora","noop"),("lora","shuffled")]:
        if first not in records or second not in records:continue
        keys=set(records[first])&set(records[second])
        valid=[key for key in keys if records[first][key]["valid"] and records[second][key]["valid"]]
        result={"first":first,"second":second,"shared_attempted":len(keys),"both_valid":len(valid),"note":"Negative CD delta and positive F-score delta favor first against the edited target; noop is also separately assessed against source."}
        if valid:
            for metric in ["cd_squared_sum","fscore_005"]:
                result["mean_delta_"+metric]=float(np.mean([records[first][k]["target_metrics"][metric]-records[second][k]["target_metrics"][metric] for k in valid]))
        report["paired_comparisons"].append(result)
    (ROOT/"results/summaries/controlled_comparisons.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
