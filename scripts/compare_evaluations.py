"""Controlled comparisons on shared attempted samples, retaining validity/completeness."""
import json,numpy as np
from remote_runtime import ROOT
MODES=["source","vae","native","random","connector","lora","no_source","noop","shuffled"]
def bootstrap_delta(records,first,second,keys,section,metric):
    groups={}
    for k in keys:
        a=records[first][k].get(section,{});b=records[second][k].get(section,{})
        if metric in a and metric in b:groups.setdefault(k[0],[]).append(a[metric]-b[metric])
    if not groups:return {}
    delta=np.array([np.mean(groups[k]) for k in sorted(groups)])
    result={"mean_delta_"+metric:float(delta.mean()),"paired_objects_"+metric:len(delta)}
    if len(delta)>1:
        rng=np.random.default_rng(20261013)
        boot=delta[rng.integers(0,len(delta),size=(1000,len(delta)))].mean(axis=1)
        result["object_bootstrap_95ci_"+metric]=[float(x) for x in np.quantile(boot,[.025,.975])]
    return result
def main():
    records={}
    for mode in MODES:
        path=ROOT/"results/evaluation"/mode/"metrics.jsonl"
        if path.exists():records[mode]={(x["id"],x["seed"]):x for x in map(json.loads,path.read_text().splitlines())}
    report={"expected_modes":MODES,"available_modes":list(records),"note":"Use shared attempted samples for partial runs. Failed meshes remain invalid and are never assigned zero distance."}
    shared=set.intersection(*(set(x) for x in records.values())) if len(records)==len(MODES) else set()
    report["shared_attempted_samples_all_modes"]=len(shared)
    report["shared_all_modes"]={}
    jointly_valid={k for k in shared if all(r[k]["valid"] for r in records.values())}
    report["jointly_valid_all_modes"]=len(jointly_valid)
    for mode,rows in records.items():
        values=[rows[key] for key in sorted(shared)];valid=[x for x in values if x["valid"]]
        result={"attempted":len(values),"valid":len(valid),"valid_rate":len(valid)/max(1,len(values))}
        if valid:
            result["target_cd_squared_sum"]=float(np.mean([x["target_metrics"]["cd_squared_sum"] for x in valid]))
            result["target_fscore_005"]=float(np.mean([x["target_metrics"]["fscore_005"] for x in valid]))
            result["source_fscore_005"]=float(np.mean([x["source_metrics"]["fscore_005"] for x in valid]))
        joint=[rows[k] for k in sorted(jointly_valid)]
        if joint:
            result["on_jointly_valid"]={m:float(np.mean([x["target_metrics"][m] for x in joint])) for m in ["cd_squared_sum","fscore_005"]}
        report["shared_all_modes"][mode]=result
    report["paired_comparisons"]=[]
    for first,second in [("connector","random"),("lora","connector"),("lora","native"),("lora","source"),("lora","no_source"),("lora","noop"),("lora","shuffled")]:
        if first not in records or second not in records:continue
        keys=set(records[first])&set(records[second])
        valid=[key for key in sorted(keys) if records[first][key]["valid"] and records[second][key]["valid"]]
        result={"first":first,"second":second,"shared_attempted":len(keys),"both_valid":len(valid),"note":"Negative CD delta and positive F-score delta favor first against the edited target; noop is also separately assessed against source."}
        if valid:
            result["by_operation"]={}
            for operation in ["add","remove"]:
                selected=[k for k in valid if records[first][k]["operation"]==operation]
                result["by_operation"][operation]={"both_valid":len(selected)}
                for metric in ["cd_squared_sum","fscore_005"]:
                    result["by_operation"][operation].update(bootstrap_delta(records,first,second,selected,"target_metrics",metric))
            result["surface_proxy_deltas"]={}
            for metric in ["unchanged_region_coverage_005","changed_target_coverage_005","removed_source_still_present_fraction_005"]:
                result["surface_proxy_deltas"].update(bootstrap_delta(records,first,second,valid,"preservation_metrics",metric))
            for metric in ["cd_squared_sum","fscore_005"]:
                groups={}
                for k in valid:
                    groups.setdefault(k[0],[]).append(records[first][k]["target_metrics"][metric]-records[second][k]["target_metrics"][metric])
                delta=np.array([np.mean(v) for v in groups.values()])
                result["mean_delta_"+metric]=float(delta.mean())
                result["paired_objects"]=len(delta)
                if len(delta)>1:
                    rng=np.random.default_rng(20261013)
                    boot=delta[rng.integers(0,len(delta),size=(1000,len(delta)))].mean(axis=1)
                    result["object_bootstrap_95ci_"+metric]=[float(x) for x in np.quantile(boot,[.025,.975])]
        report["paired_comparisons"].append(result)
    (ROOT/"results/summaries/controlled_comparisons.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
