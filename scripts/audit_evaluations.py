"""CPU audit of evaluated record uniqueness, geometry files and metric completeness."""
import hashlib,json,numpy as np,trimesh
from pathlib import Path
from remote_runtime import ROOT
from compare_evaluations import MODES
def main():
    rows=[json.loads(x) for x in (ROOT/"assets/manifests/test_prepared.jsonl").read_text().splitlines()]
    expected={(r["id"],s) for r in rows for s in [20261013,20261014,20261015]}
    report={"expected_per_mode":len(expected),"modes":{},"errors":[]}
    for mode in MODES:
        folder=ROOT/"results/evaluation"/mode;path=folder/"metrics.jsonl"
        if not path.exists():
            report["modes"][mode]={"available":False};continue
        records=[json.loads(x) for x in path.read_text().splitlines()];seen=set();valid=0;receipts=[]
        for r in records:
            k=(r["id"],r["seed"])
            if k in seen:report["errors"].append(mode+":duplicate:"+str(k))
            seen.add(k)
            if k not in expected:report["errors"].append(mode+":unexpected:"+str(k))
            if r["valid"]:
                mesh_path=Path(r["path"])
                try:
                    m=trimesh.load(mesh_path,force="mesh")
                    assert len(m.vertices)>0 and len(m.faces)>0 and np.isfinite(m.vertices).all()
                    assert m.faces.min()>=0 and m.faces.max()<len(m.vertices)
                    for section in ["target_metrics","source_metrics"]:
                        for metric in ["cd_squared_sum","fscore_001","fscore_0025","fscore_005"]:
                            value=r[section][metric];assert np.isfinite(value) and value>=0
                            if metric.startswith("fscore"):assert value<=1
                    valid+=1;receipts.append({"id":r["id"],"seed":r["seed"],"sha256":hashlib.sha256(mesh_path.read_bytes()).hexdigest(),"vertices":len(m.vertices),"faces":len(m.faces)})
                except Exception as e:report["errors"].append(mode+":"+str(k)+":"+repr(e))
            elif not r.get("error"):report["errors"].append(mode+":invalid mesh has no error:"+str(k))
        summary=json.loads((folder/"summary.json").read_text()) if (folder/"summary.json").exists() else None
        if summary and (summary["attempted"]!=len(records) or summary["complete"]!=(seen==expected)):
            report["errors"].append(mode+":summary count mismatch")
        report["modes"][mode]={"available":True,"attempted":len(records),"unique_attempted":len(seen),"valid":valid,"complete":seen==expected,"render_errors":sum(bool(r.get("render_error")) for r in records),"mesh_receipts":receipts}
    report["ok"]=not report["errors"]
    (ROOT/"results/summaries/evaluation_integrity.json").write_text(json.dumps(report,indent=2)+chr(10))
    print(json.dumps({"ok":report["ok"],"errors":report["errors"],"counts":{k:{a:b for a,b in v.items() if a!="mesh_receipts"} for k,v in report["modes"].items()}},indent=2))
    if not report["ok"]:raise RuntimeError("Evaluation audit failed")
if __name__=="__main__":main()
