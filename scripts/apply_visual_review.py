"""Apply sampled agent visual exclusions reproducibly, preserving original split identities."""
import copy,hashlib,json,random,re
from remote_runtime import ROOT
def apply():
    folder=ROOT/"assets/manifests";out=ROOT/"results/summaries"
    decisions=json.loads((out/"visual_review_decisions.json").read_text())
    exclusions={x["uid"]:x["observation"] for x in decisions["observations"] if x["decision"]=="exclude"}
    snapshot=folder/"before_visual_review";snapshot.mkdir(exist_ok=True)
    for split in ["train","val","test"]:
        target=snapshot/(split+".jsonl")
        if not target.exists():
            tracked=out/("original_"+split+"_manifest.jsonl")
            target.write_bytes((tracked if tracked.exists() else folder/(split+".jsonl")).read_bytes())
    for split in ["val","test"]:
        (folder/(split+".jsonl")).write_bytes((snapshot/(split+".jsonl")).read_bytes())
    rows=[json.loads(x) for x in (snapshot/"train.jsonl").read_text().splitlines()]
    val=[json.loads(x) for x in (snapshot/"val.jsonl").read_text().splitlines()]
    candidates=[json.loads(x) for x in (folder/"accepted_candidates.jsonl").read_text().splitlines()]
    oldpath=out/"visual_review_replacements.json"
    prior={x["excluded"]:x["replacement"] for x in json.loads(oldpath.read_text())["replacements"]} if oldpath.exists() else {}
    lookup={x["uid"]:x for x in candidates}
    retained={v for v in prior.values() if v not in exclusions}
    used={x["group"] for x in rows+val}|retained
    candidates=sorted([x for x in candidates if x["uid"] not in used and x["uid"] not in exclusions],key=lambda x:x["uid"])
    random.Random(20261013).shuffle(candidates)
    replacements=[];result=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if a["group"] not in exclusions:result.extend([a,b]);continue
        prev=prior.get(a["group"])
        x=lookup[prev] if prev in retained else candidates.pop(0)
        rec={**x,"id":x["uid"]+"_add","group":x["uid"],"operation":"add","dataset":"Nano3D-Edit-100k"}
        inv={**rec,"id":x["uid"]+"_remove","operation":"remove","source":x["target"],"target":x["source"],"instruction":"Remove the following added parts: "+re.sub(r"^add\s+","",x["instruction"],flags=re.I).rstrip(".")+".","inverted_from":rec["id"],"source_geometry_hash":x["target_geometry_hash"],"target_geometry_hash":x["source_geometry_hash"],"source_sha256":x["target_sha256"],"target_sha256":x["source_sha256"],"qc_original_forward":x["qc"],"qc":x["inverse_qc"]}
        result.extend([rec,inv]);replacements.append({"excluded":a["group"],"replacement":x["uid"],"reason":exclusions[a["group"]]})
    assert len(result)==1024 and len({x["group"] for x in result})==512
    assert not ({x["group"] for x in result}&{x["group"] for x in val})
    (folder/"train.jsonl").write_text("".join(json.dumps(x)+"\n" for x in result))
    provenance={"replacements":replacements,"excluded_pairs":len(replacements),"reviewer":decisions["reviewer"],"manifest_sha256":{s:hashlib.sha256((folder/(s+".jsonl")).read_bytes()).hexdigest() for s in ["train","val","test"]},"original_manifest_sha256":{s:hashlib.sha256((snapshot/(s+".jsonl")).read_bytes()).hexdigest() for s in ["train","val","test"]}}
    (out/"visual_review_replacements.json").write_text(json.dumps(provenance,indent=2)+"\n")
    for split in ["train","val","test"]:
        (out/(split+"_manifest.jsonl")).write_bytes((folder/(split+".jsonl")).read_bytes())
    print(json.dumps(provenance),flush=True)
if __name__=="__main__":apply()
