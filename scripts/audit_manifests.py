import hashlib,json
from remote_runtime import ROOT
def main():
    rows={s:[json.loads(x) for x in (ROOT/"assets/manifests"/(s+".jsonl")).read_text().splitlines()] for s in ["train","val","test"]}
    hashes={s:{x[k] for x in r for k in ["source_geometry_hash","target_geometry_hash"] if k in x} for s,r in rows.items()}
    assert len(rows["train"])==1024 and len(rows["val"])==128 and len(rows["test"])==40
    assert len({x["group"] for x in rows["train"]})==512 and len({x["group"] for x in rows["val"]})==64
    assert not hashes["train"]&hashes["val"]
    reviewed=json.loads((ROOT/"environment/data_review.json").read_text())
    for s in rows:
        assert hashlib.sha256((ROOT/"assets/manifests"/(s+".jsonl")).read_bytes()).hexdigest()==reviewed["manifest_sha256"][s]
        for row in rows[s]:
            for role in ["source","target"]:
                path=__import__("pathlib").Path(row[role])
                assert path.exists()
                if role+"_sha256" in row:assert hashlib.sha256(path.read_bytes()).hexdigest()==row[role+"_sha256"]
    report={"passed":True,"entries":{s:len(r) for s,r in rows.items()},"train_validation_geometry_hash_overlap":0,"all_recorded_mesh_file_sha256_match":True,"review_manifest_sha256_match":True}
    (ROOT/"results/summaries/manifest_integrity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
if __name__=="__main__":main()
