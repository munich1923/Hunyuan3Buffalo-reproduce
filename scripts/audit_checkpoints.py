"""CPU checkpoint audit after training; no CUDA initialization."""
import hashlib,json,torch
from remote_runtime import ROOT
def main():
    state=json.loads((ROOT/"environment/run_state.json").read_text())
    if state.get("current",{}).get("name") in ["overfit","connector","lora"]:
        raise RuntimeError("Training is still writing checkpoints; audit only after training")
    reports=[]
    for path in sorted((ROOT/"results/checkpoints").glob("*.pt")):
        data=torch.load(path,map_location="cpu",weights_only=True);weights=data["adapter"]
        unexpected=[k for k in weights if not (k.startswith("connector.") or k.endswith(".a.weight") or k.endswith(".b.weight"))]
        assert not unexpected,unexpected
        assert all(torch.isfinite(v).all() for v in weights.values()),path
        assert data["source_commit"]=="82920d643c0dc2f7bfd7255f45f62d386edfe60c",path
        assert data["seed"]==20261013,path
        assert any(k.startswith("connector.") for k in weights),path
        digest=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):digest.update(chunk)
        reports.append({"path":str(path),"sha256":digest.hexdigest(),"bytes":path.stat().st_size,
                        "stage":data["stage"],"step":data["step"],"adapter_tensors":len(weights),
                        "adapter_parameters":sum(v.numel() for v in weights.values()),"finite":True,"frozen_backbone_absent":True})
    (ROOT/"results/summaries/checkpoint_integrity.json").write_text(json.dumps({"ok":True,"checkpoints":reports},indent=2)+"\n")
    print("Audited checkpoints",len(reports))
if __name__=="__main__":main()
