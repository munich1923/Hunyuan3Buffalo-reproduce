"""Version-pinned downloads with payload checks and durable receipts."""
import fcntl,functools
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,subprocess,time,urllib.request
from remote_runtime import ROOT,configure
configure()
LOGS=ROOT/"logs/downloads";LOGS.mkdir(parents=True,exist_ok=True)
def api(repo,kind="models"):
    url="https://hf-mirror.com/api/"+kind+"/"+repo+"?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=45) as r:meta=json.load(r)
    name=repo.replace("/","--")
    (ROOT/"environment"/("manifest-"+name+".json")).write_text(json.dumps(meta,indent=2)+"\n")
    return meta
def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def _download(repo,kind,revision,item,base):
    name=item["rfilename"];path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
    lfs=item.get("lfs") or {};expected=lfs.get("sha256");size=lfs.get("size",item.get("size"))
    key=hashlib.sha256((repo+"/"+name).encode()).hexdigest()[:16]
    receipt=LOGS/(key+".json");log=LOGS/(key+".log")
    data={"repo":repo,"kind":kind,"revision":revision,"file":name,"local_path":str(path),"expected_sha256":expected,"expected_bytes":size,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():receipt.write_text(json.dumps(data,indent=2)+"\n")
    if path.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get("status")=="verified" and old.get("actual_bytes")==path.stat().st_size and old.get("revision")==revision:
            print("CACHED",repo,name,flush=True);return True
    save();partial=path.with_name(path.name+".partial")
    prefix="datasets/" if kind=="datasets" else ""
    url="https://hf-mirror.com/"+prefix+repo+"/resolve/"+revision+"/"+name
    cmd=["aria2c","--continue=true","--allow-overwrite=true","--auto-file-renaming=false","--file-allocation=none","--max-connection-per-server=8","--split=8","--min-split-size=16M","--max-tries=3","--retry-wait=5","--connect-timeout=20","--timeout=60","--summary-interval=30","--console-log-level=warn","--dir",str(partial.parent),"--out",partial.name,url]
    with log.open("a") as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    data["exit_code"]=p.returncode
    if p.returncode:
        data["status"]="failed";save();print("FAILED",repo,name,p.returncode,flush=True);return False
    actual=partial.stat().st_size;digest=sha256(partial)
    data.update(actual_bytes=actual,actual_sha256=digest)
    if (size is not None and actual!=size) or (expected and digest!=expected):
        data["status"]="checksum_or_size_failed";save();print("INVALID",repo,name,actual,size,flush=True);return False
    # Also reject LFS pointers and error documents masquerading as successful payloads.
    with partial.open("rb") as f:head=f.read(256)
    if name.endswith((".ckpt",".safetensors",".tar.gz",".glb")) and (head.startswith(b"version https://git-lfs") or head.lstrip().startswith((b"<html",b"<!DOCTYPE",b"<?xml"))):
        data["status"]="invalid_payload";save();return False
    partial.replace(path);data.update(status="verified",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    print("VERIFIED",repo,name,actual,digest,flush=True);return True
def download(repo,kind,revision,item,base):
    key=hashlib.sha256((repo+"/"+item["rfilename"]).encode()).hexdigest()[:16]
    with (LOGS/(key+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _download(repo,kind,revision,item,base)
def main():
    plans=[]
    for repo,kind,folder in [
        ("tencent/Hunyuan3D-2.1","models","models/Hunyuan3D-2.1"),
        ("Qwen/Qwen2.5-VL-7B-Instruct","models","models/Qwen2.5-VL-7B-Instruct"),
        ("yejunliang23/Nano3D-Edit-100k","datasets","assets/Nano3D-Edit-100k"),
        ("ziqima/Edit3D-Bench","datasets","assets/Edit3D-Bench"),
    ]:
        meta=api(repo,kind);sha=meta["sha"]
        files=meta["siblings"]
        if "Hunyuan" in repo:
            files=[x for x in files if x["rfilename"].startswith(("hunyuan3d-dit-v2-1/","hunyuan3d-vae-v2-1/")) or x["rfilename"] in ["README.md","LICENSE","Notice.txt"]]
        elif "Qwen" in repo:
            files=[x for x in files if not x["rfilename"].startswith(".")]
        elif "Nano" in repo:
            wanted={"README.md","v1_100k/info/v1_100k_info.jsonl"}
            wanted.update("v1_100k/editing_assets/part_%03d.tar.gz"%i for i in [0,33,66])
            files=[x for x in files if x["rfilename"] in wanted]
        else:
            files=[x for x in files if x["rfilename"] in ["README.md","metadata/add.csv","metadata/remove.csv"]]
        plans.extend((repo,kind,sha,x,ROOT/folder) for x in files)
    (ROOT/"environment/download_plan.json").write_text(json.dumps([{**dict(repo=r,kind=k,revision=s,base=str(b)),**x} for r,k,s,x,b in plans],indent=2)+"\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(lambda p:download(*p),plans))
    summary={"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"verified":sum(out),"total":len(out),"ok":all(out)}
    (ROOT/"environment/download_status.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)
    return 0 if all(out) else 1
if __name__=="__main__":raise SystemExit(main())
