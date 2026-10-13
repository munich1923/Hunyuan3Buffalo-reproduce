"""Version-pinned downloads with payload checks and durable receipts."""
import fcntl,functools
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,subprocess,time,urllib.request,urllib.parse
from remote_runtime import ROOT,configure,network_proxy
configure()
LOGS=ROOT/"logs/downloads";LOGS.mkdir(parents=True,exist_ok=True)
PINNED_REVISIONS={
 "tencent/Hunyuan3D-2.1":"0b94677654c57bb9a6b6845cd7b704ccf551d327",
 "Qwen/Qwen2.5-VL-7B-Instruct":"cc594898137f460bfe9f0759e9844b3ce807cfb5",
 "yejunliang23/Nano3D-Edit-100k":"d6686ff82ea14c2dfa4fdd55db332309a0241495",
 "ziqima/Edit3D-Bench":"30ace7d3b0e834189b514cf3753cd832825b6922",
}
def api(repo,kind="models"):
    cache=ROOT/"environment"/("manifest-"+repo.replace("/","--")+".json")
    if cache.exists():
        cached=json.loads(cache.read_text())
        if cached.get("sha")==PINNED_REVISIONS[repo]:return cached
    url="https://hf-mirror.com/api/"+kind+"/"+repo+"/revision/"+PINNED_REVISIONS[repo]+"?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=45) as r:meta=json.load(r)
    if meta["sha"]!=PINNED_REVISIONS[repo]:raise RuntimeError("Unexpected repository revision: "+repo)
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
    urls=[url]
    if kind=="models" and expected:
        ms_repo="Tencent-Hunyuan/Hunyuan3D-2.1" if repo=="tencent/Hunyuan3D-2.1" else repo
        mf=ROOT/"environment"/("modelscope_files_"+ms_repo.replace("/","--")+".json")
        if mf.exists():
            ms=json.loads(mf.read_text())
            matched=next((x for x in ms.get("Data",{}).get("Files",[]) if x.get("Path")==name and x.get("Sha256")==expected),None)
            if matched:
                alternative="https://modelscope.cn/api/v1/models/"+ms_repo+"/repo?Revision="+matched["Revision"]+"&FilePath="+urllib.parse.quote(name)
                urls.insert(0,alternative)
                data["alternative_source"]={"repo":ms_repo,"revision":matched["Revision"],"sha256":expected}
    save()
    cmd=["aria2c","--async-dns=false","--continue=true","--allow-overwrite=true","--auto-file-renaming=false","--file-allocation=none","--max-connection-per-server=8","--split=8","--min-split-size=16M","--max-tries=3","--retry-wait=5","--connect-timeout=20","--timeout=60","--summary-interval=30","--console-log-level=warn","--dir",str(partial.parent),"--out",partial.name,*urls]
    transfer_env=__import__("os").environ.copy()
    if network_proxy() and kind=="datasets" and __import__("os").environ.get("EXPERIMENT_DATA_RELAY")=="1":cmd.insert(1,"--all-proxy="+network_proxy())
    else:
        for key in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"]:transfer_env.pop(key,None)
    with log.open("a") as f:
        attempts=[]
        # Large cross-border transfers sometimes lose throughput on long-lived connections.
        # Periodic graceful restarts preserve aria2's completed ranges and renew connections.
        for attempt in range(360 if kind=="datasets" and size and size>1024**3 else 10):
            started=time.monotonic()
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=transfer_env)
            try:
                p.wait(timeout=60 if kind=="datasets" and size and size>1024**3 else None)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                attempts.append({"attempt":attempt+1,"elapsed_seconds":time.monotonic()-started,"restart":"renew_connections","exit_code":p.returncode})
                data["transfer_attempts"]=attempts;save()
                f.write("\nCONNECTION_REFRESH "+str(attempt+1)+"\n");f.flush()
                continue
            attempts.append({"attempt":attempt+1,"elapsed_seconds":time.monotonic()-started,"exit_code":p.returncode})
            data["transfer_attempts"]=attempts;save()
            if p.returncode and attempt<10:
                time.sleep(2);continue
            break
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
