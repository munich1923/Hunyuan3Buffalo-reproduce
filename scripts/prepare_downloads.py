"""Download data archives concurrently with model preparation; per-file locks prevent duplicate writers."""
import concurrent.futures,json
from remote_runtime import ROOT,configure
from download_artifacts import api,download
configure()
meta=api("yejunliang23/Nano3D-Edit-100k","datasets")
wanted={"README.md","v1_100k/info/v1_100k_info.jsonl"}
wanted.update("v1_100k/editing_assets/part_%03d.tar.gz"%i for i in [0,33,66])
items=[x for x in meta["siblings"] if x["rfilename"] in wanted]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    result=list(ex.map(lambda item:download("yejunliang23/Nano3D-Edit-100k","datasets",meta["sha"],item,ROOT/"assets/Nano3D-Edit-100k"),items))
(ROOT/"environment/data_download_status.json").write_text(json.dumps({"ok":all(result),"verified":sum(result),"total":len(result)},indent=2)+"\n")
raise SystemExit(0 if all(result) else 1)
