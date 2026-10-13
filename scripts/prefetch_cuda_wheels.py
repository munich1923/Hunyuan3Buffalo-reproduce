"""Fetch pinned CUDA dependencies from PyPI with segmented transfers and hash verification."""
import concurrent.futures,email,hashlib,json,pathlib,re,subprocess,urllib.request,urllib.parse,zipfile,html
from remote_runtime import ROOT,configure
configure()
folder=ROOT/".cache/wheels";folder.mkdir(parents=True,exist_ok=True)
torchwheel=folder/"torch-2.5.1+cu124-cp310-cp310-linux_x86_64.whl"
with zipfile.ZipFile(torchwheel) as z:
    name=next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    meta=email.message_from_bytes(z.read(name))
wanted=[]
for req in meta.get_all("Requires-Dist",[]):
    m=re.match(r"(nvidia-[a-z0-9-]+|triton)\s*\(?==([0-9.]+)",req)
    if m:wanted.append(m.groups())
def get(pair):
    package,version=pair
    with urllib.request.urlopen("https://pypi.org/pypi/"+package+"/"+version+"/json",timeout=40) as r:data=json.load(r)
    options=[x for x in data["urls"] if x["filename"].endswith(".whl") and "x86_64" in x["filename"] and ("cp310" in x["filename"] if package=="triton" else "manylinux" in x["filename"])]
    if not options:raise RuntimeError("No Linux cp310 wheel for "+package)
    item=options[0];path=folder/item["filename"]
    urls=[item["url"]]
    try:
        index="https://download.pytorch.org/whl/cu124/"+package+"/"
        page=urllib.request.urlopen(index,timeout=20).read().decode()
        for href in re.findall('href="([^"]+)"',page):
            href=html.unescape(href)
            if item["filename"] in href and "sha256="+item["digests"]["sha256"] in href:
                urls.insert(0,urllib.parse.urljoin(index,href).split("#")[0]);break
    except Exception:pass
    if not path.exists() or path.with_name(path.name+".aria2").exists() or not zipfile.is_zipfile(path):
        log=ROOT/"logs/install"/("prefetch_"+package+".log");log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a") as f:
            subprocess.run(["aria2c","--continue=true","--auto-file-renaming=false","--file-allocation=none","--max-connection-per-server=8","--split=8","--min-split-size=8M","--max-tries=3","--timeout=60","--summary-interval=30","--console-log-level=warn","--dir",str(folder),"--out",path.name,*urls],stdout=f,stderr=subprocess.STDOUT,check=True)
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    if h.hexdigest()!=item["digests"]["sha256"]:raise RuntimeError("Wheel hash mismatch: "+path.name)
    print("Verified",path.name,flush=True)
    return {"package":package,"version":version,"file":str(path),"sha256":h.hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(get,wanted))
(ROOT/"environment/cuda_wheel_receipts.json").write_text(json.dumps(results,indent=2)+"\n")
