"""Prefetch pinned compatible Python wheels without involving other experiments."""
import concurrent.futures,hashlib,json,re,subprocess,urllib.request
from packaging.tags import cpython_tags,compatible_tags
from packaging.utils import parse_wheel_filename
from remote_runtime import ROOT,configure,network_proxy
configure()
platforms=["manylinux_2_17_x86_64","manylinux2014_x86_64","manylinux_2_28_x86_64","manylinux_2_27_x86_64","manylinux_2_24_x86_64","manylinux_2_12_x86_64","manylinux2010_x86_64","manylinux_2_5_x86_64","manylinux1_x86_64","linux_x86_64"]
valid=set(cpython_tags((3,10),abis=["cp310","abi3","none"],platforms=platforms)) | set(compatible_tags((3,10),interpreter="cp310",platforms=platforms))
folder=ROOT/".cache/wheels"
reqs=set()
for p in (ROOT/"environment").glob("requirements-*.txt"):
 for line in p.read_text().splitlines():
  m=re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ #]+)",line.strip())
  if m:reqs.add(m.groups())
def fetch(pair):
 pkg,ver=pair
 with urllib.request.urlopen("https://pypi.org/pypi/"+pkg+"/"+ver+"/json",timeout=40) as f:data=json.load(f)
 opts=[]
 for x in data["urls"]:
  if x["filename"].endswith(".whl") and parse_wheel_filename(x["filename"])[3] & valid:opts.append(x)
 if not opts:return {"package":pkg,"version":ver,"status":"no_compatible_wheel"}
 x=opts[0];p=folder/x["filename"];expected=x["digests"]["sha256"]
 log=ROOT/"logs/install"/("wheel_"+pkg+".log")
 if not p.exists():
  with log.open("a") as f:
   cmd=["aria2c","--continue=true","--auto-file-renaming=false","--file-allocation=none","--split=4","--max-connection-per-server=4","--min-split-size=4M","--max-tries=3","--timeout=60","--console-log-level=warn","--summary-interval=30","--dir",str(folder),"--out",p.name,x["url"]]
   if network_proxy():cmd.insert(1,"--all-proxy="+network_proxy())
   subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=expected:raise RuntimeError("wheel hash mismatch "+p.name)
 print("VERIFIED",p.name,flush=True)
 return {"package":pkg,"version":ver,"path":str(p),"sha256":h,"status":"verified"}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,sorted(reqs)))
(ROOT/"environment/requirements_wheel_receipts.json").write_text(json.dumps(rows,indent=2)+"\n")
