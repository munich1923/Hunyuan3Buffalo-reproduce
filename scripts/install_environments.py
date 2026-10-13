import datetime,json,os,subprocess,sys,zipfile
from remote_runtime import ROOT,configure
configure()
logdir=ROOT/"logs/install";logdir.mkdir(parents=True,exist_ok=True)
status={"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"steps":[]}
def run(name,cmd):
    start=datetime.datetime.now(datetime.timezone.utc)
    with (logdir/(name+".log")).open("a") as f:
        f.write("\nSTART "+start.isoformat()+" command="+repr(cmd)+"\n");f.flush()
        p=subprocess.run(cmd,cwd=ROOT,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
    record={"name":name,"command":cmd,"exit_code":p.returncode,"started_utc":start.isoformat(),"ended_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    status["steps"].append(record)
    (ROOT/"environment/install_status.json").write_text(json.dumps(status,indent=2)+"\n")
    print(name,p.returncode,flush=True)
    if p.returncode:sys.exit(p.returncode)
for name in ["shape","vlm"]:
    prefix=ROOT/"environment/venvs"/name
    if not (prefix/"bin/python").exists():
        run("conda_"+name,["/opt/conda/bin/conda","create","--yes","--prefix",str(prefix),"--override-channels","--channel","https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge","python=3.10","pip"])
    python=str(prefix/"bin/python")
    wheel_dir=ROOT/".cache/wheels";wheel_dir.mkdir(parents=True,exist_ok=True)
    wheels=[]
    for pkg,version in [("torch","2.5.1"),("torchvision","0.20.1"),("torchaudio","2.5.1")]:
        fn=pkg+"-"+version+"+cu124-cp310-cp310-linux_x86_64.whl"
        path=wheel_dir/fn
        if not path.exists() or path.with_name(path.name+".aria2").exists() or not zipfile.is_zipfile(path):
            url="https://download.pytorch.org/whl/cu124/"+fn.replace("+","%2B")
            run("wheel_"+pkg,["aria2c","--continue=true","--auto-file-renaming=false","--file-allocation=none","--split=8","--max-connection-per-server=8","--min-split-size=8M","--max-tries=3","--timeout=60","--console-log-level=warn","--summary-interval=30","--dir",str(wheel_dir),"--out",fn,url])
        wheels.append(str(path))
    run("prefetch_cuda_"+name,["/opt/conda/bin/python",str(ROOT/"scripts/prefetch_cuda_wheels.py")])
    run("torch_"+name,[python,"-m","pip","install","--timeout","60","--retries","2",*wheels,"--find-links",str(wheel_dir),"--index-url","https://mirrors.aliyun.com/pypi/simple/"])
    run("deps_"+name,[python,"-m","pip","install","--timeout","60","--retries","2","-r",str(ROOT/"environment"/("requirements-"+name+".txt")),"--index-url","https://mirrors.aliyun.com/pypi/simple/"])
    if name=="shape":
        run("torch_cluster_shape",[python,"-m","pip","install","torch-cluster==1.6.3","--no-index","--find-links","https://data.pyg.org/whl/torch-2.5.0+cu124.html"])
    run("pip_check_"+name,[python,"-m","pip","check"])
    with (ROOT/"environment"/("pip-freeze-"+name+".txt")).open("w") as f:
        subprocess.run([python,"-m","pip","freeze"],stdout=f,check=True)
    with (ROOT/"environment"/("conda-explicit-"+name+".txt")).open("w") as f:
        subprocess.run(["/opt/conda/bin/conda","list","--prefix",str(prefix),"--explicit"],stdout=f,check=True)
status["completed_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
(ROOT/"environment/install_status.json").write_text(json.dumps(status,indent=2)+"\n")
