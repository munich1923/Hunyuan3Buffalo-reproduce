"""Resumable remote orchestration, one GPU and cumulative 24h allocation budget."""
import argparse,datetime,fcntl,json,os,pathlib,subprocess,sys,time,traceback
from remote_runtime import ROOT,configure
configure()
STATE=ROOT/"environment/run_state.json"
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def readstate():
    return json.loads(STATE.read_text()) if STATE.exists() else {"started_utc":utc(),"stages":[],"gpu_seconds":0}
def atomic(data,path=STATE):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(data,indent=2)+"\n");temp.replace(path)
def wait_preparation(state,names=("install","download")):
    start=time.monotonic()
    while True:
        status=[]
        for name in names:
            p=ROOT/"environment"/(name+"_status.json")
            data=json.loads(p.read_text()) if p.exists() else {}
            ready=bool(data.get("completed_utc")) if name=="install" else data.get("ok",False)
            if ready:continue
            pidfile=ROOT/"environment"/(name+"_pid.txt")
            pid=int(pidfile.read_text()) if pidfile.exists() else -1
            proc=pathlib.Path("/proc")/str(pid)
            alive=proc.exists() and ") Z" not in (proc/"stat").read_text()
            if not alive:raise RuntimeError(name+" preparation did not complete; see its durable logs/status")
            status.append(name)
        if not status:return
        state.update(status="waiting_preparation",waiting_for=status,updated_utc=utc());atomic(state)
        if time.monotonic()-start>6*3600:raise RuntimeError("Preparation exceeded 6h wait; downloads may be resumed")
        time.sleep(15)
def acquire_gpu():
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    for line in output.splitlines():
        index,used=map(int,line.split(","))
        if used>100:continue
        f=(ROOT/"environment"/("gpu_"+str(index)+".lock")).open("a")
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        return index,f
    raise RuntimeError("No unused GPU detected; no existing jobs interrupted")
def run(state,name,python,script,args=(),gpu=False,cap=None):
    if any(x["name"]==name and x["exit_code"]==0 for x in state["stages"]):return
    cmd=[str(python),str(ROOT/"scripts"/script),*map(str,args)]
    env=os.environ.copy();lock=None
    if gpu:
        remaining=86400-state["gpu_seconds"]
        if remaining<=0:raise RuntimeError("24h GPU budget exhausted")
        index,lock=acquire_gpu();env["CUDA_VISIBLE_DEVICES"]=str(index);env["EGL_DEVICE_ID"]=str(index)
        allowance=min(remaining,cap if cap else remaining)
    else:index=None;allowance=None
    log=ROOT/"logs"/("stage_"+name+".log")
    record={"experiment_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"started_shanghai":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"name":name,"command":cmd,"started_utc":utc(),"gpu":index,"log":str(log),"source_commit":"82920d643c0dc2f7bfd7255f45f62d386edfe60c"}
    state.update(status="running",current=record,updated_utc=utc());atomic(state)
    print("START",name,"GPU",index,flush=True);start=time.monotonic();timed_out=False
    with log.open("a") as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record["pid"]=p.pid;atomic(state)
        try:p.wait(timeout=allowance)
        except subprocess.TimeoutExpired:
            import signal
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    elapsed=time.monotonic()-start
    if gpu:
        state["gpu_seconds"]+=elapsed
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    record.update(exit_code=p.returncode,ended_utc=utc(),elapsed_seconds=elapsed,timed_out=timed_out)
    state["stages"].append(record);state.pop("current",None);atomic(state)
    print("END",name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(name+" failed; "+str(log))
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");a=parser.parse_args()
    state=readstate();state.pop("error",None);lock=(ROOT/"environment/pilot.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shape=ROOT/"environment/venvs/shape/bin/python";vlm=ROOT/"environment/venvs/vlm/bin/python"
    try:
        wait_preparation(state,("install",))
        run(state,"tensor_contract",shape,"test_edit_model.py",gpu=True,cap=300)
        run(state,"imports",shape,"verify_environment.py")
        run(state,"shape_interface_smoke",shape,"smoke_shape.py",gpu=True,cap=900)
        run(state,"vlm_interface_smoke",vlm,"smoke_vlm.py",gpu=True,cap=900)
        run(state,"complete_interface_smoke",shape,"smoke_forward.py",gpu=True,cap=900)
        wait_preparation(state,("download",))
        run(state,"data_contracts",shape,"test_data_contracts.py")
        run(state,"data",shape,"prepare_dataset.py")
        run(state,"render_train_diagnostic",shape,"render_views.py",["--split","train","--limit","60"],gpu=True,cap=1800)
        run(state,"render_val_diagnostic",shape,"render_views.py",["--split","val","--limit","16"],gpu=True,cap=900)
        run(state,"render_test",shape,"render_views.py",["--split","test"],gpu=True,cap=1800)
        if a.prepare_only:state.update(status="prepared_only",updated_utc=utc());atomic(state);return
        review=ROOT/"environment/data_review.json"
        if not review.exists() or not json.loads(review.read_text()).get("approved_for_pilot"):
            raise RuntimeError("Visual review pending: inspect data_review images and record actual observations first")
        # VAE gate first, before spending time extracting all features.
        run(state,"vae_gate",shape,"cache_shape.py",["--split","train","--limit","1"],gpu=True,cap=900)
        gate=json.loads((ROOT/"environment/vae_gate.json").read_text())
        if not gate["passed"]:raise RuntimeError("VAE reconstruction gate failed")
        # First 16 entries contain 8 independent add pairs and their reverses.
        run(state,"shape_cache_diagnostic",shape,"cache_shape.py",["--split","train","--limit","16"],gpu=True,cap=3600)
        run(state,"vlm_cache_diagnostic",vlm,"cache_vlm.py",["--split","train","--limit","16"],gpu=True,cap=3600)
        run(state,"shape_cache_val_diagnostic",shape,"cache_shape.py",["--split","val","--limit","16"],gpu=True,cap=3600)
        run(state,"vlm_cache_val_diagnostic",vlm,"cache_vlm.py",["--split","val","--limit","16"],gpu=True,cap=3600)
        run(state,"overfit",shape,"train_editor.py",["--stage","overfit","--seconds","6800"],gpu=True,cap=7200)
        if not json.loads((ROOT/"environment/training_overfit.json").read_text())["passed"]:raise RuntimeError("Overfit gate failed")
        run(state,"render_train_full",shape,"render_views.py",["--split","train"],gpu=True,cap=7200)
        run(state,"render_val_full",shape,"render_views.py",["--split","val"],gpu=True,cap=1800)
        run(state,"shape_cache",shape,"cache_shape.py",gpu=True,cap=14400)
        run(state,"vlm_cache",vlm,"cache_vlm.py",gpu=True,cap=21600)
        run(state,"connector",shape,"train_editor.py",["--stage","connector","--seconds","6800"],gpu=True,cap=7200)
        run(state,"lora",shape,"train_editor.py",["--stage","lora","--seconds","21000"],gpu=True,cap=21600)
        for mode in ["source","vae","native","random","connector","lora","no_source","noop","shuffled"]:
            run(state,"eval_"+mode,shape,"evaluate_editor.py",["--mode",mode,"--seconds","3500"],gpu=True,cap=3600)
        state.update(status="completed",updated_utc=utc());atomic(state)
    except Exception as e:
        state.update(status="stopped",error=repr(e),updated_utc=utc());atomic(state)
        traceback.print_exc();raise
if __name__=="__main__":main()
