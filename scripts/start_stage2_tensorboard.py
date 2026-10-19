"""Start or reuse this project's loopback-only TensorBoard; requires no GPU."""
from stage2_runtime import *
import socket


def main():
    p=parser(__doc__)
    p.add_argument("--port",type=int,default=6006)
    args=p.parse_args();cfg=config(args.config)
    state=path("environment/stage2_tensorboard_process.json")
    with lock("environment/stage2_tensorboard.lock"):
        if state.exists():
            old=read_json(state)
            try:
                command=Path(f"/proc/{old['pid']}/cmdline").read_bytes().decode().replace("\0"," ")
                if "tensorboard.main" in command and str(path(cfg["output_root"])) in command:
                    print(canonical(old));return
            except FileNotFoundError: pass
        port=args.port
        while port<args.port+100:
            with socket.socket() as sock:
                try: sock.bind(("127.0.0.1",port));break
                except OSError: port+=1
        else: raise RuntimeError("No loopback port available")
        folder=path(cfg["output_root"]);folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-B","-m","tensorboard.main","--logdir",str(folder),
                 "--host","127.0.0.1","--port",str(port)]
        with (folder/"tensorboard_server.log").open("a") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
               stdin=subprocess.DEVNULL,start_new_session=True,
               env={**os.environ,"CUDA_VISIBLE_DEVICES":""})
        for _ in range(20):
            if process.poll() is not None: raise RuntimeError("TensorBoard exited; inspect log")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=.5): break
            except OSError: time.sleep(.5)
        else: raise RuntimeError("TensorBoard startup timeout; process retained for diagnosis")
        info={"pid":process.pid,"port":port,"host":"127.0.0.1","command":command,**clock()}
        atomic_json(state,info);print(canonical(info))

if __name__=="__main__": main()
