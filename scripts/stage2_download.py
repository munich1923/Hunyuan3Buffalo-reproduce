"""Verified Range chunks; CPU only, persistent progress, no dead-end fallback."""
from stage2_runtime import *
import concurrent.futures
import re
import urllib.request
import threading

CHUNK = 4 * 1024**2


def append_range(piece, payload, headers, start, end, total):
    matches=re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",headers)
    if not matches:
        raise RuntimeError("Missing Content-Range; refusing unverified response")
    a,b,n=map(int,matches[-1])
    length=payload.stat().st_size
    if (a,b,n)!=(start,end,total) or not 0<length<=end-start+1:
        raise RuntimeError("Range mismatch; refusing response")
    with piece.open("ab") as out, payload.open("rb") as src:
        while True:
            block=src.read(1024**2)
            if not block:break
            out.write(block)
        out.flush();os.fsync(out.fileno())
    return length


def chunked_download(cfg,name,partial,size,receipts):
    directory=partial.parent/(partial.name+".chunks")
    directory.mkdir(parents=True,exist_ok=True)
    total=(size+CHUNK-1)//CHUNK
    progress=receipts/(partial.name.replace('.partial','')+'.progress.json')
    counter_lock=threading.Lock()
    stop=threading.Event()
    signed={"url":None,"time":0}
    url_lock=threading.Lock()
    # Resolve once per signed-URL lifetime, avoiding hundreds of mirror resolve calls.
    def url():
        with url_lock:
            if signed['url'] and time.monotonic()-signed['time']<1200:return signed['url']
            original=f"https://hf-mirror.com/datasets/{cfg['dataset']['repo']}/resolve/{cfg['dataset']['revision']}/{name}?download=true&stage2_chunk={time.time_ns()}"
            request=urllib.request.Request(original,headers={"Range":"bytes=0-0"})
            with urllib.request.urlopen(request,timeout=20) as response:
                if response.status!=206:raise RuntimeError('Range resolve failed')
                signed.update(url=response.geturl(),time=time.monotonic())
            return signed['url']
    def summary():
        completed=list(directory.glob('*.done'))
        byte_count=sum(x.stat().st_size for x in completed)+sum(x.stat().st_size for x in directory.glob('*.part'))
        record={"file":name,"completed_chunks":len(completed),"total_chunks":total,
                "downloaded_bytes":byte_count,"expected_bytes":size,**clock()}
        atomic_json(progress,record)
        append(receipts/'chunk_progress.jsonl',record)
    def worker(index):
        begin=index*CHUNK;end=min(begin+CHUNK,size)-1
        piece=directory/f'{index:06d}.part';done=directory/f'{index:06d}.done'
        if done.exists():
            if done.stat().st_size!=end-begin+1:raise RuntimeError('Corrupt chunk size')
            return
        attempt=0
        while not stop.is_set():
            offset=piece.stat().st_size if piece.exists() else 0
            if offset==end-begin+1:
                os.replace(piece,done)
                with counter_lock:summary()
                return
            if offset>end-begin+1:raise RuntimeError('Oversize chunk')
            payload=directory/f'{index:06d}.response';headers=directory/f'{index:06d}.headers'
            attempt+=1
            try:
                link=url()
                result=subprocess.run(['curl','-4','-sS','--connect-timeout','10','--max-time','40',
                    '--range',f'{begin+offset}-{end}','-D',str(headers),'-o',str(payload),link],
                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                if payload.exists() and payload.stat().st_size:
                    append_range(piece,payload,headers.read_text(errors='replace'),begin+offset,end,size)
                    with counter_lock:summary()
                    continue
                raise RuntimeError('No verified response bytes; curl exit '+str(result.returncode))
            except Exception as exc:
                with counter_lock:
                    append(receipts/'chunk_events.jsonl',{"event":"chunk_retry","file":name,
                           "chunk":index,"attempt":attempt,"reason":type(exc).__name__,"message":str(exc)[:150]})
                # Refresh signed URL after connectivity/HTTP failures, without logging it.
                with url_lock:signed['time']=0
                stop.wait(min(5*attempt,60))
            finally:
                for temp in [payload,headers]:
                    if temp.exists():temp.unlink()
    summary()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(worker,i) for i in range(total)]
            for f in concurrent.futures.as_completed(futures):f.result()
    except BaseException:
        stop.set();raise
    # Do not overwrite old partial/aria2 state: preserve it as recovery evidence.
    assembled=partial.with_name(partial.name+'.assembled')
    with assembled.open('wb') as out:
        for i in range(total):
            with (directory/f'{i:06d}.done').open('rb') as src:
                while True:
                    block=src.read(1024**2)
                    if not block:break
                    out.write(block)
        out.flush();os.fsync(out.fileno())
    if assembled.stat().st_size!=size:raise RuntimeError('Assembled size mismatch')
    if partial.exists():os.replace(partial,partial.with_name(partial.name+'.before_chunks'))
    os.replace(assembled,partial)
    print('CHUNKS_ASSEMBLED',name,size,flush=True)
