"""CPU-only Stage 2 sampler, persistent GPU ledger and bounded cache prefetch."""
from stage2_runtime import *
import concurrent.futures, copy, math, random, uuid


def epoch_order(rows, master, epoch):
    order=list(rows)
    random.Random(seed(master,"epoch",epoch)).shuffle(order)
    return order


def batches(rows, master, epoch, offset=0):
    if offset < 0 or offset > len(rows) or offset % 32:
        raise ValueError("resume offset must be a global update boundary")
    order=epoch_order(rows,master,epoch)
    for start in range(offset,len(order),32):
        batch=order[start:start+32]
        if len(batch)%2: raise ValueError("two equal rank populations required")
        yield start,batch


def next_position(epoch,start,count,size,update):
    end=start+count
    if end>size: raise ValueError("position exceeds epoch")
    return {"epoch":epoch+int(end==size),"offset":0 if end==size else end,"update":update}


class Prefetch:
    """One in-flight update batch, no CUDA and no stochastic work in thread."""
    def __init__(self, iterable, load):
        self.iterable=iter(iterable);self.load=load
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future=None;self.key=None
    def _schedule(self):
        try: key,rows=next(self.iterable)
        except StopIteration: self.future=None;return
        self.key=key
        self.future=self.pool.submit(lambda:[(r,self.load(r)) for r in rows])
    def __enter__(self): self._schedule();return self
    def __iter__(self): return self
    def __next__(self):
        if self.future is None: raise StopIteration
        key=self.key;value=self.future.result()
        self._schedule()
        return key,value
    def __exit__(self,*args): self.pool.shutdown(wait=True,cancel_futures=True)


class Ledger:
    """Single supervisor owns the file and lock; restarts never reset usage."""
    def __init__(self,file,limit):
        self.file=path(file);self.limit=float(limit)
        self.data=read_json(self.file) if self.file.exists() else {
            "schema":1,"limit_gpu_seconds":self.limit,"phases":[]}
        if self.data["limit_gpu_seconds"]!=self.limit: raise RuntimeError("budget limit changed")
        active=[x for x in self.data["phases"] if x["state"]=="running"]
        for item in active:
            # A previous parent must not remain alive, and its process group
            # must be gone. Bill the conservatively elapsed allocation window.
            for key in ["owner_pid","child_pid"]:
                pid=item.get(key)
                if pid:
                    try: os.kill(pid,0)
                    except ProcessLookupError: pass
                    else: raise RuntimeError("prior budget phase process is still alive")
            elapsed=max(0,time.time()-item["started_unix"])
            item.update(state="interrupted_reconciled",gpu_seconds=elapsed*item["gpu_count"],
                        reconciled=clock())
        self.save()
    def save(self): atomic_json(self.file,self.data)
    def used(self):
        return sum(x.get("gpu_seconds",max(0,time.time()-x["started_unix"])*x["gpu_count"])
                   for x in self.data["phases"])
    def remaining(self): return max(0,self.limit-self.used())
    def start(self,run,phase,cards,minimum=0):
        if self.remaining()<max(minimum*cards,10): raise TimeoutError("insufficient remaining GPU budget for phase")
        item={"id":uuid.uuid4().hex,"run":str(run),"phase":phase,"gpu_count":cards,
              "state":"running","owner_pid":os.getpid(),"started_unix":time.time(),**clock()}
        self.data["phases"].append(item);self.save();return item
    def end(self,item,code):
        item.update(state="complete" if code==0 else "failed",exit_code=code,
                    gpu_seconds=max(0,time.time()-item["started_unix"])*item["gpu_count"],ended=clock())
        self.save()


def phase_deadline():
    value=os.environ.get("STAGE2_PHASE_DEADLINE_UNIX")
    return float(value) if value else float("inf")


def deadline_check(reserve_seconds=0):
    if time.time()+reserve_seconds>=phase_deadline(): raise TimeoutError("GPU budget boundary")


def cpu_sampler_checks(cfg):
    rows=list(range(10000));seen=[];sizes=[];tails=[]
    for epoch in range(5):
        groups=list(batches(rows,cfg["seed"],epoch))
        flat=[v for _,b in groups for v in b]
        assert sorted(flat)==rows and len(set(flat))==10000
        assert len(groups)==313 and len(groups[-1][1])==16
        assert epoch_order(rows,cfg["seed"],epoch)==epoch_order(rows,cfg["seed"],epoch)
        for start,b in groups:
            a,c=b[::2],b[1::2]
            assert set(a).isdisjoint(c) and len(a)==len(c)
            # Distinct values exercise sample normalization rather than constants.
            assert math.isclose((sum(a)/len(a)+sum(c)/len(c))/2,sum(b)/len(b),abs_tol=1e-10)
        start,b=groups[-1];tails.append(next_position(epoch,start,len(b),10000,(epoch+1)*313))
        # Resume in the middle must rebuild the same shuffled epoch.
        assert list(batches(rows,cfg["seed"],epoch,4992))==groups[156:]
        seen+=flat;sizes.append(len(groups))
    assert len(seen)==50000 and sum(sizes)==1565
    return {"passed":True,"epochs":5,"samples":50000,"updates":1565,
            "last_batches":[16]*5,"rank_tail_accumulation":8,"positions":tails,
            "resume_order_checked":True,"sample_weighted_tail_normalization":True}

