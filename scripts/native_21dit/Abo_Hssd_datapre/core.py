"""Metadata, identity, split, and receipt helpers for ABO/HSSD preparation."""
import ast,contextlib,csv,datetime,fcntl,hashlib,json,math,os,random,shutil,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
CFG=ROOT/'configs/abo_hssd_v1.json'

def path(p):
    p=Path(p);p=(ROOT/p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('outside userdata workspace: '+str(p))
    return p

def setup():
    if not str(ROOT).startswith('/root/userdata/'):raise RuntimeError('remote userdata only')
    sys.dont_write_bytecode=True
    for key,part in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','HF_HOME':'hf','HF_HUB_CACHE':'hf/hub','HUGGINGFACE_HUB_CACHE':'hf/hub','TRANSFORMERS_CACHE':'hf/transformers','TORCH_HOME':'torch','TORCH_EXTENSIONS_DIR':'torch_extensions','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg','MPLCONFIGDIR':'matplotlib','PIP_CACHE_DIR':'pip'}.items():
        p=path('environment/cache/abo_hssd/'+part);p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def now():
    d=datetime.datetime.now(datetime.timezone.utc)
    return {'utc':d.isoformat(),'asia_shanghai':d.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def fp(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(path(p).read_text())
def atomic(p,data):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',encoding='utf8') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def immutable(p,data):
    if path(p).exists():
        if read(p)!=data:raise RuntimeError('immutable content changed: '+str(p))
    else:atomic(p,data)
def lines(p,rows):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with t.open('w',encoding='utf8') as f:
        for row in rows:f.write(canonical(row)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def readlines(p):return [json.loads(s) for s in path(p).read_text().splitlines() if s.strip()]
def append(p,row):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:f.write(canonical({**row,**now()})+'\n');f.flush()
def seed(*items):return int(fp(items)[:16],16)%(2**32)
@contextlib.contextmanager
def lock(p):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def cfg():return read(CFG)
def dirs(c):return path(c['assets']),path(c['results'])
def check_upstream(c):
    folder=path('src/Hunyuan3D-2.1')
    if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=c['upstream_commit']:raise RuntimeError('upstream commit mismatch')
    if subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip():raise RuntimeError('upstream dirty')
def code_contract():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
def parse_captions(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('missing captions')
    try:v=json.loads(text)
    except json.JSONDecodeError:v=ast.literal_eval(text)
    if not isinstance(v,list) or any(not isinstance(x,str) for x in v):raise ValueError('malformed caption list')
    valid=[i for i,x in enumerate(v) if x.strip()]
    if not valid:raise ValueError('no nonempty caption')
    return v,valid

def choose_caption(row,master,epoch,global_position,evaluation=False):
    indices=row['nonempty_caption_indices']
    index=indices[0] if evaluation else random.Random(seed(master,'caption',epoch,global_position,row['uid'])).choice(indices)
    return {'caption_index':index,'caption':row['captions'][index],'caption_sha256':hashlib.sha256(row['captions'][index].encode()).hexdigest()}

def allocate(n,ratios):
    raw=[n*x for x in ratios];out=[math.floor(x) for x in raw]
    for i in sorted(range(len(raw)),key=lambda j:(-(raw[j]-out[j]),j))[:n-sum(out)]:out[i]+=1
    return out

def dedup(rows):
    ordered=sorted(rows,key=lambda r:(r['source'],r['uid']));parents=list(range(len(ordered)))
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    keys={}
    for i,r in enumerate(ordered):
        for key in [('file',r['raw_sha256']),('geo',r['geometry_hash'])]:
            if key in keys:parents[find(i)]=find(keys[key])
            else:keys[key]=i
    groups={}
    for i,r in enumerate(ordered):groups.setdefault(find(i),[]).append(r)
    result=[];audit=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:(r['source'],r['uid']));rep=dict(group[0]);aliases=[];cap=[];origins=[]
        for row in group:
            aliases.append({'source':row['source'],'uid':row['uid'],'raw_sha256':row['raw_sha256']})
            for i,text in enumerate(row['captions']):cap.append(text);origins.append({'uid':row['uid'],'caption_index':i})
        rep['captions']=cap;rep['caption_origins']=origins;rep['nonempty_caption_indices']=[i for i,t in enumerate(cap) if t.strip()];rep['caption_hash']=fp(cap);rep['aliases']=aliases
        result.append(rep)
        if len(group)>1:audit.append({'representative':rep['uid'],'aliases':aliases})
    return result,audit

def split_rows(rows,c):
    names=['train','val','test'];counts=allocate(len(rows),c['split_ratios']);result={x:[] for x in names};remaining=counts[:]
    sources=sorted({r['source'] for r in rows})
    for source in sources[:-1]:
        pool=[r for r in rows if r['source']==source];random.Random(seed(c['seed'],'split',source)).shuffle(pool)
        ideal=[len(pool)*x for x in c['split_ratios']];nums=[min(math.floor(v),remaining[i]) for i,v in enumerate(ideal)]
        while sum(nums)<len(pool):
            options=[i for i in range(3) if nums[i]<remaining[i]]
            i=max(options,key=lambda j:(ideal[j]-nums[j],-j));nums[i]+=1
        offset=0
        for i,name in enumerate(names):result[name]+=pool[offset:offset+nums[i]];offset+=nums[i];remaining[i]-=nums[i]
    if sources:
        pool=[r for r in rows if r['source']==sources[-1]];random.Random(seed(c['seed'],'split',sources[-1])).shuffle(pool);offset=0
        for i,name in enumerate(names):result[name]+=pool[offset:offset+remaining[i]];offset+=remaining[i]
        assert offset==len(pool)
    for name in names:
        result[name]=[dict(r,split=name) for r in sorted(result[name],key=lambda x:x['uid'])]
    assert [len(result[n]) for n in names]==counts
    return result

class AccessBlocked(RuntimeError):pass

def download(url,dest,expected_sha=None,attempts=5):
    dest=path(dest);dest.parent.mkdir(parents=True,exist_ok=True);receipt=dest.with_name(dest.name+'.download.json')
    if dest.exists():
        current=sha(dest)
        if expected_sha and current!=expected_sha:raise RuntimeError('existing payload SHA mismatch: '+str(dest))
        if receipt.exists() and read(receipt)['sha256']==current:return read(receipt)
        if not expected_sha:raise RuntimeError('existing download has no receipt: '+str(dest))
        out={'sha256':current,'bytes':dest.stat().st_size,'url':url,**now()};atomic(receipt,out);return out
    part=dest.with_name(dest.name+'.part');error=None
    for attempt in range(attempts):
        offset=part.stat().st_size if part.exists() else 0
        try:
            headers={'User-Agent':'ABO-HSSD-research/1.0'}
            if offset:headers['Range']=f'bytes={offset}-'
            req=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req,timeout=40) as response:
                content_range=response.headers.get('Content-Range');length=response.headers.get('Content-Length')
                if offset and (response.status!=206 or not content_range or not content_range.startswith(f'bytes {offset}-')):
                    raise RuntimeError('server rejected exact resume offset')
                if response.status==206:
                    if not content_range or not content_range.startswith(f'bytes {offset}-'):raise RuntimeError('invalid Content-Range')
                    total=int(content_range.rsplit('/',1)[1])
                else:total=int(length) if length else None
                with part.open('ab' if offset else 'wb') as f:
                    while True:
                        b=response.read(1024**2)
                        if not b:break
                        f.write(b)
                    f.flush();os.fsync(f.fileno())
                if total is not None and part.stat().st_size!=total:raise RuntimeError('incomplete download')
            current=sha(part)
            if expected_sha and current!=expected_sha:raise RuntimeError('download SHA mismatch')
            os.rename(part,dest);out={'url':url,'bytes':dest.stat().st_size,'sha256':current,**now()};atomic(receipt,out);return out
        except urllib.error.HTTPError as e:
            if e.code in [401,403]:raise AccessBlocked(f'HTTP {e.code} for {url.split("?")[0]}') from e
            error=repr(e)
        except Exception as e:error=repr(e)
        if attempt<attempts-1:time.sleep(min(30,2**attempt))
    raise RuntimeError('download incomplete; partial retained: '+str(error))

def metadata(c):
    a,o=dirs(c);folder=a/'metadata';folder.mkdir(parents=True,exist_ok=True);accepted=[];rejected=[];stats={};receipts={}
    for source in c['sources']:
        f=folder/(source+'.csv');url=f'https://hf-mirror.com/datasets/JeffreyXiang/TRELLIS-500K/resolve/{c["metadata_revision"]}/{source}.csv'
        receipts[source]=download(url,f);rows=list(csv.DictReader(f.open(newline='',encoding='utf8')));good=[]
        for line,row in enumerate(rows,2):
            try:
                caps,indices=parse_captions(row['captions']);h=row['sha256'];ident=row['file_identifier']
                if len(h)!=64 or any(ch not in '0123456789abcdef' for ch in h):raise ValueError('invalid SHA256')
                if Path(ident).is_absolute() or '..' in Path(ident).parts:raise ValueError('unsafe identifier')
                good.append({'uid':source.lower()+'_'+h,'source':source,'asset_id':ident,'raw_sha256':h,'captions':caps,'nonempty_caption_indices':indices,'caption_hash':fp(caps),'metadata_revision':c['metadata_revision'],'caption_source':'TRELLIS-500K','aesthetic_score_not_filtered':row.get('aesthetic_score')})
            except Exception as exc:rejected.append({'source':source,'line':line,'asset_id':row.get('file_identifier'),'reason':repr(exc)})
        stats[source]={'all':len(rows),'with_caption':len(good),'caption_entries':sum(len(r['nonempty_caption_indices']) for r in good)};accepted+=good
    lines(folder/'candidates.jsonl',accepted);lines(folder/'rejected_metadata.jsonl',rejected)
    report={'metadata_revision':c['metadata_revision'],'trellis_commit':c['trellis_commit'],'sources':stats,'receipts':receipts,'candidates_sha256':sha(folder/'candidates.jsonl'),'data_ready':False,**now()};atomic(o/'metadata_summary.json',report)
    return accepted

def geometry_hash(vertices,faces):
    import numpy as np
    v=np.asarray(vertices,dtype=np.float64);v=(v-(v.min(0)+v.max(0))/2)/np.ptp(v,axis=0).max();v=np.round(v,6)
    order=np.lexsort((v[:,2],v[:,1],v[:,0]));inverse=np.empty(len(v),int);inverse[order]=np.arange(len(v));tri=np.sort(inverse[np.asarray(faces)],axis=1);tri=tri[np.lexsort((tri[:,2],tri[:,1],tri[:,0]))]
    h=hashlib.sha256();h.update(v[order].tobytes());h.update(tri.astype('<i8').tobytes());return h.hexdigest()

def receipt_valid(folder,contract):
    folder=path(folder);f=folder/'complete.json'
    if not f.exists():return False
    data=read(f)
    if data['contract']!=contract:raise RuntimeError('completed stage contract changed: '+str(folder))
    for name,record in data['files'].items():
        p=folder/name
        if not p.is_file() or p.stat().st_size!=record['bytes']:raise RuntimeError('completed payload missing/size changed: '+str(p))
        stat=p.stat()
        if stat.st_mtime_ns!=record['mtime_ns'] and sha(p)!=record['sha256']:raise RuntimeError('completed payload checksum changed: '+str(p))
    return True

def commit_stage(temporary,final,contract,extra=None):
    temporary=path(temporary);final=path(final)
    if final.exists():raise FileExistsError(final)
    files={str(p.relative_to(temporary)):{'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(temporary.rglob('*')) if p.is_file() and p.name!='complete.json'}
    atomic(temporary/'complete.json',{'contract':contract,'files':files,**(extra or {}),**now()});os.rename(temporary,final)
    return read(final/'complete.json')
