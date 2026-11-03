"""Original TRELLIS images: CPU-only selection, audited streaming and split sidecars."""
import argparse, concurrent.futures, csv, datetime, fcntl, hashlib, io, json, math
import os, pathlib, random, struct, sys, time, traceback
P = pathlib.Path
ROOT = P('/root/userdata/zc/hunyuan3d-buffalo-1.0-editing')
ASSETS = ROOT/'assets/abo_hssd_v1'
AUDIT = ROOT/'results/abo_hssd_v1/render_images_v1'
SOURCE = P('/mnt/nas/xyx/TRELLIS.2/datasets')
INDEX = P('/root/userdata/zc/trellis_piflow/data/ssdit_conditions_abo_hssd_sketchfab_179454_20260929/manifests/all.jsonl')
POLICY = {'version':1,'seed':20261027,'namespace':'original-render-view-v1','elevation_degrees':[-60,60],'azimuth':'unrestricted','selection':'random.Random(derived_seed).choice(sorted_decodable_valid_frames)','coordinate_frame':'TRELLIS original normalized scene; Z-up; camera-to-world','not_watertight_render':True}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(v): return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def fingerprint(v): return hashlib.sha256(canonical(v).encode()).hexdigest()
def digest(data): return hashlib.sha256(data).hexdigest()
def sha(file):
 h=hashlib.sha256()
 with P(file).open('rb') as stream:
  for data in iter(lambda:stream.read(4*1024**2),b''):h.update(data)
 return h.hexdigest()
def read(file): return json.loads(P(file).read_text())
def rows(file): return [json.loads(l) for l in P(file).read_text().splitlines() if l.strip()]
def atomic(file,data):
 file=P(file);file.parent.mkdir(parents=True,exist_ok=True);temp=file.with_name(file.name+'.tmp.'+str(os.getpid()))
 with temp.open('w') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,file)
def immutable(file,data):
 if P(file).exists():
  if read(file)!=data:raise RuntimeError('frozen input differs: '+str(file))
 else:atomic(file,data)
def jsonlines(file,values):
 file=P(file);file.parent.mkdir(parents=True,exist_ok=True);temp=file.with_name(file.name+'.tmp.'+str(os.getpid()))
 with temp.open('w') as f:
  for v in values:f.write(canonical(v)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(temp,file)
def log(name,event):
 file=AUDIT/name;file.parent.mkdir(parents=True,exist_ok=True)
 with file.open('a') as f:f.write(canonical({**event,'utc':now()})+'\n');f.flush()
def angles(frame):
 m=frame['transform_matrix']
 if len(m)!=4 or any(len(r)!=4 for r in m):raise ValueError('matrix must be 4x4')
 values=[float(v) for r in m for v in r]
 if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite matrix')
 if any(abs(float(m[3][i])-v)>1e-6 for i,v in enumerate([0,0,0,1])):raise ValueError('nonaffine matrix')
 # Verify proper rotation, not merely a finite translation.
 for i in range(3):
  for j in range(3):
   if abs(sum(float(m[k][i])*float(m[k][j]) for k in range(3))-(1 if i==j else 0))>1e-4:raise ValueError('nonorthonormal camera rotation')
 r=[list(map(float,row[:3])) for row in m[:3]]
 det=r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
 if abs(det-1)>1e-4:raise ValueError('improper rotation')
 x,y,z=[float(m[i][3]) for i in range(3)]
 if math.sqrt(x*x+y*y+z*z)<1e-8:raise ValueError('zero camera radius')
 fov=float(frame['camera_angle_x'])
 if not math.isfinite(fov) or not 0<fov<math.pi:raise ValueError('invalid horizontal FOV')
 return math.degrees(math.atan2(z,math.hypot(x,y))),math.degrees(math.atan2(y,x))
def image_info(file):
 from PIL import Image
 file=P(file);before=file.stat()
 with Image.open(file) as im:
  if im.format!='PNG' or im.size!=(1024,1024) or im.mode!='RGBA':raise ValueError('expected original 1024x1024 RGBA PNG')
  im.load();info={'width':im.width,'height':im.height,'mode':im.mode}
 after=file.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('image changed while decoding')
 return {**info,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}
def chosen_index(uid,n):
 if n<1:raise ValueError('no eligible view')
 seed=int(fingerprint([POLICY['seed'],POLICY['namespace'],uid])[:16],16)%(2**32)
 return random.Random(seed).randrange(n)
def identity(row,meta):
 if row['uid']!=row['source'].lower()+'_'+row['raw_sha256'] or meta['sha256']!=row['raw_sha256'] or meta['file_identifier']!=row['asset_id']:raise ValueError('source/UID/SHA/asset path mismatch')
def prepare_candidates():
 inventory=ROOT/'results/abo_hssd_v1/migration_20261027/source_inventory.json'
 if sha(inventory)!='df419ae645debf29ab79b8ca6cb7f1c98b47cad1045207c967201f2e3c95fc94':raise RuntimeError('migration inventory changed')
 records=rows(ASSETS/'metadata/candidates.jsonl')
 lookup={}
 for r in rows(INDEX):
  key=(r['dataset'],r['object_id'])
  if key in lookup:raise RuntimeError('duplicate condition index identity')
  lookup[key]=r
 inv={r['uid']:r for r in read(inventory)['files']}
 result=[]
 for r in records:
  old=read(ROOT/'results/abo_hssd_v1/migration_20261027/receipts'/(r['uid']+'.json'));f=ASSETS/'raw'/r['source']/r['asset_id'];stat=f.stat()
  if old['target_sha256']!=r['raw_sha256'] or old['target_mtime_ns']!=stat.st_mtime_ns or old['bytes']!=stat.st_size or inv[r['uid']]['expected_sha256']!=r['raw_sha256']:raise RuntimeError('migration proof mismatch')
  index=lookup.get((r['source'],r['raw_sha256']))
  if index and (P(index['image_path']).parent.name!=r['raw_sha256'] or P(index['image_path']).parent.parent.name!='renders_cond'):raise RuntimeError('image index path mismatch')
  result.append({k:r[k] for k in ['uid','source','asset_id','raw_sha256']}|{'trellis_index':index})
 request={'policy':POLICY,'candidates_sha256':sha(ASSETS/'metadata/candidates.jsonl'),'trellis_index_sha256':sha(INDEX),'migration_inventory_sha256':sha(inventory),'objects':result}
 immutable(AUDIT/'candidate_request.json',request)
 print(json.dumps({'candidates':len(result),'indexed':sum(bool(r['trellis_index']) for r in result),'fingerprint':fingerprint(request)}),flush=True)

def scan_one(row,metadata):
 try:
  meta=metadata[(row['source'],row['raw_sha256'])];identity(row,meta)
  folder=SOURCE/row['source']/'renders_cond'/row['raw_sha256']
  if not row['trellis_index']:return {'uid':row['uid'],'status':'missing','reason':'not_in_trellis_image_index','render_directory_exists':folder.is_dir()}
  if not folder.is_dir():return {'uid':row['uid'],'status':'missing','reason':'render_directory_missing'}
  transform=folder/'transforms.json';before=transform.stat();raw=transform.read_bytes();after=transform.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('transforms changed')
  t=json.loads(raw)
  if not math.isfinite(float(t['scale'])) or float(t['scale'])<=0 or len(t['offset'])!=3 or not all(math.isfinite(float(x)) for x in t['offset']):raise ValueError('normalization metadata invalid')
  valid=[];invalid=[];seen=set()
  for original_index,frame in enumerate(t['frames']):
   filename=frame['file_path']
   if P(filename).name!=filename or not filename.endswith('.png') or filename in seen:raise ValueError('unsafe/duplicate frame filename')
   seen.add(filename)
   try:
    elevation,azimuth=angles(frame)
    if not -60<=elevation<=60:continue
    file=folder/filename
    if file.is_symlink() or not file.resolve().is_relative_to(folder.resolve()):raise ValueError('unsafe image path')
    info=image_info(file)
    valid.append({'filename':filename,'original_view_index':original_index,'elevation_degrees':elevation,'azimuth_degrees':azimuth,'horizontal_fov_radians':frame['camera_angle_x'],'camera_to_world':frame['transform_matrix'],**info})
   except Exception as exc:invalid.append({'file_path':filename,'error':repr(exc)})
  if not valid:return {'uid':row['uid'],'status':'missing','reason':'no_valid_view_in_elevation_range','invalid_frames':invalid}
  valid.sort(key=lambda r:r['filename']);selected=dict(valid[chosen_index(row['uid'],len(valid))])
  selected['sha256']=sha(folder/selected['filename'])
  index=row['trellis_index'];indexed=folder/P(index['image_path']).name
  if not indexed.is_file() or indexed.stat().st_size!=index['image_bytes'] or sha(indexed)!=index['image_sha256']:raise ValueError('historical indexed image SHA mismatch')
  return {'uid':row['uid'],'source':row['source'],'asset_id':row['asset_id'],'raw_sha256':row['raw_sha256'],'status':'selected','selected':selected,'eligible_filenames':[x['filename'] for x in valid],'eligible_frames':valid,'invalid_frames':invalid,'original_frame_count':len(t['frames']),'source_folder':str(folder),'transforms':{'bytes':len(raw),'mtime_ns':before.st_mtime_ns,'sha256':digest(raw)},'normalization':{k:t[k] for k in ['aabb','scale','offset']},'indexed_image_verified':True,'trellis_index':index}
 except Exception as exc:return {'uid':row['uid'],'status':'rejected','reason':repr(exc),'traceback':traceback.format_exc()}

def scan_source():
 req=json.load(sys.stdin)
 if req['policy']!=POLICY:raise RuntimeError('source policy mismatch')
 metadata={}
 for source in ['ABO','HSSD']:
  with (SOURCE/source/'metadata.csv').open() as f:
   for r in csv.DictReader(f):
    key=(source,r['sha256'])
    if key in metadata:raise ValueError('duplicate source metadata SHA')
    metadata[key]=r
 metadata_shas={s:sha(SOURCE/s/'metadata.csv') for s in ['ABO','HSSD']}
 resume=req.pop('resume_records',{})
 print(canonical({'kind':'start','request_fingerprint':fingerprint(req),'source_metadata_sha256':metadata_shas}),flush=True)
 def scan_or_reuse(row):
  saved=resume.get(row['uid'])
  if saved and saved.get('status')=='selected' and not saved.get('invalid_frames'):
   try:
    identity(row,metadata[(row['source'],row['raw_sha256'])])
    folder=SOURCE/row['source']/'renders_cond'/row['raw_sha256']
    for name,record in [(x['filename'],x) for x in saved['eligible_frames']]+[('transforms.json',saved['transforms'])]:
     file=folder/name;st=file.stat()
     if file.is_symlink() or st.st_size!=record['bytes'] or st.st_mtime_ns!=record['mtime_ns']:raise RuntimeError('source changed')
    if sha(folder/'transforms.json')!=saved['transforms']['sha256']:raise RuntimeError('metadata changed')
    if saved['trellis_index']!=row['trellis_index']:raise RuntimeError('index changed')
    return saved
   except Exception:pass
  return scan_one(row,metadata)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(scan_or_reuse,req['objects']):
   print(canonical({'kind':'object','record':result}),flush=True)
 print(canonical({'kind':'end','count':len(req['objects']),'utc':now()}),flush=True)

def receive_inventory():
 AUDIT.mkdir(parents=True,exist_ok=True)
 with (AUDIT/'inventory.lock').open('a') as mutex:
  fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
  req=read(AUDIT/'candidate_request.json');expected={r['uid'] for r in req['objects']};records={};started=time.time();header=None
  for line in sys.stdin:
   packet=json.loads(line)
   if packet['kind']=='start':
    if packet['request_fingerprint']!=fingerprint(req):raise RuntimeError('source inventory request mismatch')
    header=packet;continue
   if not header:raise RuntimeError('missing source header')
   if packet['kind']=='end':
    if set(records)!=expected or packet['count']!=len(expected):raise RuntimeError('incomplete source inventory')
    inventory={'request_fingerprint':fingerprint(req),'policy':POLICY,'source_metadata_sha256':header['source_metadata_sha256'],'objects':[records[k] for k in sorted(records)]}
    immutable(AUDIT/'source_selection_inventory.json',inventory)
    atomic(AUDIT/'source_scan_complete.json',{'objects':len(records),'selected':sum(r['status']=='selected' for r in records.values()),'inventory_sha256':sha(AUDIT/'source_selection_inventory.json'),'wall_seconds':time.time()-started,'utc':now()})
    print(json.dumps(read(AUDIT/'source_scan_complete.json')),flush=True);return
   r=packet['record'];uid=r['uid']
   if uid not in expected or uid in records:raise RuntimeError('unexpected/duplicate source UID')
   records[uid]=r
   # Per-UID evidence survives a broken SSH session; final inventory is committed only at EOF marker.
   record_path=AUDIT/'source_scan_records'/(uid+'.json')
   if record_path.exists() and read(record_path)!=r:
    history=AUDIT/'source_scan_history'/uid/(sha(record_path)+'.json')
    if not history.exists():atomic(history,read(record_path))
   atomic(record_path,r)
   if len(records)%20==0:
    state={'status':'source_scan','scanned':len(records),'total':len(expected),'elapsed_seconds':time.time()-started,'last_uid':uid,'utc':now()}
    atomic(AUDIT/'status.json',state)
    if len(records)%200==0:print(json.dumps(state),flush=True)
  raise EOFError('source inventory interrupted')

def freeze():
 status=read(ROOT/'results/abo_hssd_v1/geometry_full/status.json')
 if status['status'] not in ['complete','complete_with_failures'] or status['completed']+status['failed']!=status['total']:raise RuntimeError('geometry must finish before freeze')
 sys.path.insert(0,str(ROOT/'scripts/native_21dit/Abo_Hssd_datapre'))
 from core import cfg
 from pipeline_common import manifests,geometry_row
 splits,ready=manifests(cfg());splitmap={}
 for split,items in splits.items():
  for r in items:
   for alias in r['aliases']:
    if alias['uid'] in splitmap:raise RuntimeError('split alias duplication')
    splitmap[alias['uid']]={'split':split,'representative_uid':r['uid']}
 inventory=read(AUDIT/'source_selection_inventory.json');images={r['uid']:r for r in inventory['objects']}
 success=rows(ROOT/'results/abo_hssd_v1/geometry_full/events.jsonl');failed=rows(ROOT/'results/abo_hssd_v1/geometry_full/failures.jsonl')
 if len({r['uid'] for r in success+failed})!=status['total'] or set(r['uid'] for r in success)&set(r['uid'] for r in failed):raise RuntimeError('geometry terminal identities invalid')
 candidates=rows(ASSETS/'metadata/candidates.jsonl');selected=[];missing=[];geometry_rejected=[]
 good={r['uid'] for r in success}
 for row in candidates:
  uid=row['uid']
  if uid not in good:geometry_rejected.append({'uid':uid,'reason':'geometry_failed'});continue
  geo=geometry_row(row,cfg())
  if geo is None or uid not in splitmap:raise RuntimeError('qualified geometry missing from final split')
  view=images[uid]
  identity(row,{'sha256':view.get('raw_sha256',row['raw_sha256']),'file_identifier':view.get('asset_id',row['asset_id'])})
  if view['status']!='selected':
   missing.append({'uid':uid,'source':row['source'],**splitmap[uid],'reason':view.get('reason'),'status':view['status']});continue
  selected.append({**view,**splitmap[uid],'preprocessing_receipt':geo['preprocessing_receipt'],'preprocessing_receipt_sha256':geo['preprocessing_receipt_sha256'],'policy_fingerprint':fingerprint(POLICY),'target_folder':str(P('assets/abo_hssd_v1/renders_original')/uid),'render_subject':'original textured asset; not the watertight mesh','camera_alignment_to_vae':'not established'})
 frozen={'policy':POLICY,'source_inventory_sha256':sha(AUDIT/'source_selection_inventory.json'),'geometry_events_sha256':sha(ROOT/'results/abo_hssd_v1/geometry_full/events.jsonl'),'manifest_ready_sha256':sha(ASSETS/'manifests/abo_hssd_v1/ready.json'),'objects':selected,'missing_images':missing,'geometry_rejected':geometry_rejected}
 immutable(AUDIT/'frozen_selection.json',frozen)
 atomic(AUDIT/'selection_summary.json',{'geometry_qualified':len(good),'selected':len(selected),'missing_images':len(missing),'geometry_rejected':len(geometry_rejected),'payload_bytes':sum(r['selected']['bytes']+r['transforms']['bytes'] for r in selected),'selected_by_source':{s:sum(r['source']==s for r in selected) for s in ['ABO','HSSD']},'frozen_sha256':sha(AUDIT/'frozen_selection.json'),'utc':now()})
 print(json.dumps(read(AUDIT/'selection_summary.json')),flush=True)

def files_for(row): return [(row['selected']['filename'],row['selected']),('transforms.json',row['transforms'])]
def valid_target(row,full_sha=False):
 folder=ROOT/row['target_folder'];receipt=folder/'image_complete.json'
 if not folder.exists():return False
 if not receipt.exists():raise RuntimeError('unreceipted existing image target: '+str(folder))
 r=read(receipt)
 if r['row_fingerprint']!=fingerprint(row):raise RuntimeError('image target input changed')
 if {f.name for f in folder.iterdir()}!={row['selected']['filename'],'transforms.json','image_complete.json'}:raise RuntimeError('unexpected image target file')
 for name,expected in files_for(row):
  f=folder/name;st=f.stat();old=r['files'][name]
  if st.st_size!=expected['bytes'] or old['sha256']!=expected['sha256']:raise RuntimeError('target size/SHA receipt mismatch')
  if full_sha or st.st_mtime_ns!=old['mtime_ns']:
   if sha(f)!=expected['sha256']:raise RuntimeError('target changed SHA')
 return True
def request(pilot=False):
 frozen=read(AUDIT/'frozen_selection.json');items=frozen['objects']
 if pilot:
  chosen=[]
  for source in ['ABO','HSSD']:
   pool=sorted([r for r in items if r['source']==source],key=lambda r:r['uid'])
   random.Random(int(fingerprint(['image-pilot-v1',source])[:16],16)).shuffle(pool);chosen.extend(pool[:4])
  items=chosen
 pending=[r for r in items if not valid_target(r)]
 out={'policy':POLICY,'frozen_selection_sha256':sha(AUDIT/'frozen_selection.json'),'objects':pending,'total':len(items),'reused':len(items)-len(pending)}
 print(json.dumps(out),flush=True)
def emit(packet):
 b=canonical(packet).encode();sys.stdout.buffer.write(struct.pack('>I',len(b))+b);sys.stdout.buffer.flush()
def exact(n):
 out=bytearray()
 while len(out)<n:
  b=sys.stdin.buffer.read(n-len(out))
  if not b:raise EOFError('transfer interrupted')
  out.extend(b)
 return bytes(out)
def send_source():
 req=json.load(sys.stdin)
 if req['policy']!=POLICY:raise RuntimeError('send policy mismatch')
 for row in req['objects']:
  folder=P(row['source_folder']);expected=SOURCE/row['source']/'renders_cond'/row['raw_sha256']
  if folder!=expected or not folder.resolve().is_relative_to((SOURCE/row['source']).resolve()):raise ValueError('source path not approved')
  emit({'kind':'object','row':row})
  for filename,record in files_for(row):
   f=folder/filename;before=f.stat()
   if before.st_size!=record['bytes'] or before.st_mtime_ns!=record['mtime_ns'] or f.is_symlink():raise RuntimeError('source changed since selection')
   emit({'kind':'file','filename':filename,'sha256':record['sha256'],'bytes':record['bytes']})
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk);sys.stdout.buffer.write(chunk)
   sys.stdout.buffer.flush();after=f.stat()
   if h.hexdigest()!=record['sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('source SHA/stat changed during stream')
   emit({'kind':'file_end','sha256':h.hexdigest()})
  emit({'kind':'object_end','uid':row['uid']})
 emit({'kind':'end','objects':len(req['objects']),'utc':now()})
def packet():
 n=struct.unpack('>I',exact(4))[0]
 if n>8*1024**2:raise ValueError('oversized header')
 return json.loads(exact(n))
def receive():
 AUDIT.mkdir(parents=True,exist_ok=True)
 with (AUDIT/'transfer.lock').open('a') as mutex:
  fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
  frozen=read(AUDIT/'frozen_selection.json');lookup={r['uid']:r for r in frozen['objects']};count=total=0;start=time.monotonic()
  try:
   while True:
    head=packet()
    if head['kind']=='end':
     if head['objects']!=count:raise RuntimeError('object count mismatch')
     print(json.dumps({'status':'stream_complete','objects':count,'bytes':total,'seconds':time.monotonic()-start}),flush=True);return
    if head['kind']!='object':raise RuntimeError('expected object')
    row=head['row']
    if row!=lookup.get(row['uid']):raise RuntimeError('object violates frozen inventory')
    final=ROOT/row['target_folder']
    if final.exists():raise RuntimeError('refusing existing target overwrite')
    if not final.resolve().is_relative_to((ASSETS/'renders_original').resolve()):raise RuntimeError('unsafe target')
    temp=final.with_name('.partial_'+final.name);temp.mkdir(parents=True,exist_ok=True)
    records={}
    for filename,expected in files_for(row):
     h=packet()
     if h!={'kind':'file','filename':filename,'sha256':expected['sha256'],'bytes':expected['bytes']}:raise RuntimeError('file header mismatch')
     out=temp/filename;checksum=hashlib.sha256();remaining=expected['bytes']
     with out.open('wb') as stream:
      while remaining:
       chunk=exact(min(1024**2,remaining));stream.write(chunk);checksum.update(chunk);remaining-=len(chunk)
      stream.flush();os.fsync(stream.fileno())
     trailer=packet()
     if trailer!={'kind':'file_end','sha256':expected['sha256']} or checksum.hexdigest()!=expected['sha256']:raise RuntimeError('stream source/target SHA mismatch')
     st=out.stat();records[filename]={'sha256':checksum.hexdigest(),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns};total+=st.st_size
    if packet()!={'kind':'object_end','uid':row['uid']}:raise RuntimeError('object end mismatch')
    info=image_info(temp/row['selected']['filename']);angle,_=angles(read(temp/'transforms.json')['frames'][row['selected']['original_view_index']])
    if abs(angle-row['selected']['elevation_degrees'])>1e-10:raise RuntimeError('copied camera angle mismatch')
    atomic(temp/'image_complete.json',{'uid':row['uid'],'row_fingerprint':fingerprint(row),'files':records,'utc':now()})
    os.rename(temp,final);count+=1
    log('transfer_events.jsonl',{'uid':row['uid'],'bytes':sum(x['bytes'] for x in records.values()),'seconds':time.monotonic()-start})
    if count==1 or count%20==0:
     state={'status':'receiving_images','session_objects':count,'session_bytes':total,'seconds':time.monotonic()-start,'last_uid':row['uid'],'pid':os.getpid(),'utc':now()};atomic(AUDIT/'status.json',state)
     if count==1 or count%100==0:print(json.dumps(state),flush=True)
  except Exception as exc:
   state={'status':'interrupted','error':repr(exc),'objects':count,'bytes':total,'utc':now()};atomic(AUDIT/'status.json',state);log('failures.jsonl',state);raise

def verify(pilot=False):
 frozen=read(AUDIT/'frozen_selection.json')
 items=frozen['objects']
 if pilot:
  chosen=[]
  for s in ['ABO','HSSD']:
   pool=sorted([r for r in items if r['source']==s],key=lambda r:r['uid']);random.Random(int(fingerprint(['image-pilot-v1',s])[:16],16)).shuffle(pool);chosen.extend(pool[:4])
  items=chosen
 start=time.monotonic();verified=[];missing=[]
 for i,r in enumerate(items):
  if not valid_target(r,True):missing.append(r['uid']);continue
  info=image_info(ROOT/r['target_folder']/r['selected']['filename'])
  t=read(ROOT/r['target_folder']/'transforms.json');frame=t['frames'][r['selected']['original_view_index']]
  e,a=angles(frame)
  if frame['file_path']!=r['selected']['filename'] or not -60<=e<=60 or abs(e-r['selected']['elevation_degrees'])>1e-10:raise RuntimeError('final angle mismatch')
  if sha(ROOT/r['preprocessing_receipt'])!=r['preprocessing_receipt_sha256']:raise RuntimeError('geometry receipt changed')
  verified.append({**r,'image_path':str(P(r['target_folder'])/r['selected']['filename']),'image_sha256':r['selected']['sha256'],'transforms_path':str(P(r['target_folder'])/'transforms.json')})
  if i%100==0:atomic(AUDIT/'verification_progress.json',{'verified':len(verified),'total':len(items),'pilot':pilot,'utc':now()})
 report={'passed':not missing,'verified':len(verified),'expected':len(items),'missing_targets':missing,'wall_seconds':time.monotonic()-start,'frozen_sha256':sha(AUDIT/'frozen_selection.json'),'utc':now(),'data_ready':False}
 if pilot:
  immutable(AUDIT/'pilot_acceptance.json',report) if not (AUDIT/'pilot_acceptance.json').exists() else atomic(AUDIT/'pilot_reuse_verification.json',report)
 else:
  ready=ASSETS/'manifests/abo_hssd_v1/ready.json'
  if sha(ready)!=frozen['manifest_ready_sha256']:raise RuntimeError('immutable split manifest changed')
  out=ASSETS/'renders_original'
  jsonlines(out/'paired_index.jsonl',verified);jsonlines(out/'missing_images.jsonl',frozen['missing_images']);jsonlines(out/'geometry_rejected.jsonl',frozen['geometry_rejected'])
  report.update({'geometry_qualified':len(items)+len(frozen['missing_images']),'image_coverage':len(verified)/(len(items)+len(frozen['missing_images'])),'missing_source_images':len(frozen['missing_images']),'payload_bytes':sum(r['selected']['bytes']+r['transforms']['bytes'] for r in verified),'sources':{s:sum(r['source']==s for r in verified) for s in ['ABO','HSSD']},'splits':{s:sum(r['split']==s for r in verified) for s in ['train','val','test']},'paired_index_sha256':sha(out/'paired_index.jsonl'),'not_watertight_render':True,'training_started':False,'gpu_used':False})
  atomic(AUDIT/'final_verification.json',report);atomic(out/'migration_complete.json',report)
 print(json.dumps(report),flush=True)
 if missing:raise RuntimeError('expected target files missing')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','scan_source','inventory_receive','freeze','request','send_source','receive','verify']);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 if args.mode in ['scan_source','send_source']:globals()[args.mode]();return
 os.environ['CUDA_VISIBLE_DEVICES']=''
 if args.mode=='prepare':prepare_candidates()
 elif args.mode=='inventory_receive':receive_inventory()
 elif args.mode=='freeze':freeze()
 elif args.mode=='request':request(args.pilot)
 elif args.mode=='receive':receive()
 elif args.mode=='verify':verify(args.pilot)
if __name__=='__main__':main()
