"""Pinned, resumable CPU preprocessing; no training or GPU work."""
import argparse,json,os,random,subprocess,sys,time,traceback,signal
from pathlib import Path
from core import ROOT,atomic,cfg,check_upstream,dirs,fp,lock,now,path,readlines,seed,sha,append,setup

def select(rows,c):
 out=[]
 for source in c['sources']:
  pool=sorted([r for r in rows if r['source']==source],key=lambda r:r['uid'])
  random.Random(seed(c['seed'],'geometry-pilot',source)).shuffle(pool);out.extend(pool[:4])
 return out

def record_file(f):
 st=f.stat();return {'sha256':sha(f),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}

def verify_files(base,files):
 for rel,r in files.items():
  f=base/rel
  if not f.is_file() or f.stat().st_size!=r['bytes']:raise RuntimeError('missing or changed payload: '+str(f))
  if f.stat().st_mtime_ns!=r['mtime_ns'] and sha(f)!=r['sha256']:raise RuntimeError('changed payload SHA: '+str(f))

def stage_valid(folder,name,contract):
 f=folder/'.stages'/(name+'.json')
 if not f.exists():return False
 r=json.loads(f.read_text())
 if r['contract']!=contract:raise RuntimeError('stage fingerprint changed: '+str(f))
 verify_files(folder,r['files']);return True

def stage_commit(folder,name,contract,files,extra=None):
 atomic(folder/'.stages'/(name+'.json'),{'contract':contract,'fingerprint':fp(contract),'files':{str(f.relative_to(folder)):record_file(f) for f in files},**(extra or {}),**now()})

def run(command,log,timeout,folder):
 log.parent.mkdir(parents=True,exist_ok=True);stamp=time.time_ns();resource=log.with_name(log.name+'.'+str(stamp)+'.resource.txt')
 start=time.monotonic();peak=0
 with log.open('ab') as stream:
  stream.write(('\n=== '+json.dumps(now())+' '+json.dumps(command)+' ===\n').encode());stream.flush()
  monitor='import sys,subprocess,resource,json,time; t=time.monotonic(); r=subprocess.run(sys.argv[2:]); u=resource.getrusage(resource.RUSAGE_CHILDREN); open(sys.argv[1],\"w\").write(json.dumps({\"max_rss_kib\":u.ru_maxrss,\"user_seconds\":u.ru_utime,\"system_seconds\":u.ru_stime,\"wall_seconds\":time.monotonic()-t})); sys.exit(r.returncode)'
  proc=subprocess.Popen([sys.executable,'-B','-c',monitor,str(resource)]+command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   peak=max(peak,sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()))
   if time.monotonic()-start>timeout:
    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30);raise TimeoutError(str(command))
   time.sleep(2)
 elapsed=time.monotonic()-start
 result={'command':command,'returncode':proc.returncode,'wall_seconds':elapsed,'disk_peak_observed_bytes':peak,'resource_log':str(resource.relative_to(ROOT))}
 if resource.exists():result.update(json.loads(resource.read_text()))
 append(path('results/abo_hssd_v1/resource_usage.jsonl'),result)
 if proc.returncode:raise RuntimeError('command exited '+str(proc.returncode)+'; '+str(log))
 return result

def tool_identity(blender,python):
 probe='import json,sys,numpy,scipy,trimesh,igl,importlib.metadata as m;print(json.dumps({"python":sys.version,"numpy":numpy.__version__,"scipy":scipy.__version__,"trimesh":trimesh.__version__,"libigl":m.version("libigl")}))'
 versions=json.loads(subprocess.check_output([str(python),'-B','-c',probe],text=True))
 versions.update({'blender_sha256':sha(blender),'blender_version':subprocess.check_output([str(blender),'--version'],text=True).splitlines()[:4]})
 atomic('results/abo_hssd_v1/preparation/tool_identity.json',{'tools':versions,'fingerprint':fp(versions),**now()});return versions

def contracts(row,c,tools,render,water):
 run_seed=seed(c['seed'],row['uid'],'upstream-geometry')
 conversion={'raw_sha256':row['raw_sha256'],'render_sha256':sha(render),'upstream_commit':c['upstream_commit'],'blender_sha256':tools['blender_sha256'],'blender_version':tools['blender_version'],'geo_mode':True,'views':0,'resolution':512,'seed':run_seed,'seed_wrapper':'random-numpy-v1'}
 water_base={'upstream_commit':c['upstream_commit'],'watertight_sha256':sha(water),'runtime':{k:tools[k] for k in ['python','numpy','scipy','trimesh','libigl']},'grid_res':256,'epsilon':2/256,'sample_count':124928,'seed':run_seed,'seed_wrapper':'numpy-default-rng-same-seed-truncnorm-v1'}
 return conversion,water_base

def prepare_one(row,c,inventory_sha,blender,python,tools,render,water,output):
 assets,results=dirs(c);uid=row['uid'];raw=assets/'raw'/row['source']/row['asset_id']
 migrated=json.loads((results/'migration_20261027/receipts'/(uid+'.json')).read_text())
 if migrated['target_sha256']!=row['raw_sha256']:raise RuntimeError('migration SHA mismatch')
 # Migration verification already hashed the raw files. Check metadata and only hash if changed.
 st=raw.stat();expected_size=migrated.get('bytes',migrated.get('size'))
 if expected_size is not None and st.st_size!=expected_size:raise RuntimeError('raw size mismatch')
 if st.st_mtime_ns!=migrated['target_mtime_ns'] and sha(raw)!=row['raw_sha256']:raise RuntimeError('raw SHA mismatch')
 base=output/uid
 if (assets/'processed_pilot'/uid).exists():base=assets/'processed_pilot'/uid
 final=base/'upstream_geometry_v1';base.mkdir(parents=True,exist_ok=True)
 partials=sorted(base.glob('.tmp.*'))
 if final.exists():temp=final
 elif len(partials)>1:raise RuntimeError('multiple partial directories')
 else:temp=partials[0] if partials else base/('.tmp.'+str(os.getpid()));temp.mkdir(exist_ok=True)
 converted=temp/'converted';converted.mkdir(exist_ok=True);ply=converted/'mesh.ply';prefix=temp/uid
 cc,wc=contracts(row,c,tools,render,water)
 boot=path('tmp/abo_hssd_geometry');boot.mkdir(parents=True,exist_ok=True)
 old=temp/'complete.json'
 if old.exists() and not (temp/'.stages/conversion.json').exists():
  legacy=json.loads(old.read_text());lc=legacy['contract']
  for key,val in [('raw_sha256',row['raw_sha256']),('render_sha256',cc['render_sha256']),('watertight_sha256',wc['watertight_sha256']),('upstream_commit',c['upstream_commit']),('geometry',c['geometry'])]:
   if lc[key]!=val:raise RuntimeError('legacy provenance differs: '+key)
  verify_files(temp,legacy['files'])
  history=results/'preparation/legacy_receipts'/uid;history.mkdir(parents=True,exist_ok=True)
  saved=history/(sha(old)+'.json')
  if not saved.exists():saved.write_bytes(old.read_bytes())
  evidence={'legacy_receipt':str(saved.relative_to(ROOT)),'legacy_receipt_sha256':sha(saved),'tool_identity_observed_at_adoption':True,'note':'Geometry bytes retained; prior upstream and seed launchers verified; current quality module reruns validation only.'}
  # Existing launchers must prove the same per-UID seed and pinned upstream entrypoints.
  wb=(boot/(uid+'_watertight.py')).read_text();bb=(boot/(uid+'_blender.py')).read_text()
  if ('random.seed('+str(cc['seed'])+')') not in wb or ('random.seed('+str(cc['seed'])+')') not in bb or str(water) not in wb or str(render) not in bb:raise RuntimeError('legacy seed/entrypoint evidence missing')
  stage_commit(temp,'conversion',cc,list(converted.glob('*')),evidence)
  wc['input_ply_sha256']=sha(ply)
  stage_commit(temp,'water',wc,[Path(str(prefix)+s) for s in ['_watertight.obj','_surface.npz','_sdf.npz']],evidence)
 if not stage_valid(temp,'conversion',cc):
  launcher=boot/(uid+'_blender.py');launcher.write_text('import random,runpy,sys,numpy as np\nrandom.seed('+str(cc['seed'])+');np.random.seed('+str(cc['seed'])+')\nsys.argv='+repr(['blender','--','--object',str(raw),'--output_folder',str(converted),'--geo_mode','--views','0','--resolution','512'])+'\nrunpy.run_path('+repr(str(render))+',run_name="__main__")\n')
  atomic(temp/'.stages/conversion.started.json',{'contract':cc,**now()})
  res=run([str(blender),'-b','-P',str(launcher)],boot/(uid+'_blender.log'),7200,temp)
  if not ply.is_file() or not ply.stat().st_size or list(converted.rglob('*.png')):raise RuntimeError('views=0 geometry export invariant')
  stage_commit(temp,'conversion',cc,[f for f in converted.rglob('*') if f.is_file()],{'resource':res})
 if list(converted.rglob('*.png')):raise RuntimeError('condition image unexpectedly generated')
 wc['input_ply_sha256']=sha(ply)
 if not stage_valid(temp,'water',wc):
  launcher=boot/(uid+'_watertight.py');rs=cc['seed']
  launcher.write_text('import random,runpy,sys,numpy as np\nrandom.seed('+str(rs)+'); np.random.seed('+str(rs)+')\n_rng=np.random.default_rng; np.random.default_rng=lambda seed=None:_rng('+str(rs)+' if seed is None else seed)\nfrom scipy.stats import truncnorm\n_old=truncnorm.rvs\ntruncnorm.rvs=lambda *a,**k:_old(*a,random_state=np.random,**k)\nsys.argv='+repr([str(water),'--input_obj',str(ply),'--output_prefix',str(prefix)])+'\nrunpy.run_path('+repr(str(water))+',run_name="__main__")\n')
  atomic(temp/'.stages/water.started.json',{'contract':wc,**now()})
  res=run([str(python),'-B',str(launcher)],boot/(uid+'_watertight.log'),7200,temp)
  stage_commit(temp,'water',wc,[Path(str(prefix)+s) for s in ['_watertight.obj','_surface.npz','_sdf.npz']],{'resource':res})
 vc={'water_receipt_sha256':sha(temp/'.stages/water.json'),'quality_sha256':sha(Path(__file__).with_name('quality.py')),'vae_extent':1.9998,'vae_points':81920,'points_seed':seed(c['seed'],'vae-points',uid)}
 if not stage_valid(temp,'validation',vc):
  script=Path(__file__).with_name('quality.py');qlog=boot/(uid+'_quality.json')
  with qlog.open('w') as stream:
   proc=subprocess.run([str(python),'-B',str(script),str(temp),uid,str(vc['points_seed'])],stdout=stream,stderr=subprocess.PIPE,text=True,timeout=1800)
  if proc.returncode:raise RuntimeError('geometry validation failed: '+proc.stderr[-4000:])
  quality=json.loads(qlog.read_text().strip().splitlines()[-1]);payload=[Path(str(prefix)+s) for s in ['_vae.obj','_vae_points.npy','_vae_random_surface.npy','_coordinate_transform.json']]
  stage_commit(temp,'validation',vc,payload,{'quality':quality})
 quality=json.loads((temp/'.stages/validation.json').read_text())['quality']
 contract={'schema':2,'uid':uid,'raw_sha256':row['raw_sha256'],'conversion_fingerprint':fp(cc),'water_fingerprint':fp(wc),'validation_fingerprint':fp(vc),'caption_hash':row['caption_hash'],'inventory_sha256':inventory_sha}
 if old.exists() and json.loads(old.read_text()).get('contract')==contract:
  verify_files(temp,json.loads(old.read_text())['files']);return {'uid':uid,'status':'reused','folder':str(final.relative_to(ROOT)),'quality':quality}
 files={str(f.relative_to(temp)):record_file(f) for f in temp.rglob('*') if f.is_file() and f.name!='complete.json' and not f.name.endswith('.started.json')}
 atomic(old,{'contract':contract,'contract_sha256':fp(contract),'files':files,'quality':quality,'uid_seed':cc['seed'],**now()})
 if temp!=final:os.rename(temp,final)
 return {'uid':uid,'status':'complete','folder':str(final.relative_to(ROOT)),'quality':quality}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','full'],default='pilot');parser.add_argument('--blender',required=True);parser.add_argument('--python',required=True);args=parser.parse_args()
 c=cfg();check_upstream(c);assets,results=dirs(c);blender=path(args.blender);python=Path(os.path.abspath(args.python))
 if not python.is_relative_to(ROOT) or not python.is_file():raise ValueError('dedicated Python entrypoint outside workspace')
 rows=readlines(assets/'metadata/candidates.jsonl');inv=sha(results/'migration_20261027/source_inventory.json')
 if inv!='df419ae645debf29ab79b8ca6cb7f1c98b47cad1045207c967201f2e3c95fc94':raise RuntimeError('inventory fingerprint')
 chosen=select(rows,c) if args.mode=='pilot' else rows;audit=results/('geometry_pilot' if args.mode=='pilot' else 'geometry_full');audit.mkdir(parents=True,exist_ok=True)
 with lock('environment/abo_hssd_geometry.lock'):
  tools=tool_identity(blender,python);atomic(audit/'selection.json',{'selected':[r['uid'] for r in chosen],'seed':c['seed'],'inventory_sha256':inv,**now()})
  done=0;failed=0
  for row in chosen:
   atomic(audit/'status.json',{'status':'running','pid':os.getpid(),'completed':done,'failed':failed,'total':len(chosen),'active_uid':row['uid'],'data_ready':False,**now()})
   try:
    res=prepare_one(row,c,inv,blender,python,tools,path('src/Hunyuan3D-2.1/hy3dshape/tools/render/render.py'),path('src/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample.py'),assets/('processed_pilot' if args.mode=='pilot' else 'processed'));append(audit/'events.jsonl',res);done+=1
   except Exception as exc:
    failed+=1;append(audit/'failures.jsonl',{'uid':row['uid'],'error':repr(exc),'traceback':traceback.format_exc()})
    if args.mode=='pilot':
     atomic(audit/'status.json',{'status':'failed','completed':done,'failed':failed,'total':len(chosen),'uid':row['uid'],'data_ready':False,**now()});raise
  atomic(audit/'status.json',{'status':'complete' if not failed else 'complete_with_failures','completed':done,'failed':failed,'total':len(chosen),'data_ready':False,**now()})

if __name__=='__main__':setup();main()
