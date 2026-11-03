"""Windows memory-only SSH relay; all evidence and payload files remain in userdata."""
import base64,json,os,shlex,subprocess,sys,time,zlib,traceback
ROOT='/root/userdata/zc/hunyuan3d-buffalo-1.0-editing'
AUDIT=ROOT+'/results/abo_hssd_v1/render_images_v1'
SCRIPT=ROOT+'/scripts/native_21dit/Abo_Hssd_datapre/render_images.py'
RUNTIME=ROOT+'/environment/venvs/shape/bin/python'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ForwardAgent=no','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6']
def hidden_process_options():
 if os.name!='nt':return {}
 info=subprocess.STARTUPINFO()
 info.dwFlags|=subprocess.STARTF_USESHOWWINDOW
 info.wShowWindow=0
 return {'creationflags':subprocess.CREATE_NO_WINDOW,'startupinfo':info}
def remote(code):return 'python3 -B -c '+shlex.quote(code)
def call(code,data=None,timeout=180):
 p=subprocess.run(SSH+['h100',remote(code)],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,**hidden_process_options())
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-5000:])
 return p.stdout
def read(file):return json.loads(call('import pathlib;print(pathlib.Path('+repr(AUDIT+'/'+file)+').read_text())'))
def exists(file):return call('import pathlib;print(int(pathlib.Path('+repr(AUDIT+'/'+file)+').exists()))').strip()==b'1'
def stage(mode,pilot=False,timeout=180):
 cmd=shlex.quote(RUNTIME)+' -B '+shlex.quote(SCRIPT)+' '+mode+(' --pilot' if pilot else '')
 p=subprocess.run(SSH+['h100',cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,**hidden_process_options())
 if p.returncode:raise RuntimeError(mode+': '+p.stderr.decode(errors='replace')[-6000:])
 return json.loads(p.stdout)
def event(kind,**extra):
 value={'kind':kind,'local_pid':os.getpid(),'utc_epoch':time.time(),**extra}
 code="import sys,pathlib,json,os;d=pathlib.Path("+repr(AUDIT)+");d.mkdir(parents=True,exist_ok=True);v=json.load(sys.stdin);f=d/'relay_status.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,f);h=(d/'relay_events.jsonl').open('a');h.write(json.dumps(v)+chr(10));h.close();print(json.dumps(v))"
 print(call(code,json.dumps(value).encode()).decode().strip(),flush=True)
def source_command(source,mode):
 compressed=base64.b64encode(zlib.compress(source,9)).decode()
 bootstrap="import base64,zlib,sys;sys.argv=['render_images.py',"+repr(mode)+"];exec(compile(zlib.decompress(base64.b64decode("+repr(compressed)+")),'render_images.py','exec'))"
 return remote(bootstrap)
def relay(source,mode,destination_mode,request):
 dest=subprocess.Popen(SSH+['h100',shlex.quote(RUNTIME)+' -B '+shlex.quote(SCRIPT)+' '+destination_mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,**hidden_process_options())
 src=subprocess.Popen(SSH+['4090',source_command(source,mode)],stdin=subprocess.PIPE,stdout=dest.stdin,stderr=subprocess.PIPE,bufsize=0,**hidden_process_options())
 dest.stdin.close()
 try:
  src.stdin.write(json.dumps(request).encode());src.stdin.close()
  for line in iter(dest.stdout.readline,b''):
   text=line.decode(errors='replace').rstrip();print(text,flush=True)
  drc=dest.wait();stderr=src.stderr.read().decode(errors='replace');src_rc=src.wait()
 finally:
  if dest.poll() is None:dest.terminate()
  if src.poll() is None:src.terminate()
 event('stream_exit',mode=mode,destination_exit=drc,source_exit=src_rc,source_stderr=stderr[-4000:])
 if drc or src_rc:raise RuntimeError('SSH stream did not complete')
def geometry_state():
 return json.loads(call('import pathlib;print(pathlib.Path('+repr(ROOT+'/results/abo_hssd_v1/geometry_full/status.json')+').read_text())'))
def manifest_job():
 code="""import pathlib,json,os,subprocess,sys
p=pathlib.Path(ROOT);out=pathlib.Path(AUDIT);result=out/'manifest_exit.json';process=out/'manifest_process.json'
if result.exists():
 print(result.read_text());sys.exit()
if process.exists():
 r=json.loads(process.read_text())
 if pathlib.Path('/proc',str(r['pid'])).exists():print(json.dumps({'running':True,**r}));sys.exit()
 raise RuntimeError('manifest supervisor disappeared; inspect logs')
worker='import subprocess,json,pathlib,os;cmd='+repr([str(p/'environment/venvs/abo_hssd/bin/python'),'-B',str(p/'scripts/native_21dit/Abo_Hssd_datapre/build_manifest.py')])+';out=pathlib.Path('+repr(str(out))+');f=(out/"manifest.log").open("ab");r=subprocess.run(cmd,cwd='+repr(str(p))+',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"));(out/"manifest_exit.json").write_text(json.dumps({"returncode":r.returncode}))'
with (out/'manifest_supervisor.log').open('ab') as f:
 child=subprocess.Popen([sys.executable,'-B','-c',worker],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r={'pid':child.pid};process.write_text(json.dumps(r));print(json.dumps({'started':True,**r}))
""".replace('ROOT',repr(ROOT)).replace('AUDIT',repr(AUDIT))
 return json.loads(call(code))
def preflight(source):
 if exists('source_scan_complete.json'):
  event('source_inventory_reused',summary=read('source_scan_complete.json'));return
 req=read('candidate_request.json')
 cache=json.loads(call("import pathlib,json;d=pathlib.Path("+repr(AUDIT)+")/'source_scan_records';print(json.dumps({q.stem:json.loads(q.read_text()) for q in d.glob('*.json')}))"))
 req['resume_records']=cache
 event('source_inventory_start',objects=len(req['objects']),resume_records=len(cache))
 relay(source,'scan_source','inventory_receive',req)
def transfer(source,pilot):
 for attempt in range(1,5):
  req=stage('request',pilot)
  if not req['objects']:
   event('transfer_reused' if attempt==1 else 'transfer_complete',pilot=pilot,reused=req['reused'],total=req['total']);return
  event('transfer_start',pilot=pilot,attempt=attempt,pending=len(req['objects']),payload_bytes=sum(r['selected']['bytes']+r['transforms']['bytes'] for r in req['objects']))
  try:relay(source,'send_source','receive',req)
  except Exception as exc:
   event('transfer_interrupted',pilot=pilot,attempt=attempt,error=repr(exc))
   if attempt==4:raise
   time.sleep(min(30,5*attempt));continue
  return
def main():
 event('relay_started',gpu_used=False)
 source=call('import pathlib;import sys;sys.stdout.buffer.write(pathlib.Path('+repr(SCRIPT)+').read_bytes())')
 if not exists('candidate_request.json'):event('candidate_preparation',result=stage('prepare',timeout=1200))
 for attempt in range(1,4):
  try:preflight(source);break
  except Exception as exc:
   event('source_inventory_interrupted',attempt=attempt,error=repr(exc))
   if attempt==3:raise
   time.sleep(15)
 last=None
 while True:
  state=geometry_state()
  if state['status'] in ['complete','complete_with_failures'] and state['completed']+state['failed']==state['total']:break
  pid=state.get('pid');alive=call('import pathlib;print(int(pathlib.Path("/proc",'+repr(str(pid))+').exists()))').strip()==b'1'
  if not alive:raise RuntimeError('geometry process absent before terminal; image workflow will not restart it')
  if (state['completed'],state['failed'])!=last:
   event('waiting_geometry',completed=state['completed'],failed=state['failed'],total=state['total']);last=(state['completed'],state['failed'])
  time.sleep(30)
 event('geometry_terminal',state=state)
 ready=ROOT+'/assets/abo_hssd_v1/manifests/abo_hssd_v1/ready.json'
 if call('import pathlib;print(int(pathlib.Path('+repr(ready)+').exists()))').strip()!=b'1':
  job=manifest_job();event('cpu_manifest_stage',job=job)
  while not exists('manifest_exit.json'):time.sleep(30)
  result=read('manifest_exit.json')
  if result['returncode']:raise RuntimeError('existing CPU manifest builder failed; see manifest.log')
 event('frozen_selection',summary=stage('freeze',timeout=2400))
 if not exists('pilot_acceptance.json'):
  started=time.time();transfer(source,True);report=stage('verify',True,timeout=300)
  if not report['passed'] or report['verified']!=8:raise RuntimeError('eight-object pilot not satisfied')
  before=json.loads(call("import pathlib,json;d=pathlib.Path("+repr(ROOT+'/assets/abo_hssd_v1/renders_original')+");print(json.dumps({str(f):[f.stat().st_size,f.stat().st_mtime_ns] for q in d.iterdir() if q.is_dir() for f in q.iterdir()}))"))
  transfer(source,True);reused=stage('verify',True,timeout=300)
  after=json.loads(call("import pathlib,json;d=pathlib.Path("+repr(ROOT+'/assets/abo_hssd_v1/renders_original')+");print(json.dumps({str(f):[f.stat().st_size,f.stat().st_mtime_ns] for q in d.iterdir() if q.is_dir() for f in q.iterdir()}))"))
  if before!=after:raise RuntimeError('pilot restart changed payloads')
  event('pilot_passed',verified=report['verified'],reuse_verified=reused['verified'],payloads_unchanged=True,elapsed_seconds=time.time()-started)
 transfer(source,False)
 event('final_verification_start')
 report=stage('verify',False,timeout=3600)
 event('image_migration_complete',report=report)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  try:event('workflow_failed',error=repr(exc),traceback=traceback.format_exc())
  except Exception:pass
  raise
