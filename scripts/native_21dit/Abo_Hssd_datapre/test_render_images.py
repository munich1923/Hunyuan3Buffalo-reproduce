"""CPU-only image migration contract and streaming recovery tests."""
import contextlib,copy,io,json,math,random,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import render_images as m

def frame(elevation=0,fov=.7):
 t=math.radians(elevation)
 return {'file_path':'003.png','camera_angle_x':fov,'transform_matrix':[[1,0,0,2*math.cos(t)],[0,1,0,0],[0,0,1,2*math.sin(t)],[0,0,0,1]]}
class Images(unittest.TestCase):
 def test_angles_and_boundaries(self):
  for e in [-85,-60,-20,0,20,60,85]:
   self.assertAlmostEqual(m.angles(frame(e))[0],e)
  self.assertEqual([e for e in [-60,0,60] if -60<=m.angles(frame(e))[0]<=60],[-60,0,60])
  self.assertAlmostEqual(m.angles(frame(0,1.1))[0],0)
  for invalid in [dict(frame(),camera_angle_x=0),dict(frame(),transform_matrix=[[1]*4]*4)]:
   with self.assertRaises(ValueError):m.angles(invalid)
 def test_random_reproducibility(self):
  old=random.getstate()
  a=[m.chosen_index('object'+str(i),21) for i in range(300)]
  self.assertEqual(a,[m.chosen_index('object'+str(i),21) for i in range(300)])
  self.assertEqual(old,random.getstate());self.assertGreater(len(set(a)),15)
  with self.assertRaises(ValueError):m.chosen_index('uid',0)
 def test_uid_path_sha(self):
  row={'source':'ABO','raw_sha256':'abc','uid':'abo_abc','asset_id':'3/model.glb'}
  m.identity(row,{'sha256':'abc','file_identifier':'3/model.glb'})
  for changed in [dict(row,uid='hssd_abc'),dict(row,asset_id='other')]:
   with self.assertRaises(ValueError):m.identity(changed,{'sha256':'abc','file_identifier':'3/model.glb'})
 def test_missing_no_eligible(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   root=Path(name);folder=root/'ABO/renders_cond/abc';folder.mkdir(parents=True)
   m.atomic(folder/'transforms.json',{'scale':1.,'offset':[0,0,0],'aabb':[[-.5]*3,[.5]*3],'frames':[frame(85)]})
   row={'source':'ABO','uid':'abo_abc','raw_sha256':'abc','asset_id':'3/a.glb','trellis_index':{'image_path':str(folder/'000.png')}}
   with patch.object(m,'SOURCE',root):r=m.scan_one(row,{('ABO','abc'):{'sha256':'abc','file_identifier':'3/a.glb'}})
   self.assertEqual(r['reason'],'no_valid_view_in_elevation_range')
 def test_stream_interruption_reuse_corruption(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   root=Path(name);audit=root/'audit';audit.mkdir();asset=root/'assets';source=root/'original';source.mkdir()
   png=source/'003.png';Image.new('RGBA',(1024,1024),(25,65,75,127)).save(png);transform=source/'transforms.json'
   m.atomic(transform,{'scale':1.,'offset':[0,0,0],'aabb':[[-.5]*3,[.5]*3],'frames':[frame(20)]})
   info=m.image_info(png);info['sha256']=m.sha(png)
   row={'uid':'abo_abc','source':'ABO','raw_sha256':'abc','target_folder':'assets/renders_original/abo_abc','selected':dict(info,filename='003.png',original_view_index=0,elevation_degrees=20.),'transforms':{'sha256':m.sha(transform),'bytes':transform.stat().st_size,'mtime_ns':transform.stat().st_mtime_ns}}
   def packet(v):
    b=m.canonical(v).encode();return struct.pack('>I',len(b))+b
   parts=[packet({'kind':'object','row':row})]
   for filename,r in m.files_for(row):
    parts+=[packet({'kind':'file','filename':filename,'sha256':r['sha256'],'bytes':r['bytes']}),(source/filename).read_bytes(),packet({'kind':'file_end','sha256':r['sha256']})]
   parts+=[packet({'kind':'object_end','uid':row['uid']}),packet({'kind':'end','objects':1})]
   payload=b''.join(parts);m.atomic(audit/'frozen_selection.json',{'objects':[row]})
   class Input:
    def __init__(self,data):self.buffer=io.BytesIO(data)
   with patch.multiple(m,ROOT=root,ASSETS=asset,AUDIT=audit):
    with patch('sys.stdin',Input(payload[:len(payload)//2])),contextlib.redirect_stdout(io.StringIO()):
     with self.assertRaises(EOFError):m.receive()
    self.assertFalse((root/row['target_folder']).exists())
    with patch('sys.stdin',Input(payload)),contextlib.redirect_stdout(io.StringIO()):m.receive()
    self.assertTrue(m.valid_target(row,True))
    target=root/row['target_folder']/'003.png';before=target.stat().st_mtime_ns
    self.assertTrue(m.valid_target(row));self.assertEqual(before,target.stat().st_mtime_ns)
    data=bytearray(target.read_bytes());data[-8]^=1;target.write_bytes(data)
    with self.assertRaises(RuntimeError):m.valid_target(row,True)
 def test_freeze_rejects_partial_geometry(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   root=Path(name);m.atomic(root/'results/abo_hssd_v1/geometry_full/status.json',{'status':'running','completed':1,'failed':0,'total':2})
   with patch.object(m,'ROOT',root):
    with self.assertRaises(RuntimeError):m.freeze()
 def test_corrupt_frame_excluded_before_selection(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   root=Path(name);folder=root/'ABO/renders_cond/abc';folder.mkdir(parents=True)
   Image.new('RGBA',(1024,1024),(10,20,30,255)).save(folder/'000.png')
   (folder/'001.png').write_bytes(b'corrupt')
   good=dict(frame(15),file_path='000.png');bad=dict(frame(-15),file_path='001.png')
   m.atomic(folder/'transforms.json',{'scale':1.,'offset':[0,0,0],'aabb':[[-.5]*3,[.5]*3],'frames':[bad,good]})
   row={'source':'ABO','uid':'abo_abc','raw_sha256':'abc','asset_id':'3/a.glb','trellis_index':{'image_path':str(folder/'000.png'),'image_bytes':(folder/'000.png').stat().st_size,'image_sha256':m.sha(folder/'000.png')}}
   with patch.object(m,'SOURCE',root):r=m.scan_one(row,{('ABO','abc'):{'sha256':'abc','file_identifier':'3/a.glb'}})
   self.assertEqual(r['status'],'selected');self.assertEqual(r['selected']['filename'],'000.png');self.assertEqual(len(r['invalid_frames']),1)
 def test_relay_manifest_launcher_compiles(self):
  import render_images_relay as relay
  commands=[]
  def fake_call(code,*args,**kwargs):
   compile(code,'manifest_launch_dry_run','exec');commands.append(code);return b'{"dry_run":true}'
  with patch.object(relay,'call',fake_call):self.assertTrue(relay.manifest_job()['dry_run'])
  self.assertEqual(len(commands),1)
 def test_source_command_uses_compressed_inline_code(self):
  import render_images_relay as relay
  source=Path(m.__file__).read_bytes();cmd=relay.source_command(source,'scan_source')
  self.assertLess(len(cmd),20000);self.assertIn('zlib.decompress',cmd)
 def test_split_aliases_and_missing_image_do_not_filter_geometry(self):
  import core,pipeline_common
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   root=Path(name);assets=root/'assets/abo_hssd_v1';audit=root/'audit';audit.mkdir()
   candidates=[{'uid':'abo_abc','source':'ABO','raw_sha256':'abc','asset_id':'3/a.glb'},{'uid':'hssd_def','source':'HSSD','raw_sha256':'def','asset_id':'objects/b.glb'}]
   m.jsonlines(assets/'metadata/candidates.jsonl',candidates)
   m.atomic(root/'results/abo_hssd_v1/geometry_full/status.json',{'status':'complete','completed':2,'failed':0,'total':2})
   m.jsonlines(root/'results/abo_hssd_v1/geometry_full/events.jsonl',[{'uid':r['uid']} for r in candidates])
   m.jsonlines(root/'results/abo_hssd_v1/geometry_full/failures.jsonl',[])
   m.atomic(assets/'manifests/abo_hssd_v1/ready.json',{'unchanged_split':True})
   before=(assets/'manifests/abo_hssd_v1/ready.json').read_bytes()
   view={**candidates[0],'status':'selected','selected':{'bytes':12},'transforms':{'bytes':8}}
   m.atomic(audit/'source_selection_inventory.json',{'objects':[view,{'uid':'hssd_def','status':'missing','reason':'not_in_trellis_image_index'}]})
   splits={'train':[{'uid':'abo_abc','aliases':[{'uid':r['uid']} for r in candidates]}],'val':[],'test':[]}
   with patch.multiple(m,ROOT=root,ASSETS=assets,AUDIT=audit),patch.object(core,'cfg',return_value={}),patch.object(pipeline_common,'manifests',return_value=(splits,{})),patch.object(pipeline_common,'geometry_row',return_value={'preprocessing_receipt':'receipt','preprocessing_receipt_sha256':'sha'}),contextlib.redirect_stdout(io.StringIO()):
    m.freeze()
   frozen=m.read(audit/'frozen_selection.json')
   self.assertEqual(frozen['objects'][0]['split'],'train');self.assertEqual(frozen['objects'][0]['representative_uid'],'abo_abc')
   self.assertEqual(frozen['missing_images'][0]['split'],'train');self.assertEqual(len(frozen['geometry_rejected']),0)
   self.assertEqual(before,(assets/'manifests/abo_hssd_v1/ready.json').read_bytes())
 def test_windows_ssh_launches_hide_console(self):
  import ast,types
  import render_images_relay as relay
  class Startup:
   dwFlags=0
   wShowWindow=99
  with patch.object(relay.os,'name','nt'),patch.object(relay.subprocess,'STARTUPINFO',Startup,create=True),patch.object(relay.subprocess,'STARTF_USESHOWWINDOW',1,create=True),patch.object(relay.subprocess,'CREATE_NO_WINDOW',0x08000000,create=True):
   opts=relay.hidden_process_options()
   self.assertEqual(opts['creationflags'],0x08000000)
   self.assertEqual(opts['startupinfo'].dwFlags&1,1)
   self.assertEqual(opts['startupinfo'].wShowWindow,0)
  with patch.object(relay.os,'name','posix'):self.assertEqual(relay.hidden_process_options(),{})
  tree=ast.parse(Path(relay.__file__).read_text());launches=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess' and node.func.attr in ['run','Popen']:
    launches.append(node)
    self.assertTrue(any(k.arg is None and isinstance(k.value,ast.Call) and isinstance(k.value.func,ast.Name) and k.value.func.id=='hidden_process_options' for k in node.keywords))
  self.assertEqual(len(launches),4)
 def test_invalid_png(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as name:
   file=Path(name)/'bad.png';file.write_bytes(b'bad')
   with self.assertRaises(Exception):m.image_info(file)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Images))
 m.atomic(m.AUDIT/'cpu_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'script_sha256':m.sha(Path(m.__file__)),'gpu_used':False,'utc':m.now()})
 raise SystemExit(0 if result.wasSuccessful() else 1)
