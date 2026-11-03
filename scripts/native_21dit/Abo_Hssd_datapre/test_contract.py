"""CPU contract tests and pinned-upstream sampling/sign checks."""
import copy,hashlib,json,os,random,runpy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import trimesh,igl
from core import *
from quality import topology
from prepare_geometry import stage_commit,stage_valid,verify_files

class Contracts(unittest.TestCase):
 def test_caption_parse_and_selection(self):
  caps,indices=parse_captions('["", "chair", "chair", " table "]')
  self.assertEqual(indices,[1,2,3]);row={'uid':'x','captions':caps,'nonempty_caption_indices':indices}
  old=random.getstate();a=[choose_caption(row,42,2,i) for i in range(200)]
  self.assertEqual(old,random.getstate());self.assertEqual(a,[choose_caption(row,42,2,i) for i in range(200)])
  self.assertEqual(choose_caption(row,42,9,17,True)['caption_index'],1)
  self.assertEqual({x['caption_index'] for x in a},{1,2,3})
  for text in ['[]','[null]','[" "]','bad']:
   with self.assertRaises(Exception):parse_captions(text)
 def test_uid_sha(self):
  c=cfg();assets,_=dirs(c)
  for r in readlines(assets/'metadata/candidates.jsonl'):
   self.assertEqual(r['uid'],r['source'].lower()+'_'+r['raw_sha256']);self.assertEqual(r['caption_hash'],fp(r['captions']))
   self.assertEqual(r['nonempty_caption_indices'],[i for i,t in enumerate(r['captions']) if t.strip()])
 def test_cross_source_dedup_and_splits(self):
  rows=[{'uid':str(i),'source':s,'raw_sha256':str(i),'geometry_hash':str(i),'captions':['a','a'],'nonempty_caption_indices':[0,1]} for i,s in enumerate(['ABO']*22+['HSSD']*31)]
  rows[-1]['geometry_hash']=rows[0]['geometry_hash'];rows[-2]['raw_sha256']=rows[1]['raw_sha256']
  unique,audit=dedup(rows);self.assertEqual(len(unique),51);self.assertEqual(len(audit),2)
  c=cfg();splits=split_rows(unique,c);self.assertEqual([len(splits[k]) for k in ['train','val','test']],allocate(51,c['split_ratios']))
  seen=set()
  for group in splits.values():
   for r in group:
    aliases={a['uid'] for a in r['aliases']};self.assertFalse(seen&aliases);seen|=aliases
  self.assertEqual(len(seen),53);self.assertEqual(splits,split_rows(unique,c))
  for n in range(100):self.assertEqual(sum(allocate(n,[.9,.05,.05])),n)
 def test_geometry_hash_permutations(self):
  m=trimesh.creation.box();v=m.vertices;f=m.faces;perm=np.random.default_rng(2).permutation(len(v));inverse=np.argsort(perm)
  self.assertEqual(geometry_hash(v,f),geometry_hash(v[perm],inverse[f[::-1,::-1]]))
 def test_topology(self):
  m=trimesh.creation.icosphere(subdivisions=1);self.assertTrue(topology(m.vertices,m.faces)['vertex_manifold'])
  with self.assertRaises(ValueError):topology(m.vertices,m.faces[:-1])
  with self.assertRaises(ValueError):topology(m.vertices,np.vstack([m.faces,m.faces[0]]))
  v=np.vstack([m.vertices,m.vertices+[0,0,2]]);v[len(m.vertices)]=v[0];f=np.vstack([m.faces,m.faces+len(m.vertices)]);f[f==len(m.vertices)]=0
  with self.assertRaises(ValueError):topology(v,f)
 def test_receipt_invalidation_and_interrupt(self):
  root=path('tmp/abo_hssd_tests');root.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=root) as name:
   folder=Path(name);payload=folder/'payload';payload.write_bytes(b'abc');contract={'input':'one','tool':'x'}
   self.assertFalse(stage_valid(folder,'sample',contract));atomic(folder/'.stages/sample.started.json',{'contract':contract})
   self.assertFalse(stage_valid(folder,'sample',contract));stage_commit(folder,'sample',contract,[payload]);self.assertTrue(stage_valid(folder,'sample',contract))
   with self.assertRaises(RuntimeError):stage_valid(folder,'sample',dict(contract,input='two'))
   payload.write_bytes(b'abd')
   with self.assertRaises(RuntimeError):stage_valid(folder,'sample',contract)

def sampling_check():
 c=cfg();source=path('src/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample.py');ns=runpy.run_path(str(source),run_name='abo_hssd_sampling_test')
 m=trimesh.creation.box();rng_original=np.random.default_rng
 from scipy.stats import truncnorm
 rvs_original=truncnorm.rvs
 def wrapped(seedval):
  random.seed(seedval);np.random.seed(seedval)
  np.random.default_rng=lambda seed=None:rng_original(seedval if seed is None else seed)
  truncnorm.rvs=lambda *a,**k:rvs_original(*a,random_state=np.random,**k)
 try:
  wrapped(12345);a,b=ns['SampleMesh'](m.vertices,m.faces)
  wrapped(12345);x,y=ns['SampleMesh'](m.vertices,m.faces)
  assert all(np.array_equal(a[k],x[k]) for k in a)
  assert all(np.array_equal(b[k],y[k]) for k in b)
 finally:np.random.default_rng=rng_original;truncnorm.rvs=rvs_original
 points=np.array([[0.,0.,0.],[2.,0.,0.]],np.float32)
 mode='FAST_WINDING_NUMBER'
 try:d,*_=igl.signed_distance(points,m.vertices,m.faces,return_normals=False,sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
 except Exception:
  mode='DEFAULT_FALLBACK';d,*_=igl.signed_distance(points,m.vertices,m.faces,return_normals=False)
 labels=-d;assert labels[0]>0 and labels[1]<0
 # Actual upstream sample_sdf records positive inside, negative outside. No label flip is applied.
 inside=np.max(np.abs(b['vol_points'].astype(np.float32)),axis=1)<.4;outside=np.max(np.abs(b['vol_points'].astype(np.float32)),axis=1)>.6
 assert np.all(b['vol_label'][inside]>0) and np.all(b['vol_label'][outside]<0)
 return {'upstream_sha256':sha(source),'sampling_arrays_equal':True,'surface_shapes':{k:list(v.shape) for k,v in a.items()},'surface_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()},'sdf_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in b.items()},'known_geometry':'unit cube centered at origin','probe_points':points.tolist(),'signed_distance':d.tolist(),'upstream_labels':labels.tolist(),'inside_label':'positive','outside_label':'negative','label_flip_applied':False,'sign_api':mode,'volume_inside_count':int(inside.sum()),'volume_outside_count':int(outside.sum())}

if __name__=='__main__':
 setup();test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
 if not test.wasSuccessful():sys.exit(1)
 report={'cpu_tests':test.testsRun,'cpu_passed':True,'upstream_sampling_and_sdf':sampling_check(),'data_ready':False,**now()}
 atomic('results/abo_hssd_v1/preparation/cpu_checks.json',report);print(json.dumps(report),flush=True)
