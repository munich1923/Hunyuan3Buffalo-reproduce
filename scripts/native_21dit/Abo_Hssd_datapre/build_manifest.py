"""Atomic immutable manifests after all geometry attempts; incremental CPU audit."""
import json,random,os,sys,time,traceback
import numpy as np
import trimesh,igl
from scipy.spatial import cKDTree
from core import *
from pipeline_common import geometry_row,manifest_contract,manifests

def near_hash(v,f):return geometry_hash(v,f)
def audit_row(row,c):
 assets,results=dirs(c);r=geometry_row(row,c)
 if r is None:return None
 cache=results/'geometry_audit'/ (r['uid']+'.json')
 contract={'preprocessing_receipt_sha256':r['preprocessing_receipt_sha256'],'audit_version':3,'core_sha256':sha(Path(__file__).with_name('core.py'))}
 if cache.exists():
  saved=read(cache)
  if saved['contract']!=contract:raise RuntimeError('audit contract changed '+r['uid'])
  return saved['row']
 # Original scene transforms are represented by the pinned Blender PLY.
 v,f=igl.read_triangle_mesh(str(path(r['converted_mesh'])));v=np.asarray(v);f=np.asarray(f,dtype=np.int64)
 if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(v):raise ValueError('invalid converted mesh')
 extent=float(np.ptp(v,axis=0).max())
 if extent<=1e-8:raise ValueError('degenerate converted mesh')
 r['geometry_hash']=geometry_hash(v,f)
 # Reproduce upstream normalize_to_unit_box then apply the water mesh's VAE transform.
 u=(v-v.min(0))/(extent*1.01);tr=r['coordinate_transform'];vn=(u-np.asarray(tr['center']))*tr['scale']
 original=trimesh.Trimesh(vertices=vn,faces=f,process=False)
 normals=original.face_normals
 if not np.isfinite(normals).all() or original.area<=1e-10:raise ValueError('nonfinite/degenerate original geometry')
 points,_=trimesh.sample.sample_surface(original,16384,seed=seed(c['seed'],'change',r['uid']))
 water=np.load(path(r['random_surface']),allow_pickle=False)[:,:3]
 ix=np.random.default_rng(seed(c['seed'],'change-water',r['uid'])).choice(len(water),16384,replace=False);wp=water[ix]
 d1=cKDTree(wp).query(points)[0];d2=cKDTree(points).query(wp)[0]
 r['geometry_change']={'coordinate_frame':'upstream unit-box then same water-to-VAE transform; no independent scale/ICP','points':16384,'original_vertices':len(v),'original_faces':len(f),'original_watertight':bool(original.is_watertight),'original_winding_consistent':bool(original.is_winding_consistent),'water_vertices':r['quality']['vertices'],'water_faces':r['quality']['faces'],'distance_original_to_water_mean':float(d1.mean()),'distance_water_to_original_mean':float(d2.mean()),'distance_original_to_water_p95':float(np.quantile(d1,.95)),'distance_water_to_original_p95':float(np.quantile(d2,.95)),'detail_preservation_not_guaranteed':True}
 r['normalization']={'center':[0.,0.,0.],'scale':1.,'extent':1.9998,'already_vae_coordinates':True}
 atomic(cache,{'contract':contract,'row':r,**now()});return r

def main():
 c=cfg();check_upstream(c);a,o=dirs(c);root=a/'manifests/abo_hssd_v1'
 if (root/'ready.json').exists():
  splits,ready=manifests(c);print(json.dumps({'reused':True,'counts':{k:len(v) for k,v in splits.items()}}));return
 status=read(o/'geometry_full/status.json')
 if status['status'] not in ['complete','complete_with_failures'] or status['completed']+status['failed']!=status['total']:raise RuntimeError('full geometry not complete; cannot freeze partial dataset')
 rows=readlines(a/'metadata/candidates.jsonl');processed=[];rejected=[]
 for i,row in enumerate(rows):
  r=audit_row(row,c)
  if r is None:rejected.append({'uid':row['uid'],'source':row['source'],'reason':'geometry_processing_or_validation_failed'})
  else:processed.append(r)
  atomic(o/'preparation/index_progress.json',{'scanned':i+1,'total':len(rows),'accepted':len(processed),'rejected':len(rejected),'data_ready':False,**now()})
 unique,audit=dedup(processed);splits=split_rows(unique,c)
 tmp=root.with_name('.abo_hssd_v1.tmp');tmp.mkdir(parents=True,exist_ok=True)
 allcaps={};alias_splits={}
 for sp,group in splits.items():
  lines(tmp/(sp+'.jsonl'),group);cap_index=[]
  for r in group:
   for alias in r['aliases']:
    if alias['uid'] in alias_splits:raise RuntimeError('duplicate alias crossed split')
    alias_splits[alias['uid']]=sp
   for i,t in enumerate(r['captions']):
    if not t.strip():continue
    h=hashlib.sha256(t.encode()).hexdigest()
    if h in allcaps and allcaps[h]!=t:raise RuntimeError('SHA collision')
    allcaps[h]=t;cap_index.append({'uid':r['uid'],'caption_index':i,'caption_sha256':h,'split':sp,'evaluation_caption':i==r['nonempty_caption_indices'][0]})
  lines(tmp/(sp+'_captions.jsonl'),cap_index)
 # Near matches are audit only: rounded shape-size/area descriptors, never merged.
 buckets={}
 for r in processed:
  change=r['geometry_change'];key=fp({'bounds':np.round(np.diff(np.asarray(r['quality']['bounds']),axis=0)[0]/max(np.diff(np.asarray(r['quality']['bounds']),axis=0)[0]),2).tolist(),'faces_log_bucket':int(np.log2(max(change['original_faces'],1)))})
  buckets.setdefault(key,[]).append(r['uid'])
 lines(tmp/'near_duplicate_audit.jsonl',[{'descriptor':k,'uids':v,'candidate_only_not_merged':True} for k,v in sorted(buckets.items()) if len(v)>1])
 lines(tmp/'unique_captions.jsonl',[{'caption_sha256':h,'caption':t} for h,t in sorted(allcaps.items())]);lines(tmp/'dedup_audit.jsonl',audit);lines(tmp/'rejected_geometry.jsonl',rejected)
 summary={'candidates':len(rows),'accepted_before_dedup':len(processed),'unique':len(unique),'duplicates_removed':len(processed)-len(unique),'rejected':len(rejected),'splits':{sp:{'total':len(g),'sources':{s:sum(r['source']==s for r in g) for s in c['sources']}} for sp,g in splits.items()},'unique_captions':len(allcaps),'caption_occurrences':sum(len(r['nonempty_caption_indices']) for r in unique),'data_ready':False,**now()}
 atomic(tmp/'summary.json',summary)
 files={f.name:sha(f) for f in sorted(tmp.glob('*')) if f.is_file() and f.name!='ready.json'}
 atomic(tmp/'ready.json',{'config_contract':manifest_contract(c),'files':files,'metadata_sha256':sha(a/'metadata/candidates.jsonl'),'data_ready':False})
 if root.exists():raise RuntimeError('destination already exists without ready receipt; manual inspection required')
 os.rename(tmp,root);print(json.dumps(summary),flush=True)
if __name__=='__main__':setup();main()
