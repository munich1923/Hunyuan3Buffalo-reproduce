"""Independent topology, numeric, and coordinate checks for the pinned geometry pipeline."""
import json,hashlib
from pathlib import Path
import numpy as np
import igl,trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def topology(vertices,faces,require_closed=True):
 v=np.asarray(vertices);f=np.asarray(faces,dtype=np.int64)
 if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all() or np.ptp(v,axis=0).max()<=1e-8:raise ValueError('invalid/nonfinite/degenerate vertices')
 if f.ndim!=2 or f.shape[1]!=3 or len(f)<4 or f.min()<0 or f.max()>=len(v):raise ValueError('invalid triangle indices')
 if np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,2]==f[:,0])):raise ValueError('repeated index triangle')
 e=f[:,[0,1,1,2,2,0]].reshape(-1,2);corner=(np.arange(len(f)*3).reshape(-1,3)[:,[0,1,1,2,2,0]]).reshape(-1,2)
 swap=e[:,0]>e[:,1];e[swap]=e[swap,::-1];corner[swap]=corner[swap,::-1]
 order=np.lexsort((e[:,1],e[:,0]));e=e[order];corner=corner[order];start=np.r_[True,np.any(e[1:]!=e[:-1],axis=1)];idx=np.flatnonzero(start);counts=np.diff(np.r_[idx,len(e)])
 boundaries=int(np.sum(counts==1));edge_ok=bool(np.all(counts<=2));vertex_ok=False
 if edge_ok and boundaries==0:
  assert np.all(counts==2)
  pairs=corner.reshape(-1,2,2);left=pairs[:,0,:].ravel();right=pairs[:,1,:].ravel()
  graph=coo_matrix((np.ones(len(left),dtype=np.uint8),(left,right)),shape=(f.size,f.size)).tocsr()
  _,labels=connected_components(graph,directed=False)
  vv=f.ravel();oo=np.argsort(vv,kind='stable');vs=vv[oo];ls=labels[oo];vertex_ok=bool(np.all((vs[1:]!=vs[:-1])|(ls[1:]==ls[:-1])))
 m=trimesh.Trimesh(vertices=v,faces=f,process=False)
 out={'vertices':len(v),'faces':len(f),'boundary_edges':boundaries,'edge_manifold':edge_ok,'vertex_manifold':vertex_ok,'trimesh_watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'volume':float(m.volume),'bounds':m.bounds.tolist()}
 if require_closed and not(boundaries==0 and edge_ok and vertex_ok and m.is_watertight and m.is_winding_consistent):raise ValueError('watertight topology gate failed: '+json.dumps(out))
 return out

def load_mesh(file,require_closed=False):
 v,f=igl.read_triangle_mesh(str(file));return np.asarray(v,dtype=np.float64),np.asarray(f,dtype=np.int64),topology(v,f,require_closed)

def validate(folder,uid,points_seed):
 folder=Path(folder);prefix=folder/uid;v,f,info=load_mesh(str(prefix)+'_watertight.obj',True)
 with np.load(str(prefix)+'_surface.npz',allow_pickle=False) as z:
  if set(z.files)!={'random_surface','sharp_surface'}:raise ValueError('surface keys')
  a=z['random_surface'];b=z['sharp_surface']
 for name,arr in [('random_surface',a),('sharp_surface',b)]:
  if arr.shape!=(124928,6) or arr.dtype!=np.float16 or not np.isfinite(arr).all():raise ValueError('surface contract '+name)
 with np.load(str(prefix)+'_sdf.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 expected={'vol_points':(249856,3),'vol_label':(249856,),'random_near_points':(249856,3),'random_near_label':(249856,),'sharp_near_points':(124928,3),'sharp_near_label':(124928,)}
 if set(d)!=set(expected):raise ValueError('SDF keys')
 for name,shape in expected.items():
  if d[name].shape!=shape or d[name].dtype!=np.float16 or not np.isfinite(d[name]).all():raise ValueError('SDF contract '+name)
 center=(v.min(0)+v.max(0))/2;extent=float(np.ptp(v,axis=0).max());scale=1.9998/extent;v2=(v-center)*scale
 rng=np.random.default_rng(points_seed);indices=rng.choice(len(a),81920,replace=False);p=a[indices].astype(np.float32)
 p[:,:3]=((p[:,:3].astype(np.float64)-center)*scale).astype(np.float32);n=p[:,3:6];length=np.linalg.norm(n,axis=1,keepdims=True)
 if np.any(length<=1e-8):raise ValueError('zero normal')
 n/=length;points=np.concatenate([p[:,:3],n,np.zeros((81920,1),np.float32)],axis=1)
 if points.shape!=(81920,7) or not np.isfinite(points).all() or not np.allclose(np.linalg.norm(points[:,3:6],axis=1),1,atol=1e-5):raise ValueError('VAE point input')
 trimesh.Trimesh(vertices=v2,faces=f,process=False).export(str(prefix)+'_vae.obj');np.save(str(prefix)+'_vae_points.npy',points)
 transformed=a.astype(np.float32);transformed[:,:3]=((a[:,:3].astype(np.float64)-center)*scale).astype(np.float32);np.save(str(prefix)+'_vae_random_surface.npy',transformed)
 (folder/(uid+'_coordinate_transform.json')).write_text(json.dumps({'center':center.tolist(),'scale':scale,'longest_edge_before':extent,'longest_edge_after':float(np.ptp(v2,axis=0).max()),'vae_extent':1.9998,'points_seed':points_seed,'input_source':'upstream random_surface only; without replacement; no raw-mesh resampling'},indent=2))
 info.update({'surface_shape':list(a.shape),'sdf_shapes':{k:list(x.shape) for k,x in d.items()},'vae_points_shape':list(points.shape),'points_array_sha256':hashlib.sha256(points.tobytes()).hexdigest(),'surface_array_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'watertight_sha256':hashlib.sha256(Path(str(prefix)+'_watertight.obj').read_bytes()).hexdigest()})
 return info

if __name__=='__main__':
 import sys
 print(json.dumps(validate(sys.argv[1],sys.argv[2],int(sys.argv[3]))))
