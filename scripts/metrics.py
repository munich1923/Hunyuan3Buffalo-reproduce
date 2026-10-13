import numpy as np,trimesh
from scipy.spatial import cKDTree
def compare(pred,target,seed=20261013,n=16384):
    np.random.seed(seed);p,_=trimesh.sample.sample_surface(pred,n)
    np.random.seed(seed+1);q,_=trimesh.sample.sample_surface(target,n)
    dp=cKDTree(q).query(p)[0];dq=cKDTree(p).query(q)[0]
    report={"cd_squared_sum":float(np.mean(dp**2)+np.mean(dq**2)),"cd_l1_mean":float((np.mean(dp)+np.mean(dq))/2),"sample_points":n}
    for eps,key in [(0.01,"001"),(0.025,"0025"),(0.05,"005")]:
        precision=float((dp<eps).mean());recall=float((dq<eps).mean())
        report["fscore_"+key]=2*precision*recall/max(precision+recall,1e-12)
    return report

def preservation(pred,source,target,seed=20261013,n=16384):
    # Ground-truth-assisted test metric only; never supplied as an inference condition.
    np.random.seed(seed);src,_=trimesh.sample.sample_surface(source,n)
    np.random.seed(seed+1);tar,_=trimesh.sample.sample_surface(target,n)
    np.random.seed(seed+2);out,_=trimesh.sample.sample_surface(pred,n)
    unchanged=cKDTree(tar).query(src)[0]<0.025
    if unchanged.sum()<64:return {"available":False,"unchanged_points":int(unchanged.sum())}
    d=cKDTree(out).query(src[unchanged])[0]
    report={"available":True,"unchanged_points":int(unchanged.sum()),"unchanged_region_distance_mean":float(d.mean()),"unchanged_region_coverage_005":float((d<0.05).mean()),"note":"Ground-truth-assisted surface proxies, not human part masks; regions use 0.025 distance threshold."}
    added=cKDTree(src).query(tar)[0]>0.025
    removed=cKDTree(tar).query(src)[0]>0.025
    report["changed_target_points"]=int(added.sum());report["removed_source_points"]=int(removed.sum())
    if added.sum()>=64:
        da=cKDTree(out).query(tar[added])[0]
        report["changed_target_coverage_005"]=float((da<0.05).mean());report["changed_target_distance_mean"]=float(da.mean())
    if removed.sum()>=64:
        dr=cKDTree(out).query(src[removed])[0]
        report["removed_source_still_present_fraction_005"]=float((dr<0.05).mean())
    return report
