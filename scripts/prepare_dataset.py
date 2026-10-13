"""Extract public meshes safely and create leakage-resistant editing manifests."""
import argparse,csv,hashlib,json,random,re,tarfile,concurrent.futures
from collections import defaultdict
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from remote_runtime import ROOT,configure
configure()
SEED=20261013
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def mesh(path):
    obj=trimesh.load(path,process=False)
    if isinstance(obj,trimesh.Scene):
        parts=[]
        for node in obj.graph.nodes_geometry:
            transform,geo=obj.graph[node]
            part=obj.geometry[geo].copy();part.apply_transform(transform)
            if isinstance(part,trimesh.Trimesh):parts.append(part)
        if not parts:raise ValueError("no triangle meshes")
        obj=trimesh.util.concatenate(parts)
    if not isinstance(obj,trimesh.Trimesh) or len(obj.vertices)<4 or len(obj.faces)<4:raise ValueError("empty or tiny mesh")
    if not np.isfinite(obj.vertices).all():raise ValueError("nonfinite vertices")
    if obj.faces.min()<0 or obj.faces.max()>=len(obj.vertices):raise ValueError("invalid face indices")
    if not np.isfinite(obj.area) or obj.area<=1e-10:raise ValueError("zero area")
    return obj
def geometry_hash(m):
    # Order-independent vertex-set fingerprint; conservative identity grouping.
    extent=np.ptp(m.vertices,axis=0).max()
    v=np.unique(np.round((m.vertices-m.bounds.mean(axis=0))/extent,5),axis=0).astype("<f8")
    return hashlib.sha256(v.tobytes()).hexdigest()
def normalize_pair(src,tar):
    # One source-derived transform, NEVER separately fit the target.
    center=src.bounds.mean(axis=0);extent=np.ptp(src.vertices,axis=0).max()
    if extent<=1e-8:raise ValueError("degenerate source bounds")
    scale=1.5/extent
    a=src.copy();b=tar.copy()
    a.vertices=(a.vertices-center)*scale;b.vertices=(b.vertices-center)*scale
    return a,b,center,scale
def paired_qc(src,tar,seed):
    a,b,center,scale=normalize_pair(src,tar)
    rng=np.random.RandomState(seed)
    # Deterministic area-weighted surface samples through trimesh's RNG.
    np.random.seed(seed);pa,_=trimesh.sample.sample_surface(a,4096)
    np.random.seed(seed+1);pb,_=trimesh.sample.sample_surface(b,4096)
    da=cKDTree(pb).query(pa)[0];db=cKDTree(pa).query(pb)[0]
    report={"source_vertices":len(a.vertices),"target_vertices":len(b.vertices),"source_faces":len(a.faces),"target_faces":len(b.faces),"target_extent_ratio":float(np.ptp(b.vertices,axis=0).max()/1.5),"target_center_shift":float(np.linalg.norm(b.bounds.mean(axis=0))),"source_unchanged_fraction_005":float((da<0.05).mean()),"target_unchanged_fraction_005":float((db<0.05).mean()),"sample_cd_squared":float((da**2).mean()+(db**2).mean()),"shared_center":center.tolist(),"shared_scale":float(scale)}
    # Broad anomaly gates only. Preserve all measurements for later review.
    if np.max(np.abs(b.vertices))>1.0:raise ValueError("target exceeds fixed VAE extraction bounds")
    if report["target_extent_ratio"]>4 or report["target_extent_ratio"]<0.2:raise ValueError("extreme scale change")
    if report["target_center_shift"]>2:raise ValueError("gross coordinate mismatch")
    if min(da.min(),db.min())>0.25:raise ValueError("disjoint geometry")
    if max(report["source_unchanged_fraction_005"],report["target_unchanged_fraction_005"])<0.1:raise ValueError("no substantial preserved surface")
    return report
def extract():
    base=ROOT/"assets/Nano3D-Edit-100k"
    dest=base/"extracted";dest.mkdir(parents=True,exist_ok=True)
    count=0
    for archive in sorted((base/"v1_100k/editing_assets").glob("*.tar.gz")):
        print("EXTRACT",archive.name,flush=True)
        with tarfile.open(archive,"r:gz") as tf:
            for member in tf:
                if not member.isfile():continue
                name=Path(member.name)
                if name.name not in ["src_mesh.glb","tar_mesh.glb","source.png","edit.png"]:continue
                uid=next((p for p in name.parts if re.fullmatch(r"uid_\d+",p)),None)
                if uid is None or member.size>512*1024*1024:continue
                out=dest/uid/name.name;out.parent.mkdir(parents=True,exist_ok=True)
                if out.exists() and out.stat().st_size==member.size:continue
                source=tf.extractfile(member)
                with out.open("wb") as f:
                    for block in iter(lambda:source.read(1024*1024),b""):f.write(block)
                count+=1
    print("Extracted files",count,flush=True)
    return dest
def prepare_bench():
    from download_artifacts import api,download
    repo="ziqima/Edit3D-Bench";meta=api(repo,"datasets")
    lookup={x["rfilename"]:x for x in meta["siblings"]}
    base=ROOT/"assets/Edit3D-Bench";records=[];rejects=[]
    rng=random.Random(SEED)
    for op in ["add","remove"]:
        rows=list(csv.DictReader((base/"metadata"/(op+".csv")).open()))
        rng.shuffle(rows)
        kept=0
        for r in rows:
            item={"id":"bench_"+op+"_"+r["source"],"operation":op,"instruction":r["instruction"],"source_id":r["source"],"target_id":r["target"],"dataset":"Edit3D-Bench"}
            tasks=[]
            for role in ["source","target"]:
                uid=r[role]
                candidates=[p for p in lookup if p.endswith("/"+uid+".glb")]
                candidates.sort(key=lambda p:(("/"+op+"/") not in p,p))
                if not candidates:raise ValueError("Benchmark mesh missing: "+uid)
                path=candidates[0]
                tasks.append((repo,"datasets",meta["sha"],lookup[path],base))
                item[role]=str(base/path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                if not all(ex.map(lambda args:download(*args),tasks)):raise RuntimeError("Benchmark download failed; durable receipts retained")
            try:
                a,b=mesh(item["source"]),mesh(item["target"])
                if geometry_hash(a)==geometry_hash(b):raise ValueError("identical normalized geometry")
                item["qc"]=paired_qc(a,b,SEED+len(records))
            except Exception as e:
                rejects.append({"id":item["id"],"reason":repr(e)});continue
            records.append(item);kept+=1
            if kept==20:break
        if kept<20:raise RuntimeError("Insufficient valid benchmark "+op+" entries: "+str(kept))
    out=ROOT/"assets/manifests";out.mkdir(parents=True,exist_ok=True)
    (out/"bench_rejected.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rejects))
    return records
def main():
    out=ROOT/"assets/manifests";out.mkdir(parents=True,exist_ok=True)
    bench=prepare_bench()
    excluded=set()
    for row in bench:
        for role in ["source","target"]:excluded.add(geometry_hash(mesh(row[role])))
    (out/"test.jsonl").write_text("".join(json.dumps(x)+"\n" for x in bench))
    dest=extract()
    info={x["uid"]:x for x in map(json.loads,(ROOT/"assets/Nano3D-Edit-100k/v1_100k/info/v1_100k_info.jsonl").open())}
    candidates=list(dest.glob("uid_*"));random.Random(SEED).shuffle(candidates)
    accepted=[];rejects=[];seen=set()
    for folder in candidates:
        uid=folder.name;row=info.get(uid,{})
        text=row.get("edit_instruction","").strip()
        if not re.match(r"^add\b",text,re.I):rejects.append({"uid":uid,"reason":"not_add_instruction"});continue
        try:
            s=folder/"src_mesh.glb";t=folder/"tar_mesh.glb";a=mesh(s);b=mesh(t)
            hs,ht=geometry_hash(a),geometry_hash(b)
            if hs==ht:raise ValueError("identical geometry")
            if hs in excluded or ht in excluded:raise ValueError("benchmark geometry overlap")
            if hs in seen or ht in seen:raise ValueError("duplicate shared geometry")
            qc=paired_qc(a,b,SEED+int(uid[4:]));inverse_qc=paired_qc(b,a,SEED+int(uid[4:]))
            seen.update([hs,ht])
            accepted.append({"uid":uid,"source":str(s),"target":str(t),"source_geometry_hash":hs,"target_geometry_hash":ht,"source_sha256":digest(s),"target_sha256":digest(t),"instruction":text,"category":row.get("category"),"qc":qc,"inverse_qc":inverse_qc})
        except Exception as e:rejects.append({"uid":uid,"reason":str(e)})
        if len(accepted)%50==0 and accepted:print("ACCEPTED",len(accepted),flush=True)
    report={"candidates":len(candidates),"accepted_original_pairs":len(accepted),"rejected":len(rejects),"required_original_pairs":576,"seed":SEED,"thresholds":{"max_extent_ratio":4,"min_extent_ratio":0.2,"max_center_shift":2,"preserved_surface_distance":0.05,"min_preserved_fraction":0.1},"note":"QC gates detect gross anomalies; semantic correctness requires sampled visual review."}
    (out/"qc_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"accepted_candidates.jsonl").write_text("".join(json.dumps(x)+"\n" for x in accepted))
    (out/"rejected.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rejects))
    if len(accepted)<576:raise RuntimeError("Insufficient independent add pairs: "+str(len(accepted)))
    random.Random(SEED).shuffle(accepted)
    for split,originals in [("train",accepted[:512]),("val",accepted[512:576])]:
        records=[]
        for x in originals:
            rec={**x,"id":x["uid"]+"_add","group":x["uid"],"operation":"add","dataset":"Nano3D-Edit-100k"}
            records.append(rec)
            inverse={**rec,"id":x["uid"]+"_remove","operation":"remove","source":x["target"],"target":x["source"],"instruction":"Remove the following added parts: "+re.sub(r"^add\s+","",x["instruction"],flags=re.I).rstrip(".")+".","inverted_from":rec["id"],"source_geometry_hash":x["target_geometry_hash"],"target_geometry_hash":x["source_geometry_hash"],"source_sha256":x["target_sha256"],"target_sha256":x["source_sha256"],"qc_original_forward":x["qc"],"qc":x["inverse_qc"]}
            # Reversed pair uses its own source-derived common transform later.
            records.append(inverse)
        (out/(split+".jsonl")).write_text("".join(json.dumps(x)+"\n" for x in records))
    sample=accepted[:30]
    (out/"visual_review.jsonl").write_text("".join(json.dumps(x)+"\n" for x in sample))
    print("Prepared 1024 train, 128 validation, 40 test entries",flush=True)
if __name__=="__main__":main()
