"""Neutral geometry renders using headless EGL and one shared coordinate frame."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import argparse,json,math
from pathlib import Path
import numpy as np
import pyrender
from PIL import Image,ImageDraw
from prepare_dataset import mesh,normalize_pair
from remote_runtime import ROOT,configure
configure()
RENDER_VERSION="neutral_geometry_v2"
def pose(azimuth,elevation=20,radius=3.3):
    a=math.radians(azimuth);e=math.radians(elevation)
    eye=radius*np.array([math.cos(e)*math.sin(a),math.sin(e),math.cos(e)*math.cos(a)])
    z=eye/np.linalg.norm(eye);x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    p=np.eye(4);p[:3,:3]=np.stack([x,y,z],axis=1);p[:3,3]=eye;return p
def render(m,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    m=m.copy()
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.6,0.6,0.6,1],metallicFactor=0,roughnessFactor=0.9)
    s=pyrender.Scene(bg_color=[255,255,255,0],ambient_light=[0.15]*3)
    s.add(pyrender.Mesh.from_trimesh(m,material=material,smooth=False))
    camera=s.add(pyrender.PerspectiveCamera(yfov=math.radians(40)),pose=pose(0))
    light=s.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.3),pose=pose(0))
    r=pyrender.OffscreenRenderer(512,512)
    paths=[]
    try:
        for az in [0,90,180,270]:
            p=pose(az);s.set_pose(camera,p);s.set_pose(light,p)
            color,depth=r.render(s)
            for retry in range(2):
                if np.isfinite(depth).all() and not (color[:,:,:3].mean()<10 and (depth>0).mean()>0.99):break
                color,depth=r.render(s)
            if not np.isfinite(depth).all() or (color[:,:,:3].mean()<10 and (depth>0).mean()>0.99):raise RuntimeError("Invalid all-black/nonfinite EGL frame")
            color=color.copy();color[depth==0,:3]=255;path=folder/(str(az)+".png");Image.fromarray(np.concatenate([color[:,:,:3],((depth>0)*255).astype(np.uint8)[:,:,None]],axis=2),"RGBA").save(path);paths.append(str(path))
    finally:r.delete()
    return paths
def main():
    p=argparse.ArgumentParser();p.add_argument("--split",default="all");p.add_argument("--limit",type=int);a=p.parse_args()
    splits=["train","val","test"] if a.split=="all" else [a.split]
    for split in splits:
        rows=[json.loads(l) for l in (ROOT/"assets/manifests"/(split+".jsonl")).read_text().splitlines()]
        if a.limit:rows=rows[:a.limit]
        outputs=[]
        for i,row in enumerate(rows):
            folder=ROOT/"assets/prepared"/split/row["id"]
            cached=folder/"record.json"
            if cached.exists():
                record=json.loads(cached.read_text())
                required=[record["source_normalized"],record["target_normalized"],*record["source_views"]]
                if record.get("render_version")==RENDER_VERSION and record.get("instruction")==row["instruction"] and all(Path(p).exists() for p in required):
                    outputs.append(record);continue
            src,tar,center,scale=normalize_pair(mesh(row["source"]),mesh(row["target"]))
            folder.mkdir(parents=True,exist_ok=True)
            src.export(folder/"source.glb");tar.export(folder/"target.glb")
            paths=render(src,folder/"source_views")
            target_paths=render(tar,folder/"target_views") if i<60 or split=="test" else []
            record={**row,"render_version":RENDER_VERSION,"source_normalized":str(folder/"source.glb"),"target_normalized":str(folder/"target.glb"),"source_views":paths,"target_views":target_paths,"normalization_center":center.tolist(),"normalization_scale":scale}
            (folder/"record.json").write_text(json.dumps(record,indent=2)+"\n");outputs.append(record)
            if target_paths and i<60:
                canvas=Image.new("RGB",(2048,1060),"white");draw=ImageDraw.Draw(canvas);draw.text((8,5),row["instruction"],fill="black")
                for j,path in enumerate(paths):canvas.paste(Image.open(path),(j*512,36))
                for j,path in enumerate(target_paths):canvas.paste(Image.open(path),(j*512,548))
                review=ROOT/"results/data_review"/split;review.mkdir(parents=True,exist_ok=True);canvas.save(review/(row["id"]+".png"))
            if i%25==0:print("RENDER",split,i,flush=True)
        (ROOT/"assets/manifests"/(split+"_prepared.jsonl")).write_text("".join(json.dumps(x)+"\n" for x in outputs))
if __name__=="__main__":main()
