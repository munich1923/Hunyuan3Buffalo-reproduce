import argparse,json,random,sys,time
from pathlib import Path
import numpy as np
import torch,trimesh
from PIL import Image
from remote_runtime import ROOT,configure
from edit_model import Editor,load_dit
from train_editor import load_records,tensors,item
from metrics import compare,preservation
configure();sys.path.insert(0,str(ROOT/"src/Hunyuan3D-2.1/hy3dshape"))
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh,Hunyuan3DDiTFlowMatchingPipeline
def decode(vae,z,scale):
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
        decoded=vae.decode((z/scale).to(torch.float32))
        meshes=vae.latents2mesh(decoded,bounds=1.01,mc_level=0,num_chunks=10000,octree_resolution=256,mc_algo="mc",enable_pbar=False)
    return export_to_trimesh(meshes)[0]
def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["source","vae","native","random","connector","lora","no_source","noop","shuffled"],required=True);p.add_argument("--seconds",type=float,default=3600);a=p.parse_args()
    rows=load_records("test");out=ROOT/"results/evaluation"/a.mode;out.mkdir(parents=True,exist_ok=True)
    random.seed(20261013);torch.manual_seed(20261013)
    editor=None;vae=None;pipeline=None
    if a.mode=="native":
        pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float16)
    elif a.mode!="source":
        vae=ShapeVAE.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float32).eval()
        if a.mode!="vae":
            editor=Editor(load_dit(),lora=a.mode in ["lora","no_source","noop","shuffled"]).cuda().eval()
            if a.mode=="random":editor.load_adapter(ROOT/"results/checkpoints/connector_initial.pt")
            else:
                checkpoint="connector" if a.mode=="connector" else "lora"
                editor.load_adapter(ROOT/"results/checkpoints"/(checkpoint+"_best.pt"))
    start=time.monotonic()
    existing=set()
    if (out/"metrics.jsonl").exists():
        existing={(x["id"],x["seed"]) for x in map(json.loads,(out/"metrics.jsonl").read_text().splitlines())}
    for i,row in enumerate(rows):
        for seed in [20261013,20261014,20261015]:
            if (row["id"],seed) in existing:continue
            if time.monotonic()-start>a.seconds:
                print("Evaluation time cap",flush=True);summarize(out,rows,a.mode);return
            record={"effective_instruction":"Keep the geometry unchanged." if a.mode=="noop" else rows[(i+1)%len(rows)]["instruction"] if a.mode=="shuffled" else row["instruction"],"id":row["id"],"seed":seed,"mode":a.mode,"operation":row["operation"],"instruction":row["instruction"]}
            path=out/(row["id"]+"_"+str(seed)+".glb")
            try:
                if a.mode=="source":pred=trimesh.load(row["source_normalized"],force="mesh")
                elif a.mode=="native":
                    pred=pipeline(image=Image.open(row["source_views"][0]),num_inference_steps=50,guidance_scale=5,generator=torch.Generator(device="cuda").manual_seed(seed),octree_resolution=256)[0]
                else:
                    shape,_=item(row,"test");scale=shape["scale_factor"]
                    source,target,h=tensors(row,"test",a.mode=="noop")
                    if a.mode=="vae":z=source
                    else:
                        if a.mode=="shuffled":
                            _,vlm=item(row,"test")
                            h=vlm["shuffled_hidden"][None].cuda().float()
                        g=torch.Generator(device="cuda").manual_seed(seed)
                        z=torch.randn(source.shape,device="cuda",dtype=torch.bfloat16,generator=g)
                        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                            for step in range(50):
                                t=torch.full((1,),step/50,device="cuda",dtype=torch.bfloat16)
                                z=z+editor(z,t,h,None if a.mode=="no_source" else source,checkpoint_blocks=False)/50
                    pred=decode(vae,z,scale)
                if pred is None or len(pred.faces)==0:raise ValueError("empty generated mesh")
                if not np.isfinite(pred.vertices).all():raise ValueError("nonfinite mesh")
                pred.export(path)
                target=trimesh.load(row["target_normalized"],force="mesh");source_mesh=trimesh.load(row["source_normalized"],force="mesh")
                record.update(valid=True,path=str(path),target_metrics=compare(pred,target,seed),source_metrics=compare(pred,source_mesh,seed),preservation_metrics=preservation(pred,source_mesh,target,seed))
                if seed==20261013:
                    from render_views import render
                    try:render(pred,out/(row["id"]+"_views"))
                    except Exception as render_error:
                        record["render_error"]=repr(render_error);print("RENDER_FAIL",row["id"],repr(render_error),flush=True)
            except Exception as e:
                record.update(valid=False,error=repr(e));print("FAIL",record,flush=True)
                if isinstance(e,torch.cuda.OutOfMemoryError):raise
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True)
    summarize(out,rows,a.mode)
def summarize(out,rows,mode):
    results=[json.loads(l) for l in (out/"metrics.jsonl").read_text().splitlines()] if (out/"metrics.jsonl").exists() else []
    summary={"mode":mode,"attempted":len(results),"expected":len(rows)*3,"complete":len(results)==len(rows)*3,"valid_rate":sum(x["valid"] for x in results)/max(1,len(results))}
    metric_key="source_metrics" if mode=="noop" else "target_metrics"
    summary["reference"]="source" if mode=="noop" else "target"
    for op in ["add","remove","all"]:
        valid=[x for x in results if x["valid"] and (op=="all" or x["operation"]==op)]
        if valid:summary[op]={key:float(np.mean([x[metric_key][key] for x in valid])) for key in ["cd_squared_sum","fscore_001","fscore_0025","fscore_005"]}
    summary["surface_proxy"]={}
    for op in ["add","remove","all"]:
        valid=[x for x in results if x["valid"] and (op=="all" or x["operation"]==op)]
        proxy={}
        for key in ["unchanged_region_coverage_005","unchanged_region_distance_mean","changed_target_coverage_005","changed_target_distance_mean","removed_source_still_present_fraction_005"]:
            vals=[x["preservation_metrics"][key] for x in valid if key in x["preservation_metrics"]]
            if vals:proxy[key]={"mean":float(np.mean(vals)),"available":len(vals)}
        summary["surface_proxy"][op]=proxy
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":main()
