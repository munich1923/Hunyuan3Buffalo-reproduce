"""Same50-step sampling on the8 independent add pairs used by the fit diagnostic."""
import argparse,json,time
import numpy as np,torch,trimesh
from remote_runtime import ROOT,configure
from train_editor import load_records,tensors,item
from edit_model import Editor,load_dit
from evaluate_editor import decode
from metrics import compare,preservation
from render_views import render
configure()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",choices=["overfit","lora"],required=True);parser.add_argument("--seconds",type=float,default=1500);parser.add_argument("--skip-render",action="store_true");a=parser.parse_args()
    start=time.monotonic()
    rows=[x for x in load_records("train") if x["operation"]=="add"][:8]
    assert len({x["group"] for x in rows})==8
    out=ROOT/"results/training_geometry"/a.checkpoint;out.mkdir(parents=True,exist_ok=True)
    from hy3dshape.models.autoencoders import ShapeVAE
    vae=ShapeVAE.from_pretrained(str(ROOT/"models/Hunyuan3D-2.1"),dtype=torch.float32).eval()
    editor=Editor(load_dit(),lora=a.checkpoint=="lora").cuda().eval()
    metadata=editor.load_adapter(ROOT/"results/checkpoints"/(a.checkpoint+"_best.pt"))
    results=[]
    for row in rows:
        if time.monotonic()-start>a.seconds:break
        record={"id":row["id"],"instruction":row["instruction"],"seed":20261013,"checkpoint":a.checkpoint,"checkpoint_step":metadata["step"]}
        source_mesh=trimesh.load(row["source_normalized"],force="mesh");target_mesh=trimesh.load(row["target_normalized"],force="mesh")
        record["unchanged_source_target_metrics"]=compare(source_mesh,target_mesh)
        try:
            shape,_=item(row,"train");source,target,h=tensors(row,"train")
            g=torch.Generator(device="cuda").manual_seed(20261013)
            z=torch.randn(source.shape,device="cuda",dtype=torch.bfloat16,generator=g)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                for step in range(50):
                    t=torch.full((1,),step/50,device="cuda",dtype=torch.bfloat16)
                    z=z+editor(z,t,h,source,checkpoint_blocks=False)/50
            pred=decode(vae,z,shape["scale_factor"])
            if pred is None or len(pred.faces)==0 or not np.isfinite(pred.vertices).all():raise ValueError("Invalid generated mesh")
            path=out/(row["id"]+".glb");pred.export(path)
            record.update(valid=True,path=str(path),target_metrics=compare(pred,target_mesh),source_metrics=compare(pred,source_mesh),preservation_metrics=preservation(pred,source_mesh,target_mesh))
            if not a.skip_render:
                try:render(pred,out/(row["id"]+"_views"))
                except Exception as e:record["render_error"]=repr(e)
        except Exception as e:
            record.update(valid=False,error=repr(e))
            if isinstance(e,torch.cuda.OutOfMemoryError):raise
        results.append(record);print(json.dumps(record),flush=True)
        (out/"metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results))
    valid=[r for r in results if r["valid"]]
    summary={"checkpoint":a.checkpoint,"expected":8,"attempted":len(results),"complete":len(results)==8,"valid":len(valid),"sampling":{"seed":20261013,"steps":50,"guidance":1},"note":"Training-add-pair geometry diagnostic, separate from held-out benchmark; velocity-loss gate did not require successful geometry memorization."}
    for metric in ["cd_squared_sum","fscore_005"]:
        if valid:
            summary["generated_"+metric]=float(np.mean([r["target_metrics"][metric] for r in valid]))
            summary["unchanged_source_"+metric]=float(np.mean([r["unchanged_source_target_metrics"][metric] for r in valid]))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (ROOT/"results/summaries"/("training_geometry_"+a.checkpoint+".json")).write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
