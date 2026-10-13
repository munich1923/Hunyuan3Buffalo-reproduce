"""Pilot flow-matching training; reports fitting separately from held-out validation."""
import argparse,json,random,time
from pathlib import Path
import torch
from remote_runtime import ROOT,configure
from edit_model import Editor,load_dit
configure()
def load_records(split):
    return [json.loads(l) for l in (ROOT/"assets/manifests"/(split+"_prepared.jsonl")).read_text().splitlines()]
def item(row,split):
    shape=torch.load(ROOT/"assets/cache/shape"/split/(row["id"]+".pt"),map_location="cpu",weights_only=True)
    vlm=torch.load(ROOT/"assets/cache/vlm"/split/(row["id"]+".pt"),map_location="cpu",weights_only=True)
    return shape,vlm
def tensors(row,split,no_op=False):
    shape,vlm=item(row,split);scale=shape["scale_factor"]
    source=shape["source"][None].to("cuda",torch.bfloat16)*scale
    target=(shape["source"] if no_op else shape["target"])[None].to("cuda",torch.bfloat16)*scale
    h=vlm["noop_hidden" if no_op else "hidden"][None].cuda().float()
    return source,target,h
def objective(editor,source,target,h,generator=None):
    noise=torch.randn(target.shape,device="cuda",dtype=torch.bfloat16,generator=generator)
    t=torch.rand(target.shape[0],device="cuda",generator=generator)
    noisy=(1-t[:,None,None])*noise+t[:,None,None]*target
    with torch.autocast("cuda",dtype=torch.bfloat16):
        pred=editor(noisy.to(torch.bfloat16),t.to(torch.bfloat16),h,source)
    return (pred.float()-(target-noise).float()).square().mean()
def validation(editor,rows,split,limit=16):
    values=[];editor.eval()
    with torch.no_grad():
        for i,row in enumerate(rows if limit is None else rows[:limit]):
            g=torch.Generator(device="cuda").manual_seed(9000+i)
            source,target,h=tensors(row,split)
            values.append(float(objective(editor,source,target,h,g)))
    return sum(values)/len(values)
def save(editor,path,step,stage,stats):
    torch.save({"adapter":editor.adapter_state(),"step":step,"stage":stage,"stats":stats,"seed":20261013,"source_commit":"82920d643c0dc2f7bfd7255f45f62d386edfe60c"},path)
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["overfit","connector","lora"],required=True);p.add_argument("--seconds",type=float,default=20000);a=p.parse_args()
    torch.manual_seed(20261013);random.seed(20261013)
    dit=load_dit();editor=Editor(dit,lora=a.stage=="lora").cuda()
    path=ROOT/"results/checkpoints";path.mkdir(parents=True,exist_ok=True)
    if a.stage=="lora":editor.load_adapter(path/"connector_best.pt")
    rates=[{"params":editor.connector.parameters(),"lr":1e-4 if a.stage!="lora" else 5e-5}]
    if a.stage=="lora":rates.append({"params":[p for n,p in editor.dit.named_parameters() if p.requires_grad],"lr":1e-4})
    opt=torch.optim.AdamW(rates,weight_decay=0.01)
    rows=load_records("train");val=load_records("val")
    if a.stage=="overfit":rows=[x for x in rows if x["operation"]=="add"][:8]
    if a.stage=="overfit":assert len({x["group"] for x in rows})==8,"Require 8 independent original pairs"
    maxsteps={"overfit":200,"connector":500,"lora":2000}[a.stage]
    start=time.monotonic();history=[];initial=validation(editor,rows,"train");best=float("inf")
    log=ROOT/"logs"/("train_"+a.stage+".jsonl")
    baseline_val=validation(editor,val,"val",limit=16 if a.stage=="overfit" else None)
    print(json.dumps({"stage":a.stage,"initial_train":initial,"initial_val":baseline_val,"trainable_parameters":sum(p.numel() for p in editor.parameters() if p.requires_grad)}),flush=True)
    step=0;completed_steps=0
    for step in range(1,maxsteps+1):
        if time.monotonic()-start>a.seconds:break
        opt.zero_grad(set_to_none=True);losses=[]
        for micro in range(4):
            row=rows[(step*4+micro)%len(rows)] if a.stage=="overfit" else random.choice(rows)
            no_op=a.stage!="overfit" and random.random()<0.1
            source,target,h=tensors(row,"train",no_op)
            loss=objective(editor,source,target,h)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            (loss/4).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_([p for p in editor.parameters() if p.requires_grad],1)
        opt.step();completed_steps=step
        stats={"step":step,"loss":sum(losses)/4,"elapsed_seconds":time.monotonic()-start}
        if step%20==0 or step==maxsteps:
            stats["train_probe"]=validation(editor,rows,"train")
            stats["validation"]=validation(editor,val,"val")
            history.append(stats)
            if stats["validation"]<best:
                best=stats["validation"];save(editor,path/(a.stage+"_best.pt"),step,a.stage,stats)
            save(editor,path/(a.stage+"_latest.pt"),step,a.stage,stats)
            print(json.dumps(stats),flush=True)
        with log.open("a") as f:f.write(json.dumps(stats)+"\n")
    final=validation(editor,rows,"train");final_val=validation(editor,val,"val",limit=16 if a.stage=="overfit" else None)
    report={"stage":a.stage,"steps":completed_steps,"initial_train":initial,"final_train":final,"initial_val":baseline_val,"final_val":final_val,"elapsed_seconds":time.monotonic()-start,"best_validation":best,"train_entries":len(rows),"validation_probe_entries":min(16,len(val)),"validation_split_entries":len(val),"final_validation_entries":min(16,len(val)) if a.stage=="overfit" else len(val),"history":history,"note":"Velocity loss is a fitting diagnostic, not a geometry editing success metric."}
    if a.stage=="overfit":
        report["passed"]=final<initial*0.9 and len(history)>0
    save(editor,path/(a.stage+"_latest.pt"),step,a.stage,report)
    if not (path/(a.stage+"_best.pt")).exists():save(editor,path/(a.stage+"_best.pt"),step,a.stage,report)
    (ROOT/"environment"/("training_"+a.stage+".json")).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if a.stage=="overfit" and not report["passed"]:raise RuntimeError("8-sample fitting gate did not improve fixed-noise loss by 10%; no scale-up")
if __name__=="__main__":main()
