import argparse,json
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from PIL import Image
from remote_runtime import ROOT,configure
configure()
def hidden(model,processor,views,instruction):
    content=[{"type":"image","image":Image.open(p).convert("RGB")} for p in views]
    content.append({"type":"text","text":"These are four views of the same 3D object. Apply this geometric edit: "+instruction})
    messages=[{"role":"user","content":content}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images=[x["image"] for x in content if x["type"]=="image"]
    inputs=processor(text=[text],images=images,return_tensors="pt",padding=True).to(model.device)
    with torch.no_grad():
        outputs=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
    h=outputs.hidden_states[-1][0][inputs.attention_mask[0].bool()].detach().cpu().to(torch.bfloat16)
    return h
def main():
    p=argparse.ArgumentParser();p.add_argument("--split",default="all");p.add_argument("--limit",type=int);a=p.parse_args()
    modelpath=ROOT/"models/Qwen2.5-VL-7B-Instruct"
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelpath,torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    for x in model.parameters():x.requires_grad_(False)
    processor=AutoProcessor.from_pretrained(modelpath,min_pixels=256*28*28,max_pixels=512*28*28)
    splits=["train","val","test"] if a.split=="all" else [a.split]
    for split in splits:
        rows=[json.loads(l) for l in (ROOT/"assets/manifests"/(split+"_prepared.jsonl")).read_text().splitlines()]
        if a.limit:rows=rows[:a.limit]
        for i,row in enumerate(rows):
            out=ROOT/"assets/cache/vlm"/split/(row["id"]+".pt");out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():continue
            h=hidden(model,processor,row["source_views"],row["instruction"])
            # No-op features are cached too, enabling identity training without contradictory language.
            nop=hidden(model,processor,row["source_views"],"Keep the geometry unchanged.")
            wrong=None
            if split=="test":wrong=hidden(model,processor,row["source_views"],rows[(i+1)%len(rows)]["instruction"])
            torch.save({"hidden":h,"noop_hidden":nop,"shuffled_hidden":wrong,"instruction":row["instruction"],"model_revision":"cc594898137f460bfe9f0759e9844b3ce807cfb5","layer":-1,"render_version":row.get("render_version"),"source_normalized":row["source_normalized"]},out)
            if i%10==0:print("VLM",split,i,list(h.shape),flush=True)
if __name__=="__main__":main()
