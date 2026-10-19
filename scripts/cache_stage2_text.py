"""Frozen text-only Qwen hidden cache. GPU must be explicitly assigned and empty."""
from stage2_runtime import *
from stage2_cache import cache_base,text_key,tokenize_condition


def main():
    p=parser(__doc__,gpu=True)
    p.add_argument("--split",choices=["all","train","val","test"],default="all")
    p.add_argument("--limit",type=int)
    args=p.parse_args(); cfg=config(args.config)
    output=path(cfg["output_root"])/"preparation"
    with lock("environment/stage2_text.lock"),gpu_lease(args.gpu,output):
        import torch
        from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration
        base,key=cache_base(cfg,"text",True)
        tokenizer=AutoTokenizer.from_pretrained(str(path(cfg["models"]["vlm"])),
                                                local_files_only=True)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(path(cfg["models"]["vlm"])),torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",local_files_only=True).eval().cuda()
        model.requires_grad_(False)
        splits=["train","val","test"] if args.split=="all" else [args.split]
        lengths=[]
        for split in splits:
            rows=manifests(cfg,split)
            for index,row in enumerate(rows[:args.limit] if args.limit else rows):
                target=base/split/(row["uid"]+".pt")
                expected=text_key(row,key)
                if target.exists():
                    old=torch.load(target,map_location="cpu",weights_only=True)
                    if old.get("fingerprint")!=expected: raise RuntimeError("Stale text cache")
                    lengths.append(old["hidden"].shape[0]); continue
                encoded,selected,rendered=tokenize_condition(tokenizer,row["prompt"])
                with torch.inference_mode():
                    result=model(**{k:v.cuda() for k,v in encoded.items()},
                                 output_hidden_states=True,use_cache=False,return_dict=True)
                    hidden=result.hidden_states[-1][0,selected].cpu().to(torch.bfloat16)
                if hidden.shape!=(len(selected),cfg["cache"]["hidden_dim"]) or not torch.isfinite(hidden).all():
                    raise ValueError("Invalid hidden states")
                save_tensor(target,{"hidden":hidden,"selected_indices":selected,
                    "selected_token_ids":encoded["input_ids"][0,selected].tolist(),
                    "input_token_count":encoded["input_ids"].shape[1],
                    "prompt":row["prompt"],"fingerprint":expected,"contract":key,"uid":row["uid"]})
                lengths.append(len(selected))
                if (index+1)%100==0: print("TEXT",split,index+1,flush=True)
        # Empty text is diagnostic only, not a trained unconditional branch.
        encoded,selected,_=tokenize_condition(tokenizer,"")
        with torch.inference_mode():
            result=model(**{k:v.cuda() for k,v in encoded.items()},
                         output_hidden_states=True,use_cache=False,return_dict=True)
            empty=result.hidden_states[-1][0,selected].cpu().to(torch.bfloat16)
        save_tensor(base/"empty.pt",{"hidden":empty,"contract":key,"diagnostic_only":True})
        atomic_json(base/"progress.json",{"processed_lengths":len(lengths),
                    "min":min(lengths) if lengths else None,"max":max(lengths) if lengths else None,
                    "partial":args.limit is not None,"command":sys.argv,**clock()})

if __name__=="__main__": main()
