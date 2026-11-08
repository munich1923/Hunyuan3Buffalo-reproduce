"""Small text-only H100 prefix comparison; never trains or generates 3D."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def tensor_summary(value):
    value = value.detach().float()
    return {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
            "mean": float(value.mean()), "std": float(value.std()),
            "mean_token_l2": float(value.norm(dim=-1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3vl_stage2_abo_hssd_20261105.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609":
        raise RuntimeError("Expose only the authorized physical GPU1 UUID")
    if args.output.exists():
        raise FileExistsError("Choose a new comparison output directory")
    if args.per_source < 1 or args.max_new_tokens < 1:
        raise ValueError("Sample/generation limits must be positive")
    project_root = Path(__file__).resolve().parents[3]
    upstream_root = str(project_root / "src/Hunyuan3D-2.1/hy3dshape")
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)
    from scripts.native_21dit.qwen3vl_stage2.conditioner import Qwen3VLTextConditioner
    from scripts.native_21dit.qwen3vl_stage2.contracts import qwen_files
    from scripts.native_21dit.qwen3vl_stage2.data import accepted_rows
    from scripts.native_21dit.qwen3vl_stage2.text import MODEL_ID, load_captions, sha

    config = yaml.safe_load(args.config.read_text())
    data = config["dataset"]["params"]
    rows, _ = accepted_rows(data["paired_index_root"], split="val", check_geometry=False)
    rng = random.Random(args.seed)
    selected = []
    for source in ("ABO", "HSSD"):
        candidates = sorted((r for r in rows if r["source"] == source), key=lambda r: r["uid"])
        selected.extend(rng.sample(candidates, args.per_source))
    captions = load_captions(data["metadata_path"], selected, data["metadata_revision"])
    params = config["model"]["params"]["cond_stage_config"]["params"]
    files = qwen_files(params["model_path"], params["revision"])
    model_hashes = {name: sha(path) for name, path in files.items()}
    torch.manual_seed(args.seed)
    cond = Qwen3VLTextConditioner(**params).to(device="cuda", dtype=torch.float16).eval()
    prefix = cond.task_prefix
    reports = []
    for row in selected:
        record = captions[row["uid"]]
        index = record["nonempty_caption_indices"][0]
        caption = record["captions"][index]
        report = {"uid": row["uid"], "source": row["source"], "caption_index": index, "caption": caption}
        pooled = []
        for name, task_prefix in (("without_prefix", ""), ("with_prefix", prefix)):
            cond.task_prefix = task_prefix
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden, valid = cond.encode_text(caption)
                context = cond.connector(hidden.to(cond.connector[0].weight))
                inputs = cond.tokenize_text(caption)
                output = cond.vlm.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            pooled.append((hidden.float().sum(dim=1) / valid.sum(dim=1, keepdim=True)).cpu())
            report[name] = {
                "response": cond.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True),
                "hidden": tensor_summary(hidden), "connector": tensor_summary(context),
            }
        report["pooled_hidden_cosine"] = float(F.cosine_similarity(pooled[0], pooled[1]).item())
        reports.append(report)
        print(row["uid"], "completed", flush=True)
    result = {
        "experiment": "small_text_only_prefix_review",
        "model_id": MODEL_ID, "revision": params["revision"], "model_files_sha256": model_hashes,
        "config_sha256": sha(args.config), "metadata_sha256": sha(data["metadata_path"]),
        "seed": args.seed, "per_source": args.per_source, "max_new_tokens": args.max_new_tokens,
        "prefix": prefix, "input": "text_only",
        "connector": "randomly_initialized_interface_probe_not_a_trained_quality_measure",
        "hidden_selection": "input_sequence_last_layer_no_generated_answer_tokens", "samples": reports,
    }
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Prefix comparison / 前缀对比", "", "Small text-only experiment; the connector is untrained.", ""]
    for report in reports:
        lines.extend(["## " + report["uid"] + " (" + report["source"] + ")", "", report["caption"], ""])
        for name in ("without_prefix", "with_prefix"):
            lines.extend(["### " + name, "", report[name]["response"], ""])
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
