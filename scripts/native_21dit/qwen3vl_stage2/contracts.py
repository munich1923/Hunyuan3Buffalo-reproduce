"""Text experiment contract; reuse official geometry/pretrained identities."""
from pathlib import Path

import yaml

from .shared import ROOT, batch_settings, read, verify_upstream
from .text import MODEL_ID, MODEL_REVISION, sha


def qwen_files(model_path, revision=MODEL_REVISION):
    root = Path(model_path)
    if revision != MODEL_REVISION:
        raise ValueError("Qwen revision differs from the reviewed snapshot")
    # snapshot_download(local_dir=...) writes this receipt. Standard HF snapshot
    # directories encode the revision in their path instead.
    snapshot_path = root.name == revision and root.parent.name == "snapshots"
    index = read(root / "model.safetensors.index.json")
    names = {"config.json", "model.safetensors.index.json", "tokenizer_config.json",
             "tokenizer.json", "chat_template.json", "generation_config.json",
             "merges.txt", "vocab.json", *index["weight_map"].values()}
    files = {name: root / name for name in sorted(names)}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("Incomplete pinned Qwen snapshot")
    if not snapshot_path:
        for name in files:
            receipt = root / ".cache/huggingface/download" / (name + ".metadata")
            lines = receipt.read_text().splitlines() if receipt.is_file() else []
            if not lines or lines[0] != revision:
                raise ValueError("Missing pinned HF local_dir revision receipt: " + name)
    if read(files["config.json"])["text_config"]["hidden_size"] != 4096:
        raise ValueError("Unexpected Qwen text hidden size")
    return files


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def integration_contract(config):
    from .data import accepted_rows

    training = yaml.safe_load(Path(config["training_yaml"]).read_text(encoding="utf-8"))
    batching = batch_settings(training)
    if training["dataset"]["target"] != "scripts.native_21dit.qwen3vl_stage2.data.ABOHSSDTextDataModule":
        raise ValueError("Use the independent text data module")
    model = training["model"]["params"]
    if training["model"]["target"] != "hy3dshape.models.diffusion.flow_matching_sit.Diffuser":
        raise ValueError("The official Diffuser must remain unchanged")
    expected = {
        "cond_stage_config": "scripts.native_21dit.qwen3vl_stage2.conditioner.Qwen3VLTextConditioner",
        "denoiser_cfg": "scripts.native_21dit.qwen3vl_stage2.denoiser.MaskedHunYuanDiTPlain",
        "first_stage_config": "hy3dshape.models.autoencoders.ShapeVAE",
    }
    for key, target in expected.items():
        if model[key]["target"] != target:
            raise ValueError("Unexpected model target: " + key)
    optimizer = model["optimizer_cfg"]
    if not optimizer.get("train_image_encoder") or optimizer["image_encoder_lr"] != config["lr"]:
        raise ValueError("Enable the official trainable-conditioner optimizer group")
    if training["training"]["steps"] != config["max_updates"]:
        raise ValueError("YAML/settings max_steps differ")
    if training["training"]["output_dir"] != config["run_dir"]:
        raise ValueError("YAML/settings output directory differs")
    if training["training"]["every_n_train_steps"] != config["checkpoint_every"]:
        raise ValueError("YAML/settings checkpoint interval differs")
    verify_upstream(config)
    files = {}
    for key, model_key, subfolder, digest_key in (
        ("dit", "denoiser_cfg", "hunyuan3d-dit-v2-1", "model_sha256"),
        ("vae", "first_stage_config", "hunyuan3d-vae-v2-1", "vae_sha256"),
    ):
        folder = Path(model[model_key]["from_pretrained"]) / subfolder
        weight = folder / "model.fp16.ckpt"
        digest = sha(weight)
        if digest != config[digest_key]:
            raise ValueError("Official pretrained weight changed: " + key)
        files[key + "_weights"] = digest
        files[key + "_config"] = sha(folder / "config.yaml")
        if key == "dit":
            params = yaml.safe_load((folder / "config.yaml").read_text())["model"]["params"]
            if params["context_dim"] != 1024 or params["use_attention_pooling"] or params.get("with_decoupled_ca", False):
                raise ValueError("Expected DiT context_dim=1024, no pooling or decoupled condition")
    cond = model["cond_stage_config"]["params"]
    vlm_files = qwen_files(cond["model_path"], cond["revision"])
    rows, ready = accepted_rows(config["paired_index_root"], check_geometry=False)
    if len(rows) != config["train_count"] or sha(Path(config["paired_index_root"]) / "ready.json") != config["paired_ready_sha256"]:
        raise ValueError("The accepted baseline UID/split snapshot changed")
    dataset = training["dataset"]["params"]
    metadata = Path(dataset["metadata_path"])
    if dataset["paired_index_root"] != config["paired_index_root"]:
        raise ValueError("YAML/settings paired index differs")
    return {
        "kind": "abo_hssd_text_stage2_v2", "upstream_commit": config["upstream_commit"],
        "settings_sha256": sha(config["_settings_path"]),
        "yaml_sha256": sha(config["training_yaml"]), "scripts": source_hashes(),
        "batching": batching, "condition_mask": "main_mask_bool_true_is_valid",
        "paired_ready_sha256": sha(Path(config["paired_index_root"]) / "ready.json"),
        "paired_files": ready["files"], "accepted_count": len(rows),
        "metadata_sha256": sha(metadata), "dataset": dataset,
        "official_pretrained": files, "qwen_repo": MODEL_ID, "qwen_revision": cond["revision"],
        "qwen_files": {name: sha(path) for name, path in vlm_files.items()},
        "conditioner": cond,
    }
