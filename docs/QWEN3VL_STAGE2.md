# Text-only Qwen3-VL Stage2

[中文版](QWEN3VL_STAGE2.zh-CN.md)

This is a local implementation of Buffalo-style VLM hidden-state conditioning,
using public Qwen3-VL weights. It is not a reproduction of the unavailable
Hunyuan3D-VLM weights. No H100 job, model download, or real-model inference was
performed as part of the local implementation.

## Change boundary

The original single-image baseline, its configuration, and the pinned upstream
submodule remain intact. New code lives in `scripts/native_21dit/qwen3vl_stage2/`.
The training entry remains upstream `hy3dshape/main.py`, using the unchanged
Diffuser, VAE, flow-matching loss, AdamW construction and LR scheduler. A local
`MaskedHunYuanDiTPlain` subclass adds condition padding-mask support while retaining
the official parameter names, shapes and pretrained-weight loader. No upstream
source is edited and the extension adds no trainable parameters.

This experiment has its own Dataset, DataModule, callbacks, scalar logging and
utilities. It neither imports `native21_official` nor includes that experiment's
scripts in its recovery contract. The accepted data manifests, geometry and
pinned upstream implementation remain shared inputs.

```text
original metadata caption -> optional fixed prefix -> Qwen text hidden states
    -> trainable 4096 -> 1024 -> 1024 MLP -> contexts["main"] -> DiT
tokenizer attention_mask -> contexts["main_mask"] -> each cross-attention layer
official surface NPZ -> frozen official VAE (sample_posterior=True) -> shape target
```

Only text is supplied to Qwen. The dataset does not open images or return
`image`/`mask`. The conditioner rejects image/mask inputs, unexpected inputs and
image/video placeholder token IDs. It uses AutoTokenizer, not an image processor.
It takes the final input-sequence hidden state through `vlm.model(...)`, without
language generation or logits. MLP gradients remain enabled outside Qwen's
`no_grad` block.

The existing Hunyuan inference pipeline is constructed for its original model
initialization/casting behavior, but its image-only sampling API is not used or
adapted into a text-to-3D inference API in this change. Automatic evaluation is
disabled. The separate comparison tool generates language only.

## Configuration and data

The new experiment uses:

- `configs/qwen3vl_stage2_abo_hssd_20261105.yaml`: model, data and Trainer settings.
- `configs/qwen3vl_stage2_abo_hssd_20261105.json`: source/model/data recovery contract.
- `environment/qwen3vl-stage2-requirements.txt`: additional dependency pins for a
  separate H100 shape environment; retain the reference runtime in
  [environment notes](../environment/README.md).

The defaults preserve 8,626 accepted train UIDs and the existing 479/479 validation
and test splits, object sampling with replacement, and the exact upstream surface
sampling. No missing-image exclusions are reversed and no coordinates change.
The paired index is read to preserve accepted geometry identities; its image
payloads are not required by this text-only data path.

The expected caption source is the original
`assets/abo_hssd_v1/metadata/candidates.jsonl` produced by the existing preparation
code. H100 file availability has not been checked. A relevant row has this form:

```json
{"uid":"abo_<sha256>","source":"ABO","raw_sha256":"<sha256>","metadata_revision":"288b06111bc4d06800293ab087989f852a632af2","captions":["a wooden chair","a chair with four legs"]}
```

Join by exact UID, source and source hash. Use original metadata rather than the
deduplicated manifest's merged alias captions. Missing, duplicate, empty or
mismatched records fail explicitly; there is no caption/image alias fallback.
If captions exist only in the raw metadata CSVs on H100, prepare the expected
JSONL or supply its actual path after a separate review; this implementation does
not silently change sources or rebuild splits.

Each object occurrence uniformly samples one nonempty caption **entry**, retaining
duplicates and original strings. A dedicated Python RNG seeded `20261027` is saved
at the committed sample boundary, independently of UID, point and training RNGs.
The same UID can receive different captions on different occurrences. Restore
replays the choices exactly, including when uncommitted samples were prefetched.
There is no added epoch or fixed-caption assignment.

The provisional user message is:

```text
Create a 3D object that faithfully matches the following description:
{caption}
```

`task_prefix: ""` disables it. Qwen's own chat template is used with
`add_generation_prompt=True`. All valid input tokens are retained; sequences are
variable length. The explicit `max_length: 2048` guard raises on overlength inputs
instead of truncating them. This is a project default, not a Buffalo requirement.
Official DiT has `context_dim=1024` and `use_attention_pooling=false`, so its
`text_len=1370` does not require resampling this context to 1370 tokens.

For a batch of different-length captions, tokenization uses left padding without
truncation. Qwen receives its attention mask. `encode_text()` returns the padded
hidden tensor `[B,L_max,4096]` and a boolean valid-token mask `[B,L_max]`. The MLP
produces `[B,L_max,1024]`; padded entries are zeroed and the mask is returned as
`main_mask`. The local DiT passes it through every block to SDPA cross-attention,
where boolean `True` means allowed. Qwen's causal attention and DiT's shape
self-attention are unchanged. Padding values and counts must not affect valid
conditions. All-masked samples and malformed masks are rejected.

The masked cross-attention allows PyTorch to choose its backend, including a math
fallback when fused kernels do not support the mask. Actual H100 backend behavior,
memory use and throughput still require a separately requested check. There is
no fixed-length pooling/resampler and no generated-answer tokens in conditioning.

## Parameters, precision and checkpoints

Qwen is frozen in eval mode with BF16 weights. A wrapper preserves the backbone's
floating dtypes through parent `.to()`/`.half()` calls, including FP32 rotary
buffers, without an intermediate FP16 conversion. DiT/VAE retain the existing
FP16-weight and BF16 Trainer AMP behavior. The MLP follows the pipeline's ordinary
FP16 conversion. Its structure is Linear(4096,1024), GELU, Linear(1024,1024).

`optimizer_cfg.train_image_encoder=true` enables the official optimizer's
trainable-conditioner parameter groups. Despite its historical name, it collects
the MLP's trainable parameters; frozen Qwen parameters are excluded. Initial LR is
`1e-5` for DiT and connector. The connector's ordinary weights have decay `0.05`,
biases have decay `0`, and DiT inherits decay `0.01`. Both groups share the existing
50-update warmup/cosine curve with total `1e9` updates. Condition dropping remains
disabled with the existing `drop_ratio=0.1` attribute.

The supplied YAML still defaults to microbatch 1 and accumulation 8. Any positive
integer `dataset.params.batch_size` is accepted by the DataModule and passed to
DataLoader; `training.update_every` sets accumulation independently. On the
authorized single GPU, effective batch is their product; neither value is inferred
or adjusted by Trainer. For example, batch 2 with accumulation 4 gives effective
batch 8. These values have one source in the YAML and are not duplicated in the
JSON settings. `num_workers=0` remains required by the committed sampler recovery
mechanism; this is separate from batch size.

Checkpoints every 3,000 updates and scalars every 500 updates remain unchanged.
The existing `train/grad_norm_l2_pre_clip` measures
DiT only; `train/connector_grad_norm_l2_pre_clip` is separate. The latter is also
written to `connector_scalar_updates.jsonl`; the effective-batch/window losses
remain in `scalar_updates.jsonl`. Both norms are measured after accumulation and
before clipping.

Standard full Lightning checkpoints include Qwen, MLP, DiT, VAE, optimizer,
scheduler, scalar windows and committed data/RNG state. Consequently frozen Qwen
adds approximately 17.5 GB of weight data to **each** full checkpoint; it is not
excluded by custom save hooks. Resume requires matching metadata, configuration,
code and model-file hashes. Image-baseline and abandoned checkpoints are rejected.
The new experiment has its own output directory. Recovery state version 2 stores
consumed samples and consumed DataLoader batches separately. Checkpoint loop
cursors count batches; data RNG snapshots count samples. On this fixed-batch,
single-GPU stream, after U updates these are `U*A` batches and `U*A*B` samples.
Only committed states are saved, so prefetched samples are replayed after restore.
Changing batch/accumulation settings is not an exact resume of an existing run.
The earlier local text checkpoint format is also intentionally incompatible.

## Local verification and later H100 review

Local tests need CPU torch/torchvision and the reference shape data dependencies.
Run without pretrained weights:

```powershell
python -B tests/test_qwen3vl_stage2.py -v
```

Tests exercise mixed-length text batches, image rejection, MLP gradients, frozen
Qwen, dtype protection, exact-UID metadata joins, unchanged official geometry/UID
streams, prefetch recovery and official optimizer grouping. Masked batched DiT
outputs and gradients are compared against individually unpadded official calls;
strict pretrained-key loading and padding invariance are checked. Independent
processes compare continuous versus checkpoint/resumed execution for batch and
accumulation pairs (1,8), (2,3), and (8,1). Imports of the image experiment are
blocked in an isolation probe. A tiny randomly initialized Qwen also checks the
Transformers text-only API. They do not establish real-model numerical stability
or shape generation quality.

For a later, separately requested H100 check, provision
`Qwen/Qwen3-VL-8B-Instruct` at revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Keep the standard HF snapshot directory
or the `.cache/huggingface/download/*.metadata` receipts written by
`snapshot_download(local_dir=...)`. The contract verifies the revision receipt and
hashes all consumed model/tokenizer files. Model loading is local-files-only.
No `device_map=auto` or multi-GPU dispatch is configured.

After that check is authorized, expose physical GPU1 only and invoke the separate
tool from the project root:

```bash
CUDA_VISIBLE_DEVICES=GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m scripts.native_21dit.qwen3vl_stage2.compare_prefix \
  --config configs/qwen3vl_stage2_abo_hssd_20261105.yaml \
  --output results/qwen3vl_prefix_review
```

It compares the same first nonempty caption for four validation ABO and four
validation HSSD objects with/without the prefix. Outputs include greedy language
responses, hidden-state/context shapes, finite checks, token norms and pooled
hidden cosine similarity in JSON and Markdown. The MLP is randomly initialized,
so its output is an interface check, not a learned quality comparison. Review the
results before deciding the training prefix. The tool does not start training or
3D evaluation, and is never called by a training callback.

Formal training remains a separate user decision and uses the official `main.py`
with the new YAML and seed `20261101`, on the authorized GPU only. There is no new
launcher, supervisor, checkpoint watcher, automatic pause or evaluation schedule.
