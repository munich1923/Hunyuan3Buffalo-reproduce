# Hunyuan3D-2.1 ABO/HSSD Training Reproduction

[中文版](README.zh-CN.md)

The separate text-only Qwen3-VL Stage2 implementation is documented in
[Qwen3-VL Stage2](docs/QWEN3VL_STAGE2.md). It replaces the image conditioner and
adds a local DiT condition-padding-mask extension, retaining the official training
entry, loss and optimizer. A small text-only prefix review ran on H100; the chosen
prefix is configured, but formal training has not started.

This repository contains the project code and fixed configuration for the ABO/HSSD single-image-conditioned Hunyuan3D-2.1 training experiment. The official Hunyuan3D-2.1 source is pinned as a Git submodule at commit `82920d643c0dc2f7bfd7255f45f62d386edfe60c`.

## Reproduction inputs

The repository tracks source code, split and pairing manifests, and configuration. It does not track model weights or image and geometry payloads. The paired payloads used by the experiment total about 35.3 GB; the Hunyuan3D-2.1 DiT/VAE and DINO weights total about 8.6 GB. Obtain them from their licensed sources, place them at the paths in the configuration, and verify the recorded SHA256 values before training.

The split manifests contain records for 8,626 training pairs, 479 validation pairs, and 479 test pairs. They preserve the original split and same-UID image/geometry pairing. Their recorded payload paths refer to the H100 workspace used for the experiment; update the paths and regenerate the index readiness receipt when deploying elsewhere.

## What is official and what is project-specific

- The upstream project provides `hy3dshape/main.py`, the model, loss, optimizer, image encoder, and official surface-point sampling.
- `scripts/native_21dit/Abo_Hssd_datapre/` contains the ABO/HSSD metadata, geometry and image preparation workflow. `prepare_geometry.py` calls the pinned upstream renderer and `watertight_and_sample.py`.
- `scripts/native_21dit/native21_official/` contains the ABO/HSSD paired-data adapter, immutable-index validation, checkpoint data/RNG state, and update-aligned TensorBoard scalars. The adapter is needed because the official loader expects a different WebDataset directory layout.
- `configs/` records the data-preparation settings and the one-billion-update official `main.py` training settings.

The current training config initializes from official pretrained weights, uses microbatch 1 with accumulation 8, saves every 3,000 optimizer updates, and records loss plus pre-clip DiT gradient norm every 500 updates. It does not configure automatic mesh evaluation or automatic stopping.

The ABO/HSSD source assets were assembled on a separate data host and the H100 workspace. The repository's metadata downloader does not fetch the 35.3 GB image and geometry payloads from a public endpoint. The existing prepared payloads remain on the H100; reproducing the data build elsewhere requires access to the same licensed source assets and their preparation receipts.

## Upstream checkout

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/munich1923/Hunyuan3Buffalo-reproduce.git
```

To initialize the upstream checkout after a regular clone:

```powershell
git submodule update --init --recursive
```

Training runs on the configured Linux/CUDA environment. The checked-in configuration preserves the experiment's H100 workspace paths; it is not directly runnable from Windows.

## Current reproduction config

- `configs/native21_official_abo_hssd_1b_20261103.yaml`
- `configs/native21_official_abo_hssd_1b_20261103.json`

`configs/abo_hssd_v1.json` configures data identification and preparation. It pins metadata/upstream revisions, split and geometry rules, and preparation paths; it is not the Hunyuan3D training config. See [dataset layout](docs/ABO_HSSD_DATA_LAYOUT.md) for the paired file format.

The config is assigned a new output directory so it cannot overwrite the active server run. The clean project starts a fresh experiment from official pretrained weights. It is not compatible with resuming the active run's checkpoint because the cleaned source set has a different source-contract hash.
