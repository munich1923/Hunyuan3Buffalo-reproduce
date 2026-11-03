# Reproduction inputs and flow

[中文版](REPRODUCTION.zh-CN.md)

The repository is the small project-side layer around the unmodified Hunyuan3D-2.1 trainer. It is not a self-contained copy of the datasets or pretrained weights.

## Data acquisition and preparation

`scripts/native_21dit/Abo_Hssd_datapre/download_metadata.py` fetches the pinned ABO/HSSD metadata CSVs used to identify TRELLIS-500K records. It does not download the source GLBs or image payloads. The original experiment's bulk assets were assembled from a separate NAS/data host, so that source access and any required credentials/licenses must be provisioned independently.

With the licensed raw GLBs and the fixed source manifests available under the configured workspace, `scripts/native_21dit/Abo_Hssd_datapre/prepare_geometry.py` invokes the pinned upstream Blender renderer and Hunyuan3D-2.1 `watertight_and_sample.py`. `build_manifest.py` preserves the existing deduplication and train/validation/test split. The original-image workflow uses `render_images.py` and the existing TRELLIS source-view records; it selects one fixed image for the same UID and does not render a new camera view. Some relay helpers assume the original H100/4090 host aliases and NAS layout.

`scripts/native_21dit/native21_official/prepare_abo_hssd_index.py` validates and pairs the frozen same-UID image and official surface files. The tracked `assets/native21_abo_hssd_single_image/` files record the selected pairs, counts, and provenance. Their image, surface, and receipt payloads are excluded from Git and must be present at the paths in each manifest row before training.

The accepted geometry input is the upstream `<uid>_surface.npz` in its official unit-box coordinates. Do not substitute `_vae_points.npy`, re-normalize XYZ, or use an alias image for a missing UID.

The ABO/HSSD preparation contract in `configs/abo_hssd_v1.json` pins the metadata/source revisions, split seed and ratios, geometry parameters, and preparation workspace paths. It configures data identification and preparation; the official training settings are in the separate `native21_official_abo_hssd_1b_20261103` JSON/YAML pair. See [ABO/HSSD dataset layout](ABO_HSSD_DATA_LAYOUT.md) for the tracked manifests and external payload tree.

## Training

Provision the model files listed and SHA-pinned in `configs/native21_official_abo_hssd_1b_20261103.json`. The training YAML loads the official ShapeVAE, HunYuanDiTPlain, and SingleImageEncoder implementations from the pinned submodule/local DINO directory. Run the official `src/Hunyuan3D-2.1/hy3dshape/main.py` from the configured Linux workspace with this YAML and seed `20261101`.

The configuration describes a fresh run from official pretrained weights with 1,000,000,000 maximum optimizer updates, microbatch 1, accumulation 8, checkpoints every 3,000 updates, and loss/pre-clip DiT gradient-norm logging every 500 updates. It writes to a separate reproduction output directory. The custom data module is required because the upstream dataset loader expects WebDataset-style multiview folders rather than the ABO/HSSD same-UID pairing represented here.

The active server run uses a different source-contract snapshot. The cleaned repository's code/config hashes intentionally do not authorize restoring that run's checkpoints; use the fresh reproduction configuration and official pretrained weights.
