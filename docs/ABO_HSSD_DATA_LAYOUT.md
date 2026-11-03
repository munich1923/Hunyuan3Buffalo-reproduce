# ABO/HSSD Dataset Layout

[简体中文](ABO_HSSD_DATA_LAYOUT.zh-CN.md)

The repository tracks compact manifests and provenance. Image files, surface arrays, model weights, checkpoints, and runtime receipts stay outside Git. The current manifests contain 8,626 train, 479 validation, and 479 test pairs; nine records without their own UID image were excluded without changing the original split assignment of the remaining records.

## Repository files

```text
assets/
└── native21_abo_hssd_single_image/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── rejected.jsonl
    ├── summary.json
    └── ready.json
```

Each JSONL row describes one image/geometry pair with the same UID. Important fields include:

| Field | Meaning |
| --- | --- |
| `uid` | Stable dataset identity, prefixed by `abo_` or `hssd_` and followed by the source asset SHA256. |
| `split` | `train`, `val`, or `test`; must match the JSONL filename. |
| `source` | `ABO` or `HSSD`. |
| `image` | Absolute path to the selected image on the reference H100 workspace. |
| `image_sha256` | SHA256 of the selected original PNG. |
| `surface` | Absolute path to the official `<uid>_surface.npz` file. |
| `sha256` | SHA256 of the surface archive. |
| `geometry_receipt`, `image_receipt` | Preparation receipts that bind each payload to its UID and source identity. |
| `coordinate_rule` | `official_surface_npz_unit_box_no_extra_transform`. |

The lists in this repository preserve the accepted fixed pairing and reference-server absolute paths. They do not contain the payloads. To move data, copy the payloads while preserving each row's UID relationship, update the paths, and regenerate the readiness receipt through the index builder. Do not replace a missing UID image with a deduplication alias.

## Reference payload tree

The paired rows point into this prepared-data tree on the Linux workspace:

```text
assets/abo_hssd_v1/
├── processed/<uid>/upstream_geometry_v1/
│   ├── <uid>_surface.npz
│   └── complete.json
└── renders_original/<uid>/
    ├── <selected-view>.png
    └── image_complete.json
```

Some geometry receipts may live under `processed_pilot/<uid>/upstream_geometry_v1/`; the accepted row always records its actual location. The training adapter consumes the official `<uid>_surface.npz` archive. It expects `random_surface` and `sharp_surface` arrays, each shaped `[124928, 6]` with `float16` values containing XYZ and normals. The adapter samples 81,920 points online, normalizes normals, and appends the zero label expected by the official loader, producing `[81920, 7]`.

Selected source images are original 1024x1024 RGB or RGBA PNGs. The adapter composites RGBA against white, preserves alpha as a mask, applies the accepted 1.15 padding, and resizes to 518 pixels. A training sample yields image `[3, 518, 518]` and mask `[1, 518, 518]` tensors.

## Pairing and readiness

`ready.json` records split counts, source manifest hashes, hashes of the tracked split files, the upstream commit, and the accepted coordinate rule. The index validator checks the manifest hashes, each UID/split relation, payload size and modification time, receipt hashes, and optionally full payload hashes. During training, the data module validates the frozen index and uses only the train split.

The manifest records valid file pairing and numeric input contracts. It does not prove camera-to-geometry alignment or generated-shape quality. Those questions require separate validation.
