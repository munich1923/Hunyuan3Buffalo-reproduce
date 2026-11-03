# ABO + HSSD original image migration

[中文版](ABO_HSSD_ORIGINAL_IMAGES.zh-CN.md)

This CPU-only preparation pipeline selects and copies one original TRELLIS conditional PNG for each geometry-qualified UID. It preserves the official geometry inputs; camera alignment to VAE coordinates is not established by this workflow.

## Identity and view contract

The request binds the candidates SHA, migration inventory SHA, and the existing trellis_piflow image index SHA. Each source record must match dataset, original SHA256 and file_identifier. Original GLBs are not hashed again: the completed raw migration receipts and current size/mtime are checked.

The TRELLIS camera-to-world matrix uses a Z-up scene centered at the origin. Elevation is atan2(z,hypot(x,y)); azimuth is atan2(y,x). The interval is inclusive [-60,60] degrees, with unrestricted azimuth. camera_angle_x is horizontal FOV, not elevation. Camera matrices must be finite affine proper rigid transforms; FOV must be in (0,pi).

Eligible PNGs must fully decode as original 1024x1024 RGBA. They are sorted by original filename. A local Random seeded from SHA256([20261027,"original-render-view-v1",UID]) chooses one uniformly among valid frames. Global/caption/noise/posterior/time RNGs are untouched. This samples the finite available views uniformly; it is not uniform in elevation. Images are kept byte-for-byte without RGB conversion or resizing.

Source images missing from the historical index, invalid source identities, bad transforms, missing PNGs and no eligible view are recorded explicitly. They never cause replacement objects, relaxed angles or new rendering. Geometry remains in the original split even when an image is unavailable.

## Artifacts and phases

Scripts are scripts/native_21dit/Abo_Hssd_datapre/render_images.py, render_images_relay.py, and test_render_images.py. The separate evidence root is results/abo_hssd_v1/render_images_v1. Payloads and paired_index.jsonl are under assets/abo_hssd_v1/renders_original.

The Windows relay loads its code directly from H100 and carries binary data only in memory between existing SSH connections. It writes no local data or credentials. The local machine/network must remain available during source scanning and copying.

Phases: prepare provenance request; source scan with at most two read threads; wait for complete geometry attempts; use the existing CPU manifest builder to establish the original immutable dedup/splits if needed; freeze the qualified image selection; transfer four ABO and four HSSD objects; decode/SHA/restart-reuse verification; transfer remaining selected images; full image/angle/SHA/split/geometry-receipt verification.

The existing GPU preparation controller is not modified or started by this workflow. All image commands use CPU; the manifest subprocess has CUDA_VISIBLE_DEVICES empty. The existing geometry process is observed and never automatically restarted here. Its disappearance before a terminal receipt stops the image workflow for diagnosis.

## Resume and safety

Do not launch another live relay or receiver. Inspect relay_status.json, status.json, process command lines and any failure logs first. The receiver holds transfer.lock. Target assets land in a per-UID partial directory, receive streaming SHA checks for both the PNG and full transforms.json, and are renamed only after image decode and receipt commit. Unreceipted final directories and changed frozen contracts fail closed. Interrupted partial files are retried; complete matching files are reused. Prior scan records can be reused after matching source metadata/frame stat fingerprints; changed observations are retained in source_scan_history.

source_selection_inventory.json is a preflight over candidates, not the qualified frozen image dataset. Only frozen_selection.json, created after complete geometry and immutable split construction, determines transfer subjects. paired_index.jsonl includes every available qualified original UID, its split and representative_uid for any dedup alias. It does not alter train.jsonl/val.jsonl/test.jsonl or geometry receipts.

CPU tests:
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/test_render_images.py

Manual per-stage commands:
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/render_images.py prepare
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/render_images.py freeze
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/render_images.py request --pilot
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/render_images.py verify --pilot
environment/venvs/shape/bin/python -B scripts/native_21dit/Abo_Hssd_datapre/render_images.py verify

The scan_source/send_source modes execute on 4090 from compressed in-memory source and read the NAS only. The relay coordinates inventory_receive/receive on H100. Do not manually run a source mode on H100.

## Acceptance

CPU tests cover signed angles/boundaries, FOV distinction, rigid-matrix validity, deterministic selection/global RNG isolation, identity mismatches, corrupt PNGs, zero eligible frames, premature freeze rejection, interrupted binary transfer, matching receipt reuse and changed target rejection. Eight real pilot assets must pass decode, original SHA, camera and restart-reuse checks before full copying.

final_verification.json reports copied files separately from geometry image coverage, missing source images, selected source/split counts and payload bytes. migration_complete.json concerns image migration only; data_ready remains false here. The original camera/normalization record is retained, and camera_alignment_to_vae is explicitly "not established".

Implementation validation: 11 CPU tests passed, including split alias inheritance and missing-image preservation. Source read preflight uses two threads; initial observed source VmHWM was 115508 kB, local relay working set 34930688 bytes. These are observed samples, not guaranteed peak bounds.
