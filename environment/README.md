# Reproducibility

The experiment uses two local-prefix Conda environments under environment/venvs/.
Shape uses Python 3.10 and PyTorch 2.5.1+cu124, with the geometry subset of the official Hunyuan3D 2.1 dependencies.
VLM uses the same PyTorch build but Transformers 4.51.3.
Dependency snapshots are written only after successful installation and pip check.

Commands (run in the remote experiment root):
- /opt/conda/bin/python scripts/install_environments.py
- /opt/conda/bin/python scripts/download_artifacts.py
- /opt/conda/bin/python scripts/run_pilot.py --prepare-only
- /opt/conda/bin/python scripts/run_pilot.py

The preparation-only command must finish and the data-review images must be visually checked before full pilot execution.
run_state.json records stages, process IDs, return codes and GPU allocation seconds.
Download receipts in logs/downloads/*.json record pinned revision, local file, expected and actual size/hash.
A successful artifact download does not certify model load or experiment success.
An adapter checkpoint contains only connector and LoRA weights; no base weights are duplicated.

Data QC uses broad anomaly checks, not semantic guarantees. One source-derived transform applies to both geometries.
Source extent is 1.5 in the shared coordinates, leaving space for additions. Pairs outside [-1,1] after this transform are rejected to prevent clipping by the VAE mesh extractor.
The training count of 1024 refers to 512 independent forward pairs and 512 reverse pairs. Validation is 64 independent pairs and their reversals.

Flow-matching losses measure fitting, not editing success. Final evaluation uses target geometry and same-source shuffled instructions.
See method_notes.md for deviations and assumptions.
