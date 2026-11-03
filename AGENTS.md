# Project instructions

This repository is the minimal source project for the Hunyuan3D-2.1 ABO/HSSD training reproduction.

- Keep upstream Hunyuan3D-2.1 pinned at the submodule commit recorded in the root README.
- Keep large datasets, model weights, virtual environments, caches, checkpoints, and runtime logs out of Git.
- Preserve the fixed ABO/HSSD split and same-UID image/geometry pairing. Do not silently substitute aliases or change coordinates.
- Use the official `hy3dshape/main.py` training entry and upstream model/loss/optimizer implementations.
- Do not start, stop, or resume training unless the user explicitly requests it.
