# Training environment

The reference run used Linux x86_64, Python 3.10, CUDA 12.4, PyTorch `2.5.1+cu124`, torchvision `0.20.1+cu124`, PyTorch Lightning `1.9.5`, NumPy `1.24.4`, OpenCV headless `4.10.0.84`, and OmegaConf `2.3.0`.

The official Hunyuan3D-2.1 submodule contains its upstream dependency instructions. Install the shape-training dependencies into a dedicated Linux environment, then record the resolved package set and verify `pip check` before training. Do not install a second OpenCV distribution alongside `opencv-python-headless`.

The reference YAML keeps the official Hunyuan3D model construction and BF16 Trainer setting. Runtime model weights, the paired ABO/HSSD payloads, CUDA toolkit/driver, and the exact wheel artifacts are external inputs and must be provisioned separately.
