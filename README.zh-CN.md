# Hunyuan3D-2.1 ABO/HSSD 训练复现

[English](README.md)

新增的纯文本 Qwen3-VL Stage2 实现见[条件端说明](docs/QWEN3VL_STAGE2.zh-CN.md)：
替换图片 conditioner，并在本地扩展 DiT 条件 padding mask；保留官方训练入口、损失和
优化器，真实 H100 验证尚待进行。

本仓库保存 ABO/HSSD 单图条件 Hunyuan3D-2.1 训练实验所需的项目代码和固定配置。官方 Hunyuan3D-2.1 源码以 Git submodule 形式固定在提交 `82920d643c0dc2f7bfd7255f45f62d386edfe60c`。

## 复现输入

仓库跟踪代码、划分与配对清单、配置文件；不跟踪模型权重、图片和几何数据。实验配对数据约 35.3 GB，Hunyuan3D-2.1 DiT/VAE 与 DINO 权重约 8.6 GB。请从获准的数据和模型来源获取文件，放到配置指定的位置，并在训练前核对 SHA256。

清单包含 8,626 条训练配对、479 条验证配对和 479 条测试配对，保留原始划分以及同 UID 图片/几何配对。记录中的数据路径指向实验使用的 H100 工作区；迁移到其他位置时，需更新路径并重新生成索引 ready 校验文件。

## 官方代码与项目适配代码

- 上游项目提供 `hy3dshape/main.py`、模型、损失、优化器、图像编码器和官方表面点采样实现。
- `scripts/native_21dit/Abo_Hssd_datapre/` 包含 ABO/HSSD 元数据、几何和图片准备流程。`prepare_geometry.py` 调用固定版本的上游渲染器和 `watertight_and_sample.py`。
- `scripts/native_21dit/native21_official/` 包含 ABO/HSSD 配对数据适配器、不可变索引校验、checkpoint 数据/RNG 状态管理和按优化器更新对齐的 TensorBoard 标量记录。上游加载器使用另一种 WebDataset 目录格式，因此该适配器是训练所需的。
- `configs/` 包含数据准备设置和官方 `main.py` 训练配置。

当前训练配置从官方预训练权重初始化，microbatch 为 1、梯度累积为 8，每 3,000 次优化器更新保存 checkpoint，每 500 次更新记录 loss 和裁剪前 DiT 梯度范数。配置没有自动网格评估或自动停止逻辑。

ABO/HSSD 原始资产来自单独的数据主机和 H100 工作区。仓库中的元数据下载器不会从公开地址下载约 35.3 GB 的图片和几何数据。已准备的数据仍保存在 H100；在其他地方重建数据需要相同的授权源资产和处理回执。

## 获取上游代码

递归克隆 submodule：

```powershell
git clone --recurse-submodules https://github.com/munich1923/Hunyuan3Buffalo-reproduce.git
```

普通克隆后初始化上游代码：

```powershell
git submodule update --init --recursive
```

训练需要配置好的 Linux/CUDA 环境。仓库保留了实验 H100 工作区路径，不能直接在 Windows 上运行。

## 当前训练配置

- `configs/native21_official_abo_hssd_1b_20261103.yaml`
- `configs/native21_official_abo_hssd_1b_20261103.json`

训练配置使用独立的新输出目录，不会覆盖服务器上正在运行的实验。清理后的项目从官方预训练权重开始新实验；由于源码集合的 contract hash 不同，不能用它恢复活动实验的 checkpoint。

`configs/abo_hssd_v1.json` 配置数据识别与准备流程，包括元数据/上游版本、数据划分和几何规则、准备目录；它不是 Hunyuan3D 训练配置。配对文件格式见[数据目录与文件格式](docs/ABO_HSSD_DATA_LAYOUT.zh-CN.md)。
