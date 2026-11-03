# 复现输入与流程

[English](REPRODUCTION.md)

本仓库是在未修改的 Hunyuan3D-2.1 训练器外部提供的项目适配层，不包含完整数据集或预训练权重。

## 数据获取与准备

`scripts/native_21dit/Abo_Hssd_datapre/download_metadata.py` 下载固定版本的 ABO/HSSD 元数据 CSV，用来识别 TRELLIS-500K 记录。它不会下载源 GLB 或图片数据。原实验的大批量资产来自独立 NAS/数据主机，必须单独准备对应访问权限和许可。

在配置的工作区提供获准的原始 GLB 和固定源清单后，`scripts/native_21dit/Abo_Hssd_datapre/prepare_geometry.py` 会调用固定版本的上游 Blender 渲染器和 Hunyuan3D-2.1 `watertight_and_sample.py`。`build_manifest.py` 保留既有去重与训练/验证/测试划分。原图流程使用 `render_images.py` 和已有的 TRELLIS 源视图记录，为同 UID 选择一张固定图片，不会重新渲染新视角。部分 relay 工具依赖原实验的 H100/4090 主机别名和 NAS 目录。

`scripts/native_21dit/native21_official/prepare_abo_hssd_index.py` 校验并配对同 UID 图片与官方表面文件。仓库跟踪的 `assets/native21_abo_hssd_single_image/` 文件记录选定配对、数量和来源信息；图片、表面几何及回执数据被 Git 忽略，训练前必须按清单中的路径准备好。

接受的几何输入是官方坐标系下的上游 `<uid>_surface.npz`。不要用 `_vae_points.npy` 替代，不要再次归一化 XYZ，也不要在 UID 图片缺失时使用别名图片替补。

配对清单字段和运行时目录结构见[ABO/HSSD 数据目录与文件格式](ABO_HSSD_DATA_LAYOUT.zh-CN.md)。

`configs/abo_hssd_v1.json` 是 ABO/HSSD 数据准备契约，固定元数据/来源版本、划分随机种子和比例、几何处理参数及准备工作区路径。它配置数据识别与准备；官方训练设置在独立的 `native21_official_abo_hssd_1b_20261103` JSON/YAML 配置对中。

## 训练

准备 `configs/native21_official_abo_hssd_1b_20261103.json` 中列出并以 SHA 固定的模型文件。训练 YAML 从固定的上游 submodule 和本地 DINO 目录加载官方 ShapeVAE、HunYuanDiTPlain 和 SingleImageEncoder。训练入口是官方 `src/Hunyuan3D-2.1/hy3dshape/main.py`，使用 YAML 配置和随机种子 `20261101`，并在配置好的 Linux 工作区执行。

配置描述一个从官方预训练权重启动的新实验：最多 1,000,000,000 次优化器更新，microbatch 1、梯度累积 8，每 3,000 次更新保存 checkpoint，每 500 次更新记录 loss 和裁剪前 DiT 梯度 L2 norm，并写入独立的复现结果目录。上游加载器使用 WebDataset 多视图目录结构，和这里的 ABO/HSSD 同 UID 图片/几何配对格式不同，因此需要自定义数据模块。

服务器当前实验采用另一份源码 contract。清理后的仓库不用于恢复该活动实验的 checkpoint，应使用新的复现配置和官方预训练权重启动独立实验。
