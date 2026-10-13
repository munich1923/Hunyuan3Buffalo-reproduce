# 实验交接

## 当前阶段
用户已授权执行近似复现。源码已拉取并固定；独立环境安装和模型/数据下载正在准备。
尚未运行完整模型、训练或评估。脚本已经写入，但除语法检查外尚待环境和实际权重验证。

## 工作区
/root/userdata/zc/hunyuan3d-buffalo-1.0-editing；本地通过 ssh h100 操作。
工作约定见 agent.md。所有环境、数据、权重、日志和结果保存在本目录。

## 固定路线
- 官方 Hunyuan3D 2.1 提交：82920d643c0dc2f7bfd7255f45f62d386edfe60c。
- Qwen2.5-VL-7B-Instruct + 四视角几何渲染 + 隐藏状态 MLP。
- 冻结 VLM/VAE/DiT 主干；先训练 MLP，后训练 MLP + 注意力 LoRA。
- 几何添加/删除；不安装 Paint 流程。
- 训练 1024 条、验证 128 条；来自分别 512/64 个原始添加配对及其反向删除条目。
- Nano3D v1 实际没有直接 remove 标注；只使用 add，反向构造 remove 并保留 provenance。
- 独立测试 Edit3D-Bench 添加/删除各20例，排除训练几何重复。
- 单卡累计 24 GPU 小时封顶；VAE 重建与8样本拟合门槛失败则停止。
- 本实验是开源替代的小规模机制检验，不是官方权重或指标复现。

## 已完成与证据
- src/Hunyuan3D-2.1 源码干净，提交固定；environment/hunyuan3d21_source.json 和 logs/source_*.log。
- 镜像 API 与真实 payload Range 请求已验证；直连 Hugging Face 失败。
- 权重、数据集的官方 repo 身份和 revision 已固定；下载完成后逐项 SHA-256/大小验证。
- Shape config 已核验：4096x64 latent，1024 条件宽度，1370 条件 tokens。
- 编写下载、数据筛选、EGL渲染、缓存、模型适配、训练、评估和GPU预算脚本。
- 当前只确认 Python 脚本可编译，不能视为模型或训练测试通过。

## 进程与状态
- environment/install_pid.txt：实验环境安装进程。
- environment/download_pid.txt：模型/数据下载进程；environment/data_download_pid.txt 为额外数据下载进程。
- logs/install_launcher.log、logs/download_launcher.log、logs/data_download_launcher.log。
- 成功状态分别记录 environment/install_status.json、download_status.json。
- models/ 中的 *.partial 可能为分段稀疏文件；文件大小不能用作完成百分比，必须以下载收据的 verified 状态为准。

## 下一步
1. 检查并修复安装和下载，获得两个环境的 pip check 与依赖快照。
2. 运行 scripts/run_pilot.py --prepare-only，完成数据QC和渲染；检查 results/data_review/ 的30个独立物体配对。
3. 用 scripts/run_pilot.py 运行张量合同检查、VAE门槛、缓存、8样本拟合、连接器、LoRA、对照评估。
4. 根据实际状态更新此文档并提交。完整命令和假设见 environment/README.md、method_notes.md。
