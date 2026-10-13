# 实验交接

## 当前阶段
用户已授权执行近似复现。源码已拉取并固定；独立环境安装和模型/数据下载正在准备。
尚未运行完整模型、训练或评估。Python语法检查、基础环境中的数据合同测试通过；专用环境模型接口检查待执行。

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


## 2026-10-13 14:02 Asia/Shanghai 更新
- Shape DiT（7,366,389,768 bytes）、VAE及Qwen全部5个分片均已LFS SHA-256校验通过；下载收据在logs/downloads/。
- Nano元数据100,000条校验完成。分片0/33/66仍续传，尚未生成1024清洗清单。
- Shape专用环境torch2.5.1+cu124安装成功，依赖安装及torch_cluster安装成功。
- pip check发现ninja1.11.1.1旧manylinux1 tag不兼容新pip检查，改为ninja1.13.0并重试。尚未声称环境检查通过。
- 增加固定revision API、严格DINO state加载、缩放/平移不变的身份哈希。
- 8样本诊断改为8个独立add原始配对；先只缓存诊断训练/验证子集，通过后再完整渲染/缓存。
- 多视角EGL渲染计入GPU预算，采用同一张空闲卡。目前GPU累计0秒。
- 大数据传输采用远端直连、180秒连接刷新和aria2续传；Python依赖预取全部校验保存远端。国外PyPI元数据和wheel使用临时SSH中继。
- data_review.json尚未创建；不得未看图即批准。
- 临时中继仍需在准备结束后关闭；见environment/network_relay.json。


## 2026-10-13 14:12 Asia/Shanghai 更新
- 两个独立Conda环境均完成安装，pip check均为0；pip-freeze和conda-explicit已保存。
- 正式Shape环境verify_environment通过，DiT/VAE重新hash匹配；DINO439个tensor严格加载无missing/unexpected。
- H100 GPU0张量合同检查全部通过，累计分配54.34秒。
- 正式Shape环境数据合同测试通过。
- 正在运行官方权重的合成box/sphere完整接口诊断（EGL、VAE编码重建、Qwen隐藏状态、MLP/DiT前向梯度），这些不代表训练或泛化效果。
- Nano大分片仍续传；Edit3D-Bench mesh预取运行中。GPU训练未开始。


## 2026-10-13 14:23 Asia/Shanghai 更新
- 下载主任务30/30 verified。Nano分片0、33、66完整SHA匹配；总112份模型/数据/benchmark下载收据verified。
- Edit3D-Bench取得add/remove各20有效配对，另1配对因保留表面不足排除；排除记录保存。
- NVIDIA EGL实际渲染已跑通。原服务器强制Mesa变量仅对实验子进程取消。
- 官方VAE+Qwen+MLP+4096-token DiT源条件完整前向/反向在GPU0通过；Qwen hidden1350x3584，随机MLP velocity loss约0.973。不能据此宣称编辑有效。
- 合成盒子纯FP16 VAE重建F-score@0.05=0.2475，明显较差。已排队真实Nano源网格/FP32+autocast/尺度诊断；尚未运行真实数据VAE门槛或训练。
- 当前prepare-only在CPU数据提取/QC，完成后渲染30个训练配对；精度诊断等待同一pilot锁，绝不同时运行两张GPU。
- 临时SSH中继已关闭，environment/network_relay.json enabled=false，本地forward会话退出。

## 2026-10-13 视觉复核及正式训练前更新
- 两个独立环境安装、pip check、环境快照已完成；112份下载收据verified，中继已关闭。
- 官方权重完整接口通过；真实Nano uid_0 VAE FP32/autocast F-score@0.05=0.995，合成box重建较差。正式训练首样本VAE gate仍待执行。
- 3000候选自动清洗：1798独立add配对通过，1202排除。固定512/64个独立训练/验证配对各包含add与反向remove；测试40条。
- AI agent查看30个独立配对四视角图（不代表用户亲自人工确认），排除10个语义不符/难以确认配对，并保持其他训练位置和验证集不变、确定性补入10对。
- 原始清单保存在assets/manifests/before_visual_review；替换与观察见results/summaries/visual_review_*.json，完整清单可追溯。
- 正在重新渲染低光照v2图像并复核替换样本，尚未批准pilot。不会把样本语义抽查解释成整套数据语义保证。
- 评估VAE使用与正式缓存相同的FP32/autocast精度；connector验证loss无提升时停止扩大训练。
