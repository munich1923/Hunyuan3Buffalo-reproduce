# Stage 2 实施记录

本轮已实施纯文本 VLM→Connector→预训练2.1 DiT的独立训练代码。当前属于使用通用Qwen的Stage 2近似实验，没有完成论文Stage 1；不使用源几何/图片条件、DINO或旧编辑权重。

## 已完成

- 配置：configs/stage2_nano10k.yaml；完整说明：docs/stage2_nano10k.md。
- 固定17分片获取、源mesh检查、去重分组、原类别比例10000/500/500抽样；后台任务正在获取数据，最终清单尚未ready。
- 纯文本变长hidden缓存、FP32 mean/logvar缓存，内容指纹及缓存失效。
- 纯生成模型、完整DiT更新、显式MoE aux、实际尾批归一化、block checkpoint。
- 独立encoder-only/decoder-only VAE模块；生成入口不构造encoder/pre_kl。
- 5epoch/1565更新、100更新loss日志、500更新完整checkpoint且全部保留、最终1565保存。
- TensorBoard事件与结构化日志、完整状态恢复、预算/设备/数据gate、官方scheduler采样。
- 110物体VAE gate、固定监控、100val/100test、错/空条件、具名语义review和checkpoint锁定接口。

## 验证证据

- results/summaries/stage2_cpu_tests.json：18项CPU测试通过，CUDA未初始化。
- results/stage2_nano10k/preparation/cpu_tests.log：完整测试输出。
- results/summaries/stage2_token_selection_audit.json：17000/17000通过，选取条件长度44–186，中位数92，正文逐条还原。
- results/summaries/stage2_geometry_smoke.json：uid_120/33000/66000真实mesh归一化与81920×7采样通过。
- results/summaries/stage2_vae_checkpoint_interface.json：真实VAE单体checkpoint的encoder/decoder参数名及形状无缺失；meta+mmap检查，无GPU。
- pip check通过；git diff --check通过；上游工作树保持干净。
- TensorBoard远端127.0.0.1:6006，HTTP200；日志在results/stage2_nano10k/tensorboard_server.log。

CPU测试包括原生小DiT前向/梯度一致性、显式MoE与checkpoint梯度、累积/尾批、官方scheduler、真实tokenizer、AdamW CPU卸载数值、连续/续跑一致、缓存失效、GPU占用/预算拒绝、固定视角和event写入、decoder-only对照。

## 后台任务和依赖

- CPU准备状态：results/stage2_nano10k/preparation/dataset_status.json。
- 启动记录：results/stage2_nano10k/preparation/dataset_process.json；supervisor.log、dataset.log。
- 下载日志及校验：assets/manifests/stage2_nano10k/downloads/。
- TensorBoard PID/端口：environment/stage2_tensorboard_process.json。
- 新增tensorboard2.18.0及依赖；setuptools由84.0.0固定为80.9.0以恢复pkg_resources兼容，仍在userdata内shape环境。environment/stage2_shape_packages.json记录环境。
- 支持同步CPU AdamW卸载作为可选后端，无需DeepSpeed；GPU性能与峰值仍未验证。

## 未运行与边界

没有VLM/VAE GPU缓存、VAE重建质量gate、GPU资源probe、训练、mesh生成结果或条件生成成功证据。8张H100均有占用，不使用、不终止其他任务。

新预算未批准；environment/stage2_budget_approval.template.json默认approved=false。需要独占GPU后先做缓存/VAE gate/probe，再由用户确认新预算。旧编辑37363.49秒保留不变。

本地自动隐藏SSH转发进程被自动审批拒绝（blocked by policy），未建立本地转发。可手动运行：
ssh -o BatchMode=yes -N -L 6006:127.0.0.1:6006 h100
随后访问http://127.0.0.1:6006。没有训练时没有loss曲线。

不得将代码/CPU检查通过写成条件生成成功，更不能称为编辑成功。
