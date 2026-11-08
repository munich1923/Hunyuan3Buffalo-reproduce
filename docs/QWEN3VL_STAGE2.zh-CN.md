# 纯文本 Qwen3-VL Stage2

[English](QWEN3VL_STAGE2.md)

这是使用公开 Qwen3-VL 权重实现的 Buffalo 风格隐藏状态注入实验，不是使用官方
Hunyuan3D-VLM 权重的复现。本次只在本地实现；没有连接 H100、下载真实权重、运行真实
模型推理或启动训练。

## 修改范围

保留原图片基线、其配置和固定版本的上游 submodule。新增模块位于
`scripts/native_21dit/qwen3vl_stage2/`。继续使用官方 `hy3dshape/main.py`，官方
Diffuser、VAE、flow-matching loss、AdamW 构造和 LR scheduler 不变。
本实验使用本地派生类 `MaskedHunYuanDiTPlain` 扩展条件 padding mask，保留官方参数名称、
形状和预训练权重加载方式，不新增可训练参数，不修改上游文件。

本目录独立实现 Dataset、DataModule、运行状态 callback、标量记录和工具函数，不导入
`native21_official`，恢复契约也不包含图片实验脚本。两个实验仅共享已确认的数据清单、
几何文件和固定版本的上游实现。

```text
原始 caption → 可关闭的固定前缀 → 冻结 Qwen 文本隐藏状态
    → 可训练 4096→1024→1024 MLP → contexts["main"] → DiT
tokenizer attention_mask → contexts["main_mask"] → 每层交叉注意力
官方 surface NPZ → 冻结的官方 VAE（sample_posterior=True）→ 几何目标
```

Qwen 只接受文本。数据集不打开图片、不输出 `image/mask`；conditioner 拒绝图片、mask、
额外输入及图片/视频占位 token。使用 AutoTokenizer，不使用图像 processor。
通过 `vlm.model(...)` 提取输入序列最后一层隐藏状态，训练时不生成自然语言回答，也不
计算语言 logits。只有 Qwen 位于 `no_grad` 中，MLP 保留梯度。

官方 Hunyuan pipeline 仍参与原有模型初始化和精度转换，但本次不调用其图片采样 API，
也不实现完整的文本生成三维推理 API。自动评估关闭；独立的前缀对比工具只生成文字。

## 配置与数据

- `configs/qwen3vl_stage2_abo_hssd_20261105.yaml`：新模型、数据和 Trainer 配置。
- `configs/qwen3vl_stage2_abo_hssd_20261105.json`：源码、模型和数据恢复契约。
- `environment/qwen3vl-stage2-requirements.txt`：独立 H100 shape 环境需要的额外依赖版本；
  其余运行时沿用[环境说明](../environment/README.md)。

保留现有 8,626 个训练 UID 和 479/479 验证/测试划分，保留对象有放回采样及官方表面点
采样。此前因缺少自身 UID 图片排除的记录不补回，不修改几何坐标。读取配对清单只是
为了固定已接受的 UID 和几何身份，纯文本数据链路不需要访问清单中的图片文件。

默认 caption 来源是原数据准备代码生成的
`assets/abo_hssd_v1/metadata/candidates.jsonl`；尚未核查 H100 文件现状。相关记录格式：

```json
{"uid":"abo_<sha256>","source":"ABO","raw_sha256":"<sha256>","metadata_revision":"288b06111bc4d06800293ab087989f852a632af2","captions":["a wooden chair","a chair with four legs"]}
```

严格按 UID、来源和原始资产哈希匹配，使用 metadata 原文，不使用去重 manifest 中合并的
别名 caption。缺失、重复 UID、空描述或身份不符都会报错，不静默替补。如果 H100 只有
原始 metadata CSV，需要另行审查并准备这种 JSONL 或配置实际文件路径；本代码不自动
改换来源或重建划分。

每次抽到对象，都从该对象的非空 caption **条目**中均匀随机选一条，保留重复条目及原始
字符串。caption 使用 seed=`20261027` 的独立 Python RNG，在已提交样本边界保存其状态，
不影响 UID、点采样和训练随机流。同一个 UID 多次出现可以抽到不同描述；恢复时可精确
重放选择，包括已预取但尚未提交的数据。不额外定义 epoch，也不把每个 UID 固定到一句话。

暂定 user 消息为：

```text
Create a 3D object that faithfully matches the following description:
{caption}
```

`task_prefix: ""` 可关闭前缀。使用 Qwen 自带 chat template，并设置
`add_generation_prompt=True`。保留全部有效输入 token，允许变长序列。
`max_length: 2048` 是本项目的显式长度保护，超长时报错，不静默截断；这不是 Buffalo
规定。官方 DiT 的 `context_dim=1024`、`use_attention_pooling=false`，因此
`text_len=1370` 不要求把新条件重采样成 1370 个 token。

不同长度的 caption 通过左侧 padding 组成 batch，不截断原文。Qwen 接收对应的
attention mask；`encode_text()` 返回 `[B,L_max,4096]` 隐藏状态和 `[B,L_max]` 布尔有效
位置 mask。MLP 输出 `[B,L_max,1024]`，补齐位置清零，并通过 `main_mask` 将有效位置
逐层传给 DiT 的 SDPA 交叉注意力；布尔 `True` 表示允许关注。Qwen 原有因果注意力和
DiT 的几何 self-attention 不变。全空 mask 或形状、类型不符会报错。

mask 路径允许 PyTorch 自动选择注意力后端，必要时使用 math 回退。实际 H100 上的后端
兼容性、显存和速度仍需另行指示后检查。没有加入固定长度池化或 resampler，也不把
生成回答的 token 加入条件。

## 训练参数、精度与恢复

Qwen 冻结并保持 eval、BF16。封装层保护父模块 `.to()`/`.half()` 转换中的 backbone
精度，也保留 FP32 rotary buffer，避免先转 FP16 再转回造成信息损失。
DiT/VAE 保留原 FP16 权重、Trainer BF16 AMP 行为；MLP 跟随原 pipeline 的 FP16 转换。
MLP 为 Linear(4096,1024)、GELU、Linear(1024,1024)。

打开 `optimizer_cfg.train_image_encoder=true` 后，官方代码把 conditioner 中可训练的
MLP 加入优化器，冻结的 Qwen 不加入。DiT 和 MLP 初始 LR 均为 `1e-5`。MLP 普通权重
decay 为 `0.05`、bias 为 `0`；DiT 继承 `0.01`。共用原 warmup 50、总步数 `1e9` 的
cosine 调度。保留 `drop_ratio=0.1` 属性及条件丢弃关闭的既定行为。

提供的 YAML 默认仍是 microbatch 1、累积 8，但实现支持任意正整数
`dataset.params.batch_size`，由 DataModule 直接传给 DataLoader；累积次数通过
`training.update_every` 独立配置。当前单 GPU 的有效 batch 是两者乘积；Trainer 不会
自动调整它们。例如 batch 2、累积 4 的有效 batch 也是 8。两个值只在 YAML 中配置，
JSON 不再重复维护。当前可恢复采样机制仍要求 `num_workers=0`，这与 batch_size 无关。

每 3000 更新存 checkpoint、每 500 更新记录标量保持不变。
原 `train/grad_norm_l2_pre_clip` 只计算 DiT；新增
`train/connector_grad_norm_l2_pre_clip` 单独计算 MLP，额外写入
`connector_scalar_updates.jsonl`。有效 batch/window loss 仍在 `scalar_updates.jsonl`。
两个 norm 均在梯度累积完成后、裁剪前计算。

使用标准完整 Lightning checkpoint，保存 Qwen、MLP、DiT、VAE、优化器、调度器、标量
窗口和已提交的数据/RNG 状态。因此每份完整 checkpoint 会额外包含约 **17.5 GB** 的冻结
Qwen 权重，没有自定义裁剪 checkpoint。恢复要求 metadata、配置、代码和模型文件哈希
匹配。图片基线及已废弃实验的 checkpoint 不兼容；新配置写入独立结果目录。
恢复状态版本 2 分别记录已消费样本数和 DataLoader batch 数，Lightning 的循环游标
使用后者。当前固定 batch、单 GPU 下，U 次更新对应 `U*A` 个 batch、`U*A*B` 个样本。
仅保存已提交的采样状态，因此恢复会重新产生已预取但未消费的样本。改变 batch 或累积
设置不属于原实验的精确恢复；此前本地文本实现的旧状态格式也不兼容。

## 本地验证与后续 H100 检查

本地检查需要 CPU torch/torchvision 和参考 shape 数据依赖，不需要真实预训练权重：

```powershell
python -B tests/test_qwen3vl_stage2.py -v
```

测试覆盖混合长度文本 batch、拒绝图片、MLP 梯度、Qwen 冻结、精度保护、同 UID metadata
匹配、保持官方 UID/几何流不变、预取恢复及官方优化器分组。将带 mask 的批量 DiT 输出和
梯度与逐样本、无 padding 的官方调用对比，检查参数严格加载及 padding 不影响结果。
独立进程验证 batch/累积组合 `(1,8)`、`(2,3)`、`(8,1)` 的连续执行与 checkpoint 恢复一致，
并通过禁止导入图片实验模块的独立检查验证隔离。
另用随机初始化的微型 Qwen 架构验证 Transformers 的纯文本 API。这些检查不能证明真实
8B 模型数值稳定，也不能证明生成质量。

后续经单独指示在 H100 检查时，需准备 `Qwen/Qwen3-VL-8B-Instruct`，固定 revision 为
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。保留标准 HF snapshot 目录，或
`snapshot_download(local_dir=...)` 产生的
`.cache/huggingface/download/*.metadata` 回执；契约检查 revision 回执并记录使用到的
权重/tokenizer 文件哈希。加载强制使用本地文件，不配置 `device_map=auto` 或跨 GPU 分发。

获得该推理检查的单独指示后，从项目根目录运行独立工具：

```bash
CUDA_VISIBLE_DEVICES=GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m scripts.native_21dit.qwen3vl_stage2.compare_prefix \
  --config configs/qwen3vl_stage2_abo_hssd_20261105.yaml \
  --output results/qwen3vl_prefix_review
```

默认取验证集 ABO/HSSD 各 4 个对象，同一个对象使用同一条首个非空 caption 比较有/无
前缀。JSON 和 Markdown 记录贪心生成的文字、隐藏状态和 connector 输出的形状、有限值、
token norm 及池化隐藏状态余弦相似度。MLP 尚未训练，其输出仅用于接口检查，不能作为
训练后效果指标。审阅结果后再决定正式训练前缀。工具不启动训练或三维评估，不挂接训练
回调。

正式训练仍需单独决定，使用官方 `main.py`、新 YAML 和 seed=`20261101`，只使用授权的
物理 GPU1。本次没有新增 launcher、supervisor、checkpoint 监视、自动暂停或评估调度。
