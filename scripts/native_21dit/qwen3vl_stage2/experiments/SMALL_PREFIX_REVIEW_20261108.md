# 小实验：Qwen3-VL Stage2 前缀对比

- 状态：H100 推理已完成；用户已选择采用任务前缀。正式训练尚未启动。
- 目的：比较同一条 ABO/HSSD 原始 caption 在有、无固定任务前缀时的 Qwen 文本输出与条件接口统计。
- 输入：仅文本；验证集 ABO、HSSD 各 4 个 UID，每个 UID 使用首条非空原始 caption。
- 参数：seed `20261101`，贪心生成，最多 128 个新 token；固定 Qwen3-VL revision 由 Stage2 YAML 指定。
- 代码：`scripts/native_21dit/qwen3vl_stage2_prefix_review/compare.py`。
- 结果：H100 `results/qwen3vl_stage2_abo_hssd_20261105/small_prefix_review_20261108/` 下的 `comparison.json` 与 `comparison.md`。
- 控制台日志：H100 `results/qwen3vl_stage2_abo_hssd_20261105/small_prefix_review_20261108.log`。
- 本地只读镜像：`results/qwen3vl_stage2_abo_hssd_20261105/small_prefix_review_20261108/` 及旁边的同名 `.log`。
- 运行环境：Python 3.10、PyTorch `2.5.1+cu124`、Transformers `4.57.1`、PyTorch Lightning `1.9.5`；H100 GPU UUID `GPU-47f8e8ad-bf0a-79d9-2e95-645bf2ee0609`。
- 结果：8/8 样本完成；hidden/context 有限值检查全部通过；平均 pooled hidden cosine 为 `0.9913`。
- 观察：带前缀响应更常采用 3D 规格说明形式；无前缀响应更常采用通用商品描述形式。两组都可能增加 caption 未提供的尺寸或材质细节。
- 数据说明：验证集 479 条；抽样 UID 与 caption 来源、原始资产哈希校验通过。metadata SHA-256 为 `48897fb9cf6d1ebf85c4e6027598dfdef031c63b4a1c5792550b798d9f171b2b`。
- 主训练设定：使用官方 `main.py` 从预训练权重新开；当前 YAML 为 batch size 1、`update_every=8`、上限 1,000,000,000 更新、每 3,000 更新保存 checkpoint、每 500 更新记录 TensorBoard 标量。正式训练尚未启动。

Qwen 条件隐藏状态和 connector 统计只用于接口检查；connector 尚未训练，因此不能据此判断三维生成质量。日志保留了首次导入路径失败和随后成功重试的记录；运行使用离线模式，没有下载模型或数据。Transformers 对 generation config 中 `temperature`、`top_p`、`top_k` 的忽略发出 warning；调用显式使用 `do_sample=False`，按 greedy 解码。
