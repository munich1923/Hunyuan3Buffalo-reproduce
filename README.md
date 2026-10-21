# Hunyuan3Buffalo-reproduce

Hunyuan3D Buffalo 的项目侧近似复现实验。当前工作是 Stage 2 条件生成：原始文本 → 冻结 Qwen2.5-VL → 新 Connector → 预训练 Hunyuan3D-2.1 完整 DiT，从纯高斯噪声生成三维几何。

目前缺少完成 Buffalo Stage 1 后的 VLM 权重，因此称为 **Stage 2 近似实验**，不宣称完整复现或官方训练实现。

## 当前状态

- 正式双卡 FSDP 入口及独立验证已完成，正式训练尚未启动。
- 20 次真实更新覆盖 640 个唯一 UID；完整 3B 模型新进程续跑的全部本地参数及 Adam 状态比较最大误差为 0。
- EMA、CFG、LoRA 和 CPU 优化器卸载关闭。
- 训练预算与最终候选/测试评估预算分开；正式启动及预算决定见交接记录。

执行方案：[docs/stage2_nano10k_fsdp.md](docs/stage2_nano10k_fsdp.md)。
验证报告：[results/summaries/stage2_fsdp_training_implementation.md](results/summaries/stage2_fsdp_training_implementation.md)。
资源估算：[results/summaries/stage2_fsdp_training_resources.md](results/summaries/stage2_fsdp_training_resources.md)。
最新状态及工作约定：[handoff.md](handoff.md)、[agent.md](agent.md)。

## 代码与实验环境

服务器实验工作区：`/root/userdata/zc/hunyuan3d-buffalo-1.0-editing`。

本仓库保存项目代码、配置、环境定义和小型实验报告。模型权重、数据、缓存、虚拟环境及大型 checkpoint 留在服务器，由 `.gitignore` 排除。

固定上游为 [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)，提交 `82920d643c0dc2f7bfd7255f45f62d386edfe60c`。`src/Hunyuan3D-2.1/` 保持独立仓库，目前按既有规则排除在主仓库之外，尚未注册为 submodule。

本地代码副本通过 SSH 获取服务器提交，再使用本地 GitHub 认证推送；服务器不保存 GitHub 凭据，不转发 SSH agent。

## 历史编辑实验

此前编辑近似实验未证明可靠的未见物体指令编辑能力，与当前 Stage 2 条件生成实验分别报告。

历史报告：[results/summaries/final_report.md](results/summaries/final_report.md)；模式说明：[results/summaries/evaluation_modes.md](results/summaries/evaluation_modes.md)。大型网格和对比图保存在服务器。
