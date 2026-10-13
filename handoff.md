# 实验交接

更新时间：2026-10-13T18:05:27.488117+08:00

## 当前状态
- 正式连接器已完成500步；LoRA正在运行，最多2000步。最近已落盘探针step=380，train=0.4607116，val16=0.4391631。
- 尚未开始正式benchmark生成；未见物体的几何编辑有效性尚无结论。
- 唯一工作区：/root/userdata/zc/hunyuan3d-buffalo-1.0-editing。本地只通过ssh -o BatchMode=yes h100操作。
- 当前runner和子进程以environment/pilot_pid.txt及environment/run_state.json current.pid为准。不要重复启动，不能绕过pilot.lock运行GPU。
- 已完成阶段累计GPU分配9984.09秒；正在运行阶段尚未计入此字段。总预算86400秒，一次只使用一张空闲H100。

## 已完成
- 官方源码src/Hunyuan3D-2.1固定82920d643c0dc2f7bfd7255f45f62d386edfe60c，上游无补丁。
- 独立Conda环境environment/venvs/shape与vlm；torch2.5.1+cu124，Transformers分别4.46.0/4.51.3。pip check通过，freeze/conda-explicit保存environment/。
- 计划30项模型/数据下载全部verified；含benchmark共112份大小/SHA收据，logs/downloads/。HF直连失败，使用固定revision镜像和官方备用来源；临时网络中继已关闭。
- 官方VAE、Qwen最后层状态、MLP、源latent DiT前向/反向及冻结/LoRA合同检查通过。
- 正式首训练源uid_66677_add的VAE gate通过：CD平方和0.0004055963，F-score@0.05=0.9963859。不能解释为全部数据VAE保真保证。
- Nano3D分片0/33/66，共3000候选；1798独立add配对通过自动QC。删除来自反向构造，并有provenance。
- 训练1024条=512独立配对两方向；验证128条=64配对两方向。源/目标共同归一化，禁止独立归一化或ICP；跨split几何哈希交集0。
- AI agent实际查看46候选四视角，排除16，最终训练前30个独立配对均有观察，其余482配对只有自动几何QC。不是用户本人审核。
- Edit3D-Bench固定add/remove各20条；全量render及Shape/Qwen缓存完成，六目录数量/shape/有限性审计通过。test另含shuffled特征。
- 8独立add配对200步诊断：固定噪声train loss0.9926369→0.6293402，val16 loss1.0304584→0.6628609。通过预设下降10%诊断门槛；仅证明velocity loss拟合，不能称为几何记忆成功。
- 正式connector500步：train16 loss0.9658027→0.6386403，完整128val loss0.9231679→0.6159285。最佳探针0.6408666；不是从overfit初始化。random加载同一connector_initial.pt。
- CPU源网格首seed参照：F-score@0.05 add0.604/remove0.833；删除集需关注被删区域残留。正式三seed基线仍待运行。
- Matplotlib3.10.7已补入专用Shape环境，pip check再通过；缓存设置在本工作区，初步训练曲线已生成。

## 方法与边界
- Qwen2.5-VL-7B-Instruct：四视角源几何+指令，抽取所有有效输入token最后层hidden，不生成回答。
- MLP3584→1024→1024/GELU，pool至1370 tokens；源Shape-VAE latent固定，与4096×64 noisy target按token拼接，只取target预测。
- 冻结VLM/VAE/DiT主干；LoRA attention rank8/alpha16，同时训练MLP。训练10%no-op。
- 缓存/编辑解码使用FP32 VAE权重与输入+FP16 autocast，DiT BF16、adapters FP32。
- 9评估模式source/vae/native/random/connector/lora/no_source/noop/shuffled，3seeds、50步。edited无CFG，native使用默认CFG5、源单视角、不输入指令；不是完全对等的指令模型。
- 每模式最多3500秒、runner硬限3600秒，可能部分覆盖，必须报告attempted/expected。交替add/remove，先覆盖所有物体首seed；shuffled保持原manifest循环映射。
- CD/F-score基于共同坐标、16384表面点，无对齐优化。变化区域指标是ground-truth辅助表面proxy，不是真实人工part mask。
- 模式比较按共同样本及物体级三seed均值计算1000次bootstrap；只比较有效对，同时报告无效比例。不能给无效网格填0距离。
- 本实验是开源替代机制检验，不能宣称官方Buffalo模型/指标复现。

## 证据入口
- environment/run_state.json：逐阶段命令/提交/时间/GPU/退出码/累计预算。
- results/summaries/：完整原始/最终清单、视觉决策、完整缓存审计、训练报告、源参照。
- environment/training_overfit.json、training_connector.json；training_lora.json仅在该阶段结束后生成。
- results/checkpoints/：initial/best/latest adapters。大文件不入Git。
- logs/train_*.jsonl、stage_*.log；合成box纯FP16失败和EGL全黑帧失败均保留。
- 全量train render第999条曾全黑，重启新进程续缓存成功，失败时间已入预算，不删除此数据。
- environment/method_notes.md：全部方法差异；environment/README.md：环境操作。

## 下一步
1. 观察当前LoRA到结束，核对完整128val结果和实际steps。
2. runner自动串行完成9个benchmark对照；检查错误、完整性以及实际生成几何。
3. 仅CPU执行：
   environment/venvs/shape/bin/python scripts/compare_evaluations.py
   environment/venvs/shape/bin/python scripts/analyze_instruction_sensitivity.py
   environment/venvs/shape/bin/python scripts/build_comparison_sheets.py
   environment/venvs/shape/bin/python scripts/plot_training.py
   environment/venvs/shape/bin/python scripts/summarize_pilot.py
4. 实际查看比较图，记录AI观察和例子，严格区分接口通过、velocity拟合、未见编辑。
5. 完成最终报告、文档、Git提交，检查本实验后台结束。当前仓库无Git remote。

## 恢复
- 先读取agent.md与本文件，git status，检查/proc和run_state；尚在运行时只观察。
- 失败后修复实现问题，保留原日志/预算，确认runner退出后再运行environment/venvs/shape/bin/python scripts/run_pilot.py；成功阶段自动跳过。
- 已发布清单为权威；不要重划分已复核数据。data_review.json批准的是固定manifest SHA。
- 正式权重、VAE gate或8样本拟合失败停止扩大；connector完整val无提升不继续LoRA；预算不可绕过。

## 追加训练几何诊断
主流程完成后运行environment/venvs/shape/bin/python scripts/run_training_geometry.py，使用同一pilot锁和GPU累计预算，对overfit_best和lora_best各生成8个原始独立add训练样本（单seed/50步）。这是几何拟合检查，不修改已通过的velocity-loss gate，也不替代独立benchmark。
