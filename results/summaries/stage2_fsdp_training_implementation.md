# Stage 2 FSDP实施与独立验证报告

更新时间：2026-10-21T16:05:26.037893+08:00

正式训练0更新。技术验证通过，正式启动仍待用户计划第6.4节要求的单独确认。EMA、CFG、LoRA和CPU优化器卸载关闭。当前是通用冻结Qwen替代Stage 1 VLM的Stage 2近似实验，不能称完整Buffalo复现。

## 实现结果

项目侧新增父调度器、worker、全局采样/有界预取、共享预算、CPU卸载DCP、分阶段评估及配置；固定2.1上游未修改。入口支持--config、--gpus 2,5、--run-id、--budget、--resume、--verify-only及--preflight-only。完整合同和可执行命令见docs/stage2_nano10k_fsdp.md。

全局batch32，每rank1×16，epoch尾批每rank8；全5epoch50000呈现、1565更新。每micro同步reduce-scatter，完整DiT与Connector、显式MoE辅助loss、FSDP全局clip、独立posterior/noise/time随机流。验证全500/rank250，100更新区间按样本加权日志及TensorBoard。只保存500/1000/1500/1565，CPU完整分片/原子提交，不覆盖或额外保存。

父调度在完整保存后退出训练worker，双卡model-only采样、GPU2 decoder-only、无GPU CPU指标和四视图，再完整恢复。中断先补评估；保存残留保留至incomplete目录，日志原始副本保留至interrupted_logs目录，活动日志回退到最近完整点，TensorBoard恢复清理重算步。显式恢复须用本run最新完整点；首次保存前支持--resume auto从初始化重算。

预算按实际指派设备数包含加载/保存/等待；父进程预留60 GPU秒用于轮询、SIGTERM及kill/reap，worker使用提前截止时间。收到SIGTERM触发finally，仅终止本实验进程组。正式启动状态持久化，不能在切换到采样时误写未开始。

## 验证证据

- 原18项CPU回归通过，CUDA未初始化；新11项CPU回归通过，日志为results/stage2_nano10k/stage2_nano10k_fsdp_verify_v4/parent_cpu_regression.log。
- tiny实际上游MoE和activation checkpoint与单进程参考最大参数误差1.955777406692505e-8；新进程下一更新全参数/Adam误差0。
- v4真实3B20次更新、640唯一UID、覆盖最长uid_91425。第2–20次含I/O平均16.2693566169秒。
- 第20次DCP保存，连续第21参考；新两个rank恢复第20次再做第21次，全本地参数/Adam逐元素比较最大误差0。分别3846个张量记录、合计9166438858个元素，atol1e-6/rtol1e-5不变；loss/UID/数据位置/CPU与CUDA及训练随机流一致。
- save→exit→sample→decode→CPU→fresh resume闭环完成。两例有效mesh、8张固定视图、指标文件齐全；语义审核pending，不据两例或F-score宣称学会条件生成。
- 全部GPU阶段≥2GiB余量，最小GPU5训练2.4274414061GiB；保存和完整恢复峰值通过。限额按真实CUDA总量计算，未提高cap掩盖问题。

原始GPU证据results/stage2_nano10k/stage2_nano10k_fsdp_verify_v4/verification_report.json保持不变。v4后只改父调度与CPU测试；AST审计列出state/child/formal/main及新增辅助函数，数值worker、model、FSDP、DCP、配置/数据/缓存逐文件指纹未变。当前组合报告results/summaries/stage2_fsdp_readiness.json绑定当前完整合同，复用未变数值核心证据，不冒称父调度修改重新GPU验证。

## 失败历史与修正

v1：tiny梯度参考通过，显存报告漏device参数，实际3B更新前失败。v2：tiny新进程标量一致，比较文件的连续torch zip读取失败，改为长度分帧并做CPU roundtrip。v3：完整阶段闭环和余量通过，但3B下一更新Adam exp_avg超容差；未宣称通过。BF16 SDPA有限反向诊断发现默认重复梯度最大差0.000244140625/0.00048828125，确定性模式重复相同。固定deterministic_algorithms=True、CUBLAS_WORKSPACE_CONFIG=:4096:8、TF32关闭后v4全量比较通过。allocator为max_split_size_mb:256；平台不支持expandable_segments，未依赖该设置。

所有失败、重启及诊断保留并计入新账本；旧历史probe单独保留。未重下载数据、重编码文本/posterior或重跑VAE gate；缓存预检复用原11000条成功证据及22k文件stat指纹。

## 资源与下一步

16GPU小时上限已耗2.1811，剩13.8189。仅完整1565更新预计14.1453GPU小时；含验证/保存/加载/200例过程监控预计还需18.075，加已耗共20.256GPU小时。解码边际时间沿用同内核7.05秒参考，启动/长期I/O/共享负载存在外推误差，详见stage2_fsdp_training_resources.md/json。

现有剩余额度不足按当前速度完成5epoch，不自动加预算、降验收或增保存点。最终候选/测试预算另议。当前无本实验GPU worker，账本无running条目；TensorBoard保留。下一步让用户基于上述具体结果确认正式启动及预算选择。

个人userdata配额仍未知，quota工具不可用；共享挂载总容量不代表个人额度。四份正式checkpoint约146.8GB，另需临时和验证产物空间。已有失败与验证大产物保留且不入Git。
