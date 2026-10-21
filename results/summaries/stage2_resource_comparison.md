# Stage 2 资源探测结果（2026-10-21）

数据准备完成：10000训练＋500验证＋500测试，11000文本与posterior缓存完整，VAE gate通过。两种探测均为有限资源检查，正式训练尚未启动。

| 项目 | GPU2 CPU AdamW卸载 | GPU2＋GPU5 FSDP FULL_SHARD |
|---|---:|---:|
| 全局batch | 32 | 32 |
| 每卡microbatch／累积 | 1／32 | 1／16 |
| 更新次数 | 5 | 5 |
| 后4次平均秒／更新 | 53.5351 | 13.4390 |
| 训练峰值reserved | 32.11 GiB | 36.82／36.75 GiB |
| 全流程峰值reserved | 32.11 GiB | 45.43／42.49 GiB |
| 完整checkpoint大小 | 36.667 GB | 36.674 GB |
| checkpoint保存含校验 | 63.57秒 | 60.04秒 |
| checkpoint恢复 | 本probe未恢复；optimizer CPU单测通过 | 实际3B恢复73.60秒 |
| 8例验证 | 2.71秒 | 1.87秒（两rank重复同8例以保持collective次数） |
| 50步采样 | 6.70秒 | 8.53秒 |
| VAE解码 | 7.49秒 | 7.05秒 |

两卡完整探测于11:50:35 +08:00退出0，已释放GPU。小尺寸实际上游MoE＋block activation checkpoint：两次更新与单进程全局batch参考最大误差2.1234e-7；分片保存恢复后下一次更新全参数完全相同。实际3B checkpoint所有文件SHA256校验、DCP严格加载与每个非空本地参数及Adam状态前4值核对通过；尚未进行3B新进程续跑后的下一更新全参数对比，不能把该项写成已验证。

GPU2／GPU5准入空闲59005／48673 MiB，allocator cap53104／43805 MiB，保留初始空闲10%工程余量。训练阶段reserved约36.8 GiB；保存恢复及采样阶段峰值更高，GPU5全流程reserved为42.494 GiB；按同型号卡CUDA总量与nvidia-smi口径换算，实际allocator上限约42.53 GiB，距离峰值仅几十MiB。43,805 MiB是准入记录的名义预算，不能直接当实际allocator字节上限。共享任务未来增长仍会改变余量，正式启动必须重新准入；不据此承诺长期无OOM。正式FSDP入口应先减少保存/恢复阶段临时张量与allocator缓存堆积（全流程最大allocated约32.37 GiB，明显低于reserved），再专门验证I/O阶段余量；本轮没有修改成功probe或伪称该优化已完成。

本次两卡更新墙钟约快3.98倍；但两次探测共享负载并不受控，不能解释为纯FSDP加速比。按13.439秒乘1565更新，纯更新约5.84小时、两卡约11.68 GPU小时；这是5更新外推，不是完整预算，尚不含预检、验证、监控生成、解码、保存、语义审核与最终验证/测试。CPU卸载对应纯更新约23.27小时。

两卡只有有限probe入口，尚未声称完整FSDP正式训练入口（分布式epoch数据调度、每100loss与TensorBoard、每500完整保存和验证监控）已经交付。正式启动前需补齐该入口、确认预算，并重新核验显存；当前不自动训练。不把可解码mesh或5次loss当条件能力证明。

证据：
- results/stage2_nano10k/probe_gpu2_cpuoffload_20261021/probe_report.json
- results/stage2_nano10k/probe_fsdp_gpu2_gpu5_20261021/probe_report.json
- results/stage2_nano10k/probe_fsdp_gpu2_gpu5_20261021/tiny/tiny_report.json
- results/stage2_nano10k/fsdp_gpu2_gpu5_20261021/status.json
- results/summaries/stage2_cpuoffload_probe.json
- results/summaries/stage2_fsdp_probe.json
