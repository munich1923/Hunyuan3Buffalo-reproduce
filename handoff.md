# 当前 Stage 2 实施交接（2026-10-19）

用户已授权实施纯文本条件生成Stage 2近似实验。详见docs/stage2_nano10k.md、configs/stage2_nano10k.yaml和results/summaries/stage2_implementation.md。

代码已实现，18项CPU测试通过，CUDA未初始化；17000条候选prompt选择审计通过。上游固定提交仍干净。TensorBoard在远端127.0.0.1:6006运行，没有训练曲线。本地自动SSH转发被审批拒绝，需手动转发。

CPU数据准备于2026-10-19 21:11:59 +08:00失败（镜像429限流、备用直连不可达）。2026-10-20 10:38:33 +08:00已修复并续跑，runner PID 3468366：下载改为串行单连接、失败指数退避、禁止IPv6、取消每90秒强制重启；保留已校验包和aria2断点文件。实测014分片恢复至约62%，约6.4 MiB/s，尚未完成。原失败状态已归档为preparation/dataset_status.json.before_recovery_*。恢复证据见results/summaries/stage2_download_recovery.json。

2026-10-20 18:16左右低速恢复：原连接持续6–10 KiB/s。固定同文件range实测原请求122800 B/s，加download参数1010187 B/s，新请求1162154 B/s。下载器已加入每次续传新请求参数、64KiB/s aria2最低速、低于128KiB/s持续120秒的看门狗（以日志网络速率衡量，不依赖稀疏文件长度）；保留串行单连接及指数退避。4项下载测试通过，含持续低速终止/保留断点/恢复。旧准备器受控停止，归档状态before_lowspeed_*，新runner PID 3625244。实时速度见逐包日志，尚未宣称全部下载完成。证据results/summaries/stage2_lowspeed_recovery.json。

2026-10-20 持久恢复：用户明确要求Luna xhigh子agent实时监控，不使用schedule。子agent /root/nano_live_monitor 已启动。新guardian PID 3665265 等现有runner 3661216 自然退出后接管，保留当前下载连接。网络失败使用exit75并无限退避重试（60至300秒），校验/QC/配置错误仍明确failed并由监控报告。6项专项测试通过。guardian状态/进程见preparation/guardian_status.json及guardian_process.json；不自动启动GPU。

2026-10-20 23:12替换下载机制：旧guardian3665265及worker3736124已受控停止，仅操作本实验进程。旧aria2 partial（1836609643字节）保留且不假定为连续已验证前缀。新runner3781857/worker3781869调用stage2_download.py：4路4MiB Range，校验Content-Range再接受数据；40秒请求超时仍保留已验证响应前缀，逐块持久化，直接复用短期CDN签名链接（不写日志），不再循环不可达HF端点。最终整包SHA仍必须通过。实测23:13:21已确认121634816新字节且无chunk错误。并行提取器3781858提前处理16个校验包，已写出1000个src_mesh；未完成首包收据，不能声称完成质检。证据results/summaries/stage2_chunk_recovery.json，实时progress位于downloads/part_099.tar.gz.progress.json；只读监控agent已通知。没有使用GPU或schedule。

实时状态为results/stage2_nano10k/preparation/dataset_status.json，进程记录dataset_process.json，逐包日志和心跳位于assets/manifests/stage2_nano10k/downloads/。它仅获取固定17分片、做源mesh检查去重并划分10000/500/500；不自动启动GPU阶段。不要重复启动准备器；锁会拒绝并行。最终ready.json尚未生成，不能声称数据准备已完成。

无可独占GPU，未做GPU缓存/VAE gate/probe/训练。正式训练须当前数据缓存和VAE gate通过、独占GPU、资源probe通过，并由用户确认新预算。预算模板默认未批准。旧实验预算和下方历史记录保持不变。生成阶段使用decoder-only VAE，不构造encoder。

checkpoint每500更新保存且全部保留，最终1565额外保存；loss每100更新记录。所有新增产物/环境缓存/临时目录在远端userdata内。

---

# 实验交接

更新时间：2026-10-14T11:09:55.057885+08:00（Asia/Shanghai）。

## 状态
- 实验收尾已完成：训练、全部9组benchmark、两个checkpoint各8对训练几何、缺图恢复、报告和文档均已结束。
- CPU指令敏感性、全部审计、报告生成和最终复核已完成且退出码为0。无本实验运行任务。
- 完成执行不代表编辑复现成功：当前开源替代方法未表现出可靠的指令编辑能力。
- 累计GPU分配37363.49秒（10.3787/24小时），全部失败与恢复时间已计入。后续不得清零预算。

## 固定环境与数据
- 唯一工作区：/root/userdata/zc/hunyuan3d-buffalo-1.0-editing；ssh -o BatchMode=yes h100。本地仅AI/SSH操作，不存放实验产物。
- 官方Hunyuan3D2.1源码固定82920d643c0dc2f7bfd7255f45f62d386edfe60c，src/Hunyuan3D-2.1干净无补丁。
- environment/venvs/shape与vlm为专用Conda prefix；torch2.5.1+cu124，Transformers4.46.0/4.51.3，pip检查和依赖锁已保存。
- 模型/数据权重验证、固定revision与112收据在logs/downloads/；HF直连不可用时用了固定revision镜像和官方备用来源，网络中继已关闭。
- Nano3D3000候选，1798独立add配对通过自动QC。train1024条=512配对正反向，val128条=64独立配对正反向；test为Edit3D-Bench40另物体(add20/remove20)。无跨split几何哈希交集。
- 源/目标共用源导出坐标归一化，无独立归一化/ICP。AI检查46候选、排除16，最终30对有视觉观察，其余482训练对仅自动QC。

## 方法与checkpoint
- Qwen2.5-VL-7B：源四视角+指令，最后层全部有效输入token hidden，不生成文本回答。
- MLP3584→1024→1024/GELU，pool1370。源Shape-VAE4096×64 token与噪声目标拼接，只预测目标速度。
- 冻结VLM/VAE/DiT原权重；LoRA attention rank8/alpha16，与连接器联合训练，10%noop任务。
- VAE FP32权重与输入+FP16 autocast；DiT BF16/adapters FP32。
- overfit_best step200；connector_best step500；LoRA训练2000步但评估lora_best step1920。校验见checkpoint_integrity.json。
- 3seeds、50Euler步、编辑guidance1；native源单视角官方DINO条件/CFG5/无指令，并非同等输入的编辑模型。
- CD/F-score基于16384表面点，source-derived共同坐标；变化区域指标为ground-truth辅助距离proxy，不是部件mask。
- source/vae/native/random/connector/lora/no_source/noop/shuffled详细含义见results/summaries/evaluation_modes.md。

## 实测结论
- velocity loss：connector full128val0.923168→0.615929；LoRA0.615929→0.427614，表示速度场训练有所拟合。
- 先前8样本gate仅要求velocity loss下降10%，未在扩大训练前验证几何记忆，是原执行验收限制；本轮补做几何检查，不改写原历史。
- 8对训练几何(单seed)：源baseline F-score0.964831；overfit0.959653；LoRA0.918637。两checkpoint均0/8对F-score优于源。不能认定几何过拟合成功。
- 未见benchmark：源F-score0.718601，random0.704222，connector0.656865，LoRA0.663540，shuffled0.665361。
- LoRA vs源F-score差-0.055060，物体bootstrap95%CI[-0.080108,-0.032240]；LoRA vs shuffled差-0.001821，CI[-0.004245,+0.000254]。正确指令无可辨认优势。
- 预选6benchmark及全部8训练例子已由AI看四视角，观察见benchmark_visual_observations.json和training_geometry_visual_observations.json；不是用户本人审核，不构成全体语义准确率。
- 网格有效率只保证加载/坐标/索引有效，仍可能孔洞、碎片或部件未编辑。本实验不能宣称官方Buffalo编辑复现成功。

## 恢复记录
- 原shuffled39/120处退出-6(SIGABRT)，此前EGL全黑帧失败；没有足够日志锁定原生崩溃根因。
- scripts/recover_experiment.py用同一pilot锁与24h预算，保留39条metrics，仅生成缺失条目。
- --skip-render将生成和EGL分离；scripts/repair_result_renders.py每网格独立子进程，共检查336组、补齐26组，剩余缺图0。
- 原报告旧overfit缺final_validation_entries字段导致失败；修复为按历史实际16条处理，原始训练记录保持不变。
- 原状态与失败收尾完整快照environment/recovery_before.json；原日志追加保留。恢复完成后正确/打乱输出相似度F-score0.989973(120样本)。

## 查看与证据
- results/summaries/final_report.md：完整报告；evaluation_modes.md：模式说明。
- results/visual_comparisons/：40张benchmark图，行source/target/vae/native/random/connector/lora/no_source/noop/shuffled。
- results/training_geometry_comparisons/：8张source/target/overfit/LoRA图。
- results/evaluation/<mode>/：GLB、首seed四视角、metrics.jsonl、summary.json、protocol.json。
- results/training_geometry/<checkpoint>/：训练样本GLB、四视角与metrics。
- results/summaries/evaluation_integrity.json：1080评估全部120/120、有效且无重复；checkpoint_integrity.json：adapter哈希/冻结权重检查。
- controlled_comparisons.json：同样本/物体级bootstrap；instruction_sensitivity.json：正确/错误/保持不变输出相似性。
- training_geometry_audit.json：8对逐样本几何检查、GLB哈希与变化proxy。
- environment/run_state.json：命令、开始/结束、GPU、退出码、累计预算；environment/recovery_status.json和finalization_status.json：CPU收尾退出码。
- logs/stage_*.log、train_*.jsonl、recovery_*.log保存所有运行记录。

## 再运行与后续
- 先读取agent.md和本交接，git status、检查本实验PID，不能与现有runner并行。
- 固定样本/权重/随机种子不要重划分。无Git remote；只在远端本地Git维护。
- 收尾恢复：environment/venvs/shape/bin/python scripts/recover_experiment.py，成功GPU阶段跳过；CPU脚本可独立重复。
- 不追加训练或扩大数据。本轮只完成既定收尾；如继续研究，应先诊断源token复制偏置、指令连接器利用和训练几何失败，再决定方法或数据变化。
