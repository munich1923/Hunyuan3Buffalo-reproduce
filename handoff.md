# 正式训练已获授权，无GPU时间预算上限（2026-10-21T17:02:49.264690+08:00）

用户明确指示“不要设置预算，直接开始启动训练，没有任何预算限制”，覆盖此前16 GPU小时上限及单独启动确认。formal_start_authorized=true，gpu_budget_seconds=null。实际GPU用量继续记录在原账本，历史验证费用不清零；不再按时长停止。仍执行既定5epoch/1565更新及过程监控，EMA/CFG关闭；最终候选/测试评估未自动扩展。

正式run为results/stage2_nano10k/stage2_nano10k_fsdp_r1/，父进程PID405547，启动时使用GPU2＋5。当前阶段sample_initial：先执行初始化40例监控，然后自动进入train_to_500；不加载probe/verify权重。父调度独立后台运行。账本limit_gpu_seconds已迁移为null，保留历史费用和限额变更记录。

无限预算改动仅Ledger记账/父调度deadline与授权门控/CPU测试；12项CPU回归通过，AST与SHA审计证明sampler、随机流、模型、优化器、worker、FSDP、DCP及配置/数据/缓存未变，复用已完成GPU验证。组合报告results/summaries/stage2_fsdp_unlimited_readiness.json。不得在运行期间修改*stage2*.py。OOM/非有限值等失败处理、物理GPU映射、共享显存与2GiB余量检查继续执行，不干扰其他实验。

以下为历史状态；预算及正式启动权限以本节和最新用户指示为准。

# 当前交接：Stage 2 FSDP实现与独立验证完成（2026-10-21T16:05:26.037893+08:00）

- 正式训练0更新；formal_start_authorized=false，最终候选/测试评估未授权；EMA/CFG关闭。用户计划第6.4节要求资源结果后单独正式启动确认。
- v4已成功退出：results/stage2_nano10k/stage2_nano10k_fsdp_verify_v4/status.json，状态verification_complete_waiting_formal_start。原始verification_report.json保留当时合同。
- 20次真实不同UID更新共640唯一物体，最长条件uid_91425覆盖；均值16.2694秒/更新（第2–20次，含I/O、确定性计算）。
- tiny实际MoE/activation checkpoint参考通过；tiny及完整3B新进程下一更新全部本地参数和Adam逐元素比较最大误差0，loss/UID/position/RNG一致。3B两rank检查4583219602与4583219256元素，容差保持atol1e-6/rtol1e-5。
- 保存、退出、独立两卡采样、GPU2 decoder-only两例有效mesh、无GPU CPU指标及8张四视图、完整恢复闭环通过。两例仅验证实现，不证明语义能力；审核表仍pending。
- 所有GPU阶段余量≥2GiB，最小GPU5训练2.4274GiB。真实CUDA总量换算cap，未调高共享90%限额；保存/恢复CPU分片峰值通过。
- v4后父调度边界修复经11项CPU回归通过。只改parent/test；当前数值核心/配置/数据/缓存与v4完全相同。组合readiness绑定当前合同，详细差异及来源：results/summaries/stage2_fsdp_readiness.json、stage2_fsdp_contract_audit.json。不得将原始GPU报告改写成新调度代码全量GPU重验。
- 预算账本results/stage2_nano10k/stage2_fsdp_budget_20261021.json无running条目；已用2.1811，余13.8189GPU小时。完整正式流程估约18.075GPU小时；仅更新约14.145GPU小时，剩余额度不足按当前实测完成5epoch。详情stage2_fsdp_training_resources.md/json。不延长、不降低验收、不增临时保存点。
- v1/v2接口失败、v3默认BF16 SDPA反向不确定导致Adam比较失败均保留。有限诊断后固定deterministic_algorithms=True、CUBLAS_WORKSPACE_CONFIG=:4096:8及TF32关闭，v4通过；不放宽容差。
- 正式入口scripts/run_stage2_fsdp_training.py；配置configs/stage2_nano10k_fsdp.yaml和stage2_nano10k_fsdp_budget.json。文档docs/stage2_nano10k_fsdp.md完整描述执行与停止规则。正式启动须用户确认后只改授权标志；正式模型从原始预训练DiT＋新Connector开始，绝不加载verify/probe checkpoint。
- 固定上游82920d643c0dc2f7bfd7255f45f62d386edfe60c保持干净，无.codegraph/，不建立索引。未重复下载、缓存、VAE gate。userdata个人配额仍未知，不能用共享df总空闲替代。

以下为历史阶段记录，以本节为当前状态。

2026-10-21T15:05:57.041223+08:00 独立verify_v3未通过3B全量下一更新比较：恢复后flow/aux/total loss与参考一致，但Adam exp_avg某tensor超过atol1e-6/rtol1e-5，gradient norm 0.1123358384 vs 0.1123401150。两个rank已退出；正式训练0更新。不放宽容差，正在有限BF16 SDPA反向重复性诊断。已通过：20次真实不同UID更新、tiny全参数及Adam新进程完全一致、DCP完整保存、model-only采样、GPU2 decoder-only两例有效mesh、CPU指标及4视角、所有已测阶段至少2GiB余量（最小GPU5训练2.740GiB）。3B精确续跑尚未通过，不可启动正式训练。所有失败和诊断计入共享16GPUh账本。

2026-10-21T14:31:28.789134+08:00 正式FSDP入口已初步实现，独立verify_v3执行中；正式训练0更新，EMA/CFG关闭。18旧CPU回归、5新CPU回归及比较文件分帧回归通过。v1/v2失败及费用保留在共享16GPUh账本；未把失败写成通过。详见results/summaries/stage2_fsdp_training_implementation.md。不要并发启动另一个GPU任务。

补充显存口径：GPU5全流程reserved42.494GiB，按CUDA总量换算实际allocator上限约42.527GiB，仅余33.2MiB；名义预算43805MiB不能当成实际字节上限。训练本身reserved36.75GiB，最大allocated全流程32.37GiB。正式FSDP训练前须优化并验证保存/恢复/采样阶段缓存及临时张量峰值；本轮仅记录限制，未宣称长期显存余量充分。

2026-10-21 11:50:35 +08:00 两卡FSDP probe完整成功退出0，GPU2＋GPU5已释放。5更新13.9201/13.1108/13.3773/13.9416/13.3264秒，后4均值13.4390秒，全局batch32。训练reserved36.82/36.75GiB，全流程45.43/42.49GiB。完整分片checkpoint36.674GB保存60.04秒、恢复73.60秒；8验证1.87秒；50步采样8.53秒；解码7.05秒有效mesh。小MoE恢复后下一更新完全一致；实际3B做了文件SHA、严格DCP加载和各tensor前4值比较，未做新进程下一步全参数比较。证据results/summaries/stage2_resource_comparison.md与stage2_fsdp_probe.json。正式训练0更新；FSDP目前仅probe入口，后续正式分布式入口仍需补齐日志、epoch调度、每500保存及评估，预算确认后才能启动。

2026-10-21 11:32:50 +08:00 CPU卸载probe完整成功退出0：5次更新63.1769/48.7252/51.5700/55.0728/58.7723秒，后4次平均53.5351秒；peak allocated32144878080B/reserved34477178880B；CPU peak RSS51340376KiB。完整checkpoint36666710331B，写入与校验63.5738秒；8验证2.7109秒；50步生成6.6968秒；VAE解码7.4901秒且有效mesh。证据probe_gpu2_cpuoffload_20261021/probe_report.json、summaries/stage2_cpuoffload_probe.json。仅资源可行，未评价语义成功；正式训练0更新。FSDP supervisor88432正在CPU缓存复核，GPU部分尚未启动，持续监控。

2026-10-21 用户已授权CPU卸载probe结束后进行GPU2＋GPU5两卡FULL_SHARD探测并要求持续监控。新增stage2_fsdp.py / probe_stage2_fsdp.py / run_stage2_fsdp_probe.py；先小尺寸实际上游MoE的梯度更新及分片恢复一致性，再完整3B五更新、真实checkpoint恢复、验证8例、50步采样和解码。代码尚待实际GPU验证，不将语法通过当资源成功。CPU卸载probe已完成首更新63.1769秒，peak allocated29.94GiB/reserved32.11GiB。

2026-10-21 11:09:17 +08:00 CPU优化器卸载探测已启动，supervisor70224/worker70232；独立配置configs/stage2_nano10k_cpuoffload.yaml仅optimizer改为cpu_adamw，其余参数保持一致；从原始初始化重新做5更新，不续接旧失败的未保存更新。状态和日志results/stage2_nano10k/gpu2_cpuoffload_20261021/，run结果probe_gpu2_cpuoffload_20261021/。18CPU回归再次通过。共享GPU2 cap继续保留；正式训练仍未启动。

2026-10-21 11:05状态核查：text和posterior各11000条完整完成；VAE gate于05:50通过，mean有效110/110且Fscore005中位0.9757215，posterior有效110/110且中位1.0。GPU AdamW probe于06:18:56第二更新forward触发进程显存cap OOM（首更新32micro完成28.301秒，peak allocated49.53GiB/reserved50.77GiB；当时进程约52.28GiB，卡仍余5.34GiB），没有完成5更新，也没有正式训练。旧失败全部保留。按已批准备用方案，准备独立cpu_adamw config/run做GPU2 CPU状态卸载探测；正式训练仍未获预算授权。probe最长条件改为复用CPU preflight已验证长度，避免分配GPU后再无谓重复读取全部posterior。

2026-10-21 00:23:30 +08:00 GPU2已实际完成首批posterior编码42条，抽查mean/logvar FP32各[4096,64]、有限值及指纹通过，编码阶段整卡占用25054MiB（含其他进程），本阶段尚未等于训练显存探测。runner继续全量posterior→VAE gate→5更新probe，正式训练0更新。posterior抽查证据results/summaries/stage2_gpu2_posterior_spotcheck.json。

2026-10-21 00:19:54 +08:00 text缓存完整完成11000条（train10000/val500/test500），所选token长度44至186，无全序列池化。00:19:55已切换posterior缓存PID3822637，尚在加载/权重校验时不能称posterior完成。4共享测试和18CPU回归最终均通过，原始回归日志gpu2_readiness_20261021/cpu_regression.log。

2026-10-21编码复现性补充：发现固定上游encoder内部仍调用randperm及random-start FPS；在尚无posterior缓存时加入逐UID派生encoder seed与fork_rng，同时升级shape contract指纹，不修改上游或已生成文本缓存。

# 共享GPU2准备与资源探测（2026-10-21）

用户已明确取消独占要求并授权GPU2执行资源探测。已加入显式STAGE2_ALLOW_SHARED_GPU=2准入；仅约束本实验PyTorch allocator为每阶段启动空闲显存的90%，不限制或操作其他进程，不能保证其他任务未来不会增长。4项CPU共享准入测试及18项完整CPU回归通过；CPU测试锁与真实任务锁隔离。runner PID3816750，首阶段text缓存PID3816761，状态results/stage2_nano10k/gpu2_readiness_20261021/status.json。串行text→posterior→110例VAE gate→5更新资源probe；失败即停止，不自动启动正式训练，预算仍未批准。原始数据ready在2026-10-20 23:49已完成。下方较早的未ready/独占描述是历史状态。

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

23:20并发资源观测：确认4路持续新增约0.8至1.3MiB/s、无429后增加为8路Range，新runner3787799，extractor3781858不变。原始partial经91块独立对照及全前缀gzip结构检查后复用309块/1296039936字节，仍要求最终SHA。详见stage2_old_prefix_validation.json；不能把复用跳跃计作网络吞吐。

23:35:43 +08:00 最终part099整包SHA256 3142975cc46c2faf0a366175d117bd5e90b2de1e7acc3c8df74c3253d317a3c0通过，5327786711字节，17分片已全部校验。下载故障已解决，非仅进程存活。前16分片16000源mesh已提取；099提取及全量QC/split仍运行，不等同训练ready。

23:39 CPU QC加速：17包源mesh全部17000提取完成，QC单任务改为4个spawn进程，pool.map维持原UID顺序/原始去重代表和split种子，已完成检查记录复用。新runner3801512；4例串行/并行输出完全一致专项检查通过。下载相关6项检查重跑通过。最新QC 1360 条，ready尚待。

2026-10-20 23:49:02 +08:00 CPU数据准备完成：17包下载及固定SHA通过、17000源mesh提取、17000候选基础QC通过（未发现哈希重复，不等同语义质量人工审核）。ready.json生成，10000/500/500固定划分完成；逐split文件SHA、UID和geometry_hash跨集合无交集、源mesh存在及userdata路径已复核。证据results/summaries/stage2_dataset_integrity.json。prepare runner3801512正常exit0，后续不应再重启下载器。GPU缓存/VAE gate/probe/训练未执行，仍须明确GPU和预算授权。
