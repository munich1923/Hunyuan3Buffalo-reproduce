# Nano3D 10,000物体 Stage 2近似实验：GPU2＋GPU5 FSDP执行合同

唯一工作区：/root/userdata/zc/hunyuan3d-buffalo-1.0-editing。所有模型、数据、环境、缓存、TMP、代码和产物留在此userdata目录。固定上游不修改。执行成功与否以独立验证报告和原始日志为准。

## 目标、依据与边界

纯文本原prompt → 冻结Qwen2.5-VL-7B-Instruct → 新Connector → 从预训练Hunyuan3D-2.1初始化且完整更新的DiT → 纯高斯噪声生成几何。当前缺少Buffalo Stage 1完成后的VLM，因此称Stage 2近似实验，不宣称官方训练源码或完整复现。

依据：https://arxiv.org/html/2608.02711v1 、https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0 。上游固定82920d643c0dc2f7bfd7255f45f62d386edfe60c，项目适配不得修改上游。

不使用DINO、LoRA、旧编辑Connector/权重、EMA、CFG或CPU优化器卸载。mesh、点云、图片、source latent及target信息不进入条件分支。src_mesh仅离线制备posterior；decoder不进入训练主损失。验证更新的权重不接入正式训练。

## 固定输入及缓存

Nano3D revision d6686ff82ea14c2dfa4fdd55db332309a0241495；17分片000,007,014,021,028,033,035,042,049,056,063,066,070,077,084,091,099。17,000候选池内分层抽样，沿用固定train10000/val500/test500，不重新划分。UID和几何哈希无跨集交集。条件为原prompt，监督为同UID src_mesh.glb，不用edit_instruction、tar_mesh或图片。保留材质/颜色文字，只评价几何可表达内容，不调用外部caption服务。

路径相对根目录：assets/manifests/stage2_nano10k/，assets/cache/stage2_nano10k/text/及shape/。主seed20261019。Qwen revision cc594898137f460bfe9f0759e9844b3ce807cfb5；Shape revision 0b94677654c57bb9a6b6845cd7b704ccf551d327。

文本经chat template、不生成回答；取最后一层prompt正文token及assistant prefix最后token，排除system正文/padding。BF16 hidden [L,3584]，当前L44–186；MLP Linear3584→1024/GELU/Linear1024→1024。每卡microbatch1，保留实际长度，无pooling/固定1370/padding。

监督：应用场景变换并合并，包围盒中心居中、最长边1.5、保持方向；逐UID确定性采样81920点，XYZ+单位法线+零sharp；冻结encoder/pre_kl，FP32 mean/logvar各[4096,64]，不乘scale，logvar官方[-30,20]。encoder内部randperm/random-start FPS绑定逐UID种子/fork_rng。

11k文本、11kposterior和110物体mean+posterior VAE gate已完成。父进程复用已有全11k成功预检并检查22k文件stat指纹；只重检变化内容，不重新编码/gate，rank不重复扫描全缓存。CPU线程最多预取一个更新批，不消耗训练RNG。

## 训练定义及数值合同

FP32逐呈现重新采样：eta~N(0,I)，z1=1.0039506158752403*(mean+exp(0.5*logvar)*eta)，z0~N(0,I)，t~Uniform(0,1)，zt=(1-t)*z0+t*z1，target=z1-z0。L_flow=mean((velocity-target)^2)，L_total=L_flow+L_moe。t0噪声，t1数据。

DiT几何输入/输出[1,4096,64]，不拼source latent，排除内部time token。MoE辅助loss显式返回，上游0.01已包含，不重复乘系数/隐式梯度；与主loss按实际样本数相同归一化，checkpoint重算不重复累计。路由统计为microbatch范围。

物理GPU2+GPU5，记录UUID；block FULL_SHARD外加root，use_orig_params=True，非重入activation checkpoint在block内部；每micro同步reduce-scatter，不用no_sync。FP32参数/梯度/Adam、BF16 autocast、TF32关闭。

固定torch.use_deterministic_algorithms(True)及CUBLAS_WORKSPACE_CONFIG=:4096:8，以满足新进程下一更新全参数/Adam比较。默认BF16 SDPA反向重复性诊断出现梯度差异，确定性模式诊断完全一致；确定性配置必须实际验证资源和全量续跑，不能只检查开关。

5 epoch、50000次呈现；global32，每rank micro1累积16；epoch尾批global16、每rank8；每epoch313、总1565更新。全局唯一shuffle后32分组，rank0偶数、rank1奇数，不drop/补重复。每rank loss除本次本卡实际样本数，再FSDP平均得到全局均值。

AdamW：DiT LR1e-5、Connector1e-4，betas(.9,.99)、eps1e-6、weight_decay.01、foreach=False；FSDP全局clip1.0。80更新warmup，之后cosine到峰值10%，按优化器更新推进。初始化、shuffle、各rank posterior/noise/time、验证、采样各独立随机流。

## checkpoint、评估及恢复

初始化40监控 → 双卡训练到500/1000/1500/1565 → 完整DCP保存并退出 → 双卡model-only采样 → 退出 → GPU2 decoder-only → 退出 → CPU指标/渲染 → 新训练worker完整恢复。

监控固定train16+val16+wrong8，每节点40，共200。换条件保持UID派生初始噪声；生成rank处理不同任务，保持相同collective次数；必须正常model forward，不能.velocity绕过root hooks。

scheduler固定FlowMatchEulerDiscreteScheduler与np.linspace(0,1,50)，保留端点/末尾零步长。模型时间timestep/num_train_timesteps；BF16 DiT，scheduler上游FP32/dtype规则。guidance_scale1。最终latent除scale后decoder：bounds1.01，mc_level0，octree256，chunks10000，mc_algo=mc。空条件仅分布外诊断。

仅保存checkpoints/step_000500/、step_001000/、step_001500/、step_001565_final/，不覆盖/轮换/删除/额外epoch保存。约36.7GB/份、四份146.8GB，另留临时提交和验证比较空间。共享FS总空闲不是用户配额证明；无quota工具时如实记录未知。

DCP CPU分片state包含完整模型/Adam；rank文件包含scheduler、epoch/offset/update、CPU/current CUDA RNG、训练generator、未输出区间。合同绑定config/数据/模型/缓存/代码/包装/world2。保存前清梯度/graph/空闲缓存，不构造GPU完整state；结束释放临时state。payload SHA及complete标记原子提交，只从完整边界续跑。

固定两rank/相同包装，不承诺跨卡数精确恢复。异常从最近完整checkpoint恢复，可能重算不足500更新，不宣称中断点精确续跑；阶段恢复先补未完成评估，再前进训练。合同变化新run。最优索引不复制大权重。

每epoch全500验证，每rank250唯一物体，sum/count allreduce，逐UID固定随机量，MoE aux eval停用。保存节点补同一验证，重合复用。每100更新及1565末65次记录样本加权flow/aux/total、LR、global norm、吞吐/耗时、各rank GPU/CPU峰值及预算，rank0 JSONL/TensorBoard；每micro/更新检查非有限loss/梯度。区间状态随checkpoint保存，续跑保留run/步数。

## 共享准入、预算与停止

每阶段重新读GPU2/5空闲，以90%空闲字节准入，除真实CUDA total bytes换算allocator fraction；记录cap。所有GPU阶段peak reserved距cap至少2GiB，缺余量停止，不提高cap、不操作他人进程。allocator max_split_size_mb256，不依赖平台不支持的expandable_segments。

新增GPU验证+正式训练+过程监控共享16GPU小时硬上限57600 GPU秒，账本results/stage2_nano10k/stage2_fsdp_budget_20261021.json。双卡加载/等待/训练/保存/验证/采样×2，decoder×1，释放GPU后的CPU×0；失败/重启不重置，历史probe独立保留。最终候选/测试预算另议。上限不承诺完成5epoch。

父进程持实验/设备/预算锁，子进程为本实验独立进程组；OOM、非有限值、合同变化、分布式错误或预算耗尽停止本实验rank，保留日志，不影响他人。不加临时checkpoint、不延长预算、不降低验收。

## 独立验证、入口与命令

verify-only权重不接正式：
1. CPU全epoch覆盖、尾批/恢复、flow/scale/token与预算。
2. 实际tiny上游MoE+activation checkpoint梯度参考。
3. 20次真实不同UID更新，覆盖最长条件和I/O/训练峰值。
4. 第20次完整保存，连续第21参考全部本地参数及全部Adam逐元素快照；新进程恢复21，atol1e-6/rtol1e-5，核对loss/UID/位置/RNG。
5. 保存→退出→采样→解码→CPU→新worker恢复闭环，所有GPU阶段2GiB余量。

父入口scripts/run_stage2_fsdp_training.py；worker scripts/stage2_fsdp_worker.py；DCP scripts/stage2_fsdp_io.py；采样器/预取/账本scripts/stage2_schedule.py；decoder/CPU scripts/stage2_phases.py。

已完成v4独立验证的历史命令（下述当前父调度细化后不重跑同一run；数值核心变化才新建验证run）：
    cd /root/userdata/zc/hunyuan3d-buffalo-1.0-editing
    bash scripts/stage2_launch.sh run_stage2_fsdp_training.py --config configs/stage2_nano10k_fsdp.yaml --gpus 2,5 --run-id stage2_nano10k_fsdp_verify_v4 --budget configs/stage2_nano10k_fsdp_budget.json --verify-only

--preflight-only仅CPU预检；同合同已完成阶段复用，不可在活父进程外并发启动。当前验证通过，预算配置verification_report已指向results/summaries/stage2_fsdp_readiness.json；仅用户明确正式启动授权后formal_start_authorized置true。正式从原始预训练DiT+新Connector开始：
    bash scripts/stage2_launch.sh run_stage2_fsdp_training.py --config configs/stage2_nano10k_fsdp.yaml --gpus 2,5 --run-id stage2_nano10k_fsdp_r1 --budget configs/stage2_nano10k_fsdp_budget.json

同run恢复加--resume auto或最新完整checkpoint绝对路径；不接受较旧保存点。首次保存前中断可用--resume auto重算；已完成初始化监控复用。未完成的.pending/非提交目录原子移动到incomplete/保留后再重试，不删除或覆盖。未提交更新日志先保留到interrupted_logs/再回退到最近完整更新，避免重算步重复；原始失败证据保留。入口不自动最终评估。TensorBoard继续127.0.0.1:6006；缺失才启动，不终止他人服务。本地转发：
    ssh -o BatchMode=yes -N -L 6006:127.0.0.1:6006 h100

## 质量证据及编辑门槛

loss、几何、条件使用分别报告；固定16384点，不独立重缩放预测mesh。Chamfer/Fscore仅参考；文本有多解，语义不由距离推断。审核记录审核者、主体/关键几何/结构、固定噪声换条件、理由和失败案例。

最多选2个checkpoint：val监控关键几何+结构优先、主体次之、val flow loss并列；train不选checkpoint。每候选val100正确+20wrong+8empty；锁定后test100正确+20wrong，val/test对照各10跨类+10同类，必须几何可区分。测试不调参/选checkpoint，预算另批。

进入编辑训练要求留出有效mesh≥95%，test主体≥80%，关键几何+结构同时≥70%，val/test对照各≥15/20，且比初始化train与val生成及条件符合度均改善。loss下降、输出变化或训练拟合不能单独证明成功；不默认归因数据规模。


## 当前执行结果与预算决策（2026-10-21T16:05:26.037893+08:00）

完整执行报告：results/summaries/stage2_fsdp_training_implementation.md。组合readiness通过：v4数值核心GPU证据＋11项CPU父调度边界回归，差异逐文件与AST审计。原18项CPU回归和tiny真实MoE/检查点参考通过；3B新进程下一更新全部本地参数和Adam、loss/UID/position/RNG一致，两rank最大误差0。20次真实更新640唯一UID，覆盖最长条件。闭环两例mesh及四视图齐全；不据此宣称生成语义成功。全部阶段最低余量2.4274GiB。

默认BF16 SDPA反向不确定导致v3严格比较失败，未放宽容差；固定确定性设置后v4通过。v1/v2接口失败及v3数值失败完整保留、计费，不抹去失败。v4之后只完善父调度预算/中断恢复/状态，未改数值worker/FSDP/DCP/config/数据/缓存，复用已完成的数值验证，不无谓重新运行GPU任务。

新预算已耗2.1811，剩13.8189GPU小时；真实I/O/确定性更新16.2694秒，纯更新约14.145GPU小时，完整正式流程估约18.075GPU小时。详见results/summaries/stage2_fsdp_training_resources.md/json。当前剩余额度无法按实测估算覆盖5epoch；预算硬上限仍16，正式0更新，不自行增预算/降验收。

父调度预留60 GPU秒关闭余量，worker截止时间同步提前；SIGTERM沿finally清理本实验进程组。formal_started.json持久记录正式启动，阶段切换不会误报未开始。合同变化拒绝旧run；预算失败/中断计费不重置。

正式训练仅在用户按计划第6.4节明确确认后，将预算文件formal_start_authorized置true并执行上文stage2_nano10k_fsdp_r1命令。确认若仍坚持现有剩余预算，按上限停止、保留最近完整保存点，预计不能完成全5epoch；不得追加临时checkpoint。最终候选/测试不自动执行。
