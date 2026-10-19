# Stage 2 Nano3D 条件生成：实施与运行说明

本轮使用通用 Qwen2.5-VL，而非完成论文 Stage 1 的 Hunyuan3D-VLM，属于 Stage 2 近似实验。完整预训练 Hunyuan3D-2.1 DiT 与新 Connector 联合训练；Qwen/VAE 冻结。旧编辑实验没有证明编辑能力，其权重、数据划分、预算及产物均保留。

## 唯一路径与固定版本

所有代码、数据、依赖、临时文件、下载/编译缓存、日志和产物均位于：
/root/userdata/zc/hunyuan3d-buffalo-1.0-editing

项目起点 db710fc8741d71fa77a99de5e712c25e585efed6；
固定上游 82920d643c0dc2f7bfd7255f45f62d386edfe60c；
Nano3D d6686ff82ea14c2dfa4fdd55db332309a0241495。
上游不打补丁，不建立 CodeGraph 索引。本地只使用 SSH/AI，不保存实验产物。

所有入口优先经 scripts/stage2_launch.sh 启动：它在导入模型库前设置 userdata 下的缓存、临时目录、关闭字节码并限制 CPU 线程。Python 内部也独立检查实际解析路径，拒绝工作区外路径。模型文件进行内容校验并绑定缓存指纹。

## 数据和输入合同

候选17个分片：000,007,014,021,028,033,035,042,049,056,063,066,070,077,084,091,099。
压缩包总量约104.53GB，其中原有约19.32GB。按全库类别比例，在17,000候选池内分层抽样，不是全库随机抽样。

只检查 source mesh 的完整性、可解析性、场景变换、非有限数据、有效面片、尺度退化及保守几何重复。几何重复组只取一名代表。按最大余数法分配10000训练、500验证、500测试，类别不足时失败并报告，不静默补数或改比例。旧编辑指令/target配对QC不适用。不使用ABO、HSSD、外部caption服务。

条件只来自原 prompt；目标为同UID src_mesh.glb。tar_mesh、编辑指令、图片均不进入训练。颜色/材质文本保留，但只评估几何可表达的属性。

条件：
prompt → frozen Qwen last hidden [L,3584] BF16 cache
→ Linear3584→1024 / GELU / Linear1024→1024
→ [1,L,1024] cross-attention。
选择用户正文和最后一个assistant prefix上下文token，不生成回答、不池化、不补足1370；microbatch1，无padding。

监督：
scene-transformed src_mesh → bounds-center / longest-side1.5
→ fixed 81920 points [XYZ,unit-normal,zero-sharp-flag]
→ frozen encoder+pre_kl
→ FP32 mean/logvar，各[4096,64]。
encoder只离线使用；stage2_vae.py分别构造encoder-only和decoder-only模块，生成阶段不实例化encoder/pre_kl，单体checkpoint使用mmap选取相应权重。logvar采用上游[-30,20]截断；缓存未乘scale。
训练重新采样 z1=1.0039506158752403*(mean+exp(.5*logvar)*eta)。

缓存分别按模型内容、版本、模板/token规则、prompt/mesh哈希、归一化、采样与种子指纹隔离。旧缓存不复用。数据ready.json绑定split和评估协议内容。数据准备先完成不可变清单，再进行GPU缓存。

主种子20261019；抽样/初始化/打乱/表面采样/训练/评估使用命名独立随机流。每次训练重新取posterior噪声、flow噪声、t；验证随机量固定。

## 模型、优化与采样

从预训练2.1初始化DiT，训练完整DiT和Connector，包括MoE。无source latent拼接、DINO、LoRA、EMA、CFG。无条件分支未训练，空条件只作分布外诊断。

zt=(1-t)*z0+t*z1，z0~N(0,I)，t~Uniform(0,1)；
velocity target=z1-z0；主损失是latent velocity MSE，不decode mesh反传。
保留官方MoE负载均衡目标alpha=.01。项目侧替换MoE forward，显式返回aux，取消隐式注入；主项/aux统一除实际累积样本数。block checkpoint返回aux而不使用全局累计列表，避免重算双计。

训练10000物体，5epoch，microbatch1，累积32，每epoch312个完整更新+16样本尾批，1565更新。尾批按16归一化。AdamW betas(.9,.99)、eps1e-6、wd.01、clip1。DiT/Connector峰值LR1e-5/1e-4，80更新线性预热，cosine降至10%。FP32参数/状态，BF16 autocast，block checkpoint。

默认GPU AdamW。如果GPU探测不满足余量，配置training.optimizer可改为cpu_adamw并建立新probe/run。此实现是同步CPU主参数+Adam状态卸载，已做CPU数值对照；不依赖DeepSpeed，GPU传输、性能和峰值仍待实测，不声称与DeepSpeed实现相同。

采样使用固定上游FlowMatchEulerDiscreteScheduler，显式np.linspace(0,1,50)，保留官方末尾零步长；timestep/1000作为模型时间，FP32更新后按官方转换dtype。从纯噪声开始，最后除scale，VAE解码提取bounds1.01、level0、resolution256、chunks10000、mc。guidance1。

## 日志、TensorBoard和checkpoint

每100次优化器更新按区间实际样本数加权记录flow_mse、moe_aux、total_loss、两组LR、梯度范数、耗时/吞吐、显存/CPU峰值；最终补记65更新。每microstep检查非有限值。JSONL与TensorBoard标量一致。

checkpoint仅在500、1000、1500、1565_final保存，全部保留，不轮换、不覆盖、不额外按epoch保存。完整权重、Adam、调度位置、数据位置、随机状态、训练generator、指纹及未完成日志区间一并保存，文件校验+原子提交。最优checkpoint只保存路径索引。

同run续跑必须指纹完全相同，只恢复更新边界；未保存更新可能需要重算。TensorBoard purge_step处理恢复后的重复显示。异常失败不写成训练成功。

TensorBoard2.18.0与兼容setuptools80.9.0位于userdata内shape环境；仅127.0.0.1监听。默认6006，冲突自动换可用端口，不杀其他服务。state记录真实端口和PID。没有训练时不创建假的loss曲线。

## 验证与人工审核接口

不做独立8物体训练。CPU合同通过后，完整数据缓存，再做VAE gate、资源probe，预算确认后才正式训练。

VAE gate：训练/验证内每类10个，共110个，分别decode mean和固定posterior样本。每组有效率>=95%、Fscore@.05中位数>=.80（最长边1.5），分类别报告。失败先诊断，不默认归因于规模。gate绑定shape cache、评估配置和协议。

每epoch对全部500验证物体做固定随机量latent loss。
初始化与四个保存节点生成16训练+16验证+8错条件，共40个输出/节点。
每个保存节点计算固定验证loss用于候选并列排序。

首次完整生成只做固定100验证+100测试，其余400+400保留。
验证/测试各20组条件对照（10跨类、10同类），val另8个空条件诊断。
数据准备生成condition_pair_review.json，需在完整生成前确认对照prompt具有几何可区分性。若不成立，不静默重选对照。
所有生成来自同UID固定高斯种子，正确/错误/空条件只换条件。

几何：16384表面点Chamfer/Fscore，无预测单独缩放或ICP；记录有效率、逐物体、分类别结果。显示统一CPU固定视角，无纹理，不向条件分支回流。CPU渲染最多显示50000面片，GLB和指标使用全几何；渲染限制须保留说明。

语义不会从loss/Chamfer自动推断。每个结果的reviews/*.json需具名填写：
subject_correct、key_geometry_correct、overall_structure_usable；
对照还需pair_geometrically_distinguishable、condition_change_correct。
未审核时语义状态明确pending。

候选选择只使用val监控，按结构+关键几何通过率、主体通过率、固定验证MSE排序，最多2个；在100val确认后锁定selected_checkpoint.json。test只允许锁定checkpoint。
acceptance还需improvement_review.json具名确认训练/验证生成及条件符合度较初始化改善。

进入编辑前：留出有效率>=95%，100test主体>=80%、关键几何+整体结构>=70%，val/test各20条件对照至少15成功，且训练/验证相较初始化均改善。所有阈值是本项目工程门槛，不是论文指标。只下降loss不能进入编辑。

## 执行命令

以下GPU命令必须有独占卡，--gpu为物理卡号；程序还会锁定GPU并检查显存/计算进程，不自动找卡。

CPU准备：
bash scripts/stage2_launch.sh prepare_stage2_runner.py
bash scripts/stage2_launch.sh test_stage2_cpu.py
bash scripts/stage2_launch.sh start_stage2_tensorboard.py

GPU缓存（另行执行，禁止并行占同卡）：
bash scripts/stage2_launch.sh cache_stage2_text.py --gpu N
bash scripts/stage2_launch.sh cache_stage2_shape.py --gpu N
bash scripts/stage2_launch.sh evaluate_stage2.py --mode vae --gpu N

GPU资源测量：
bash scripts/stage2_launch.sh train_stage2.py --run-id probe_20261019 --probe-only --gpu N

正式启动前用资源报告填写environment/stage2_budget_approval.json：
approved=true、approved_by、正数gpu_budget_seconds、当前config_fingerprint、probe_report、probe_sha256。只有用户确认新预算后才填写批准；不得自动批准。旧编辑37363.49秒单独保留。
bash scripts/stage2_launch.sh train_stage2.py --run-id RUN --gpu N --budget environment/stage2_budget_approval.json
续跑增加 --resume results/stage2_nano10k/RUN/checkpoints/step_XXXXXX。

评估：
先填写监控review；evaluate_stage2.py --mode candidates --run-id RUN
对候选运行 --mode val --checkpoint PATH --run-id RUN --gpu N
填写val review；--mode lock-selection --run-id RUN
--mode test --run-id RUN --gpu N
填写test/条件/改善review；--mode acceptance --run-id RUN

本地转发：
ssh -o BatchMode=yes -N -L 6006:127.0.0.1:6006 h100
浏览http://127.0.0.1:6006；真实端口以environment/stage2_tensorboard_process.json为准。

## 资源边界与未运行项

单卡H10080GB仅为候选。FP32参数/梯度/Adam静态约45.46GiB，不含激活临时区；probe要求至少10%余量。11k posterior纯FP32约21.48GiB；4完整checkpoint约136GiB量级，卸载布局与序列化实际需测。解压空间、配额、CPU可分配内存和I/O尚未知。

probe覆盖最长实际条件、32次累积、5完整优化器更新、首次状态分配、完整checkpoint I/O，以及8例验证前向和一例50步采样+解码。验证全量成本按8例实测外推并明确标注估计。报告的训练计算估计明确不含缓存、完整验证/生成/审核和其他保存成本；启动预算需一起估算，不据此直接自动启动。

本轮未获独占GPU，不运行VLM/VAE编码、VAE gate、资源probe、训练或生成评估。代码及CPU测试通过不代表GPU路径已验证，更不代表模型已经具有条件生成或编辑能力。
