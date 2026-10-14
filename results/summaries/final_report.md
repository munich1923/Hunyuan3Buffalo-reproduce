# Hunyuan3D Buffalo 编辑能力近似复现实验报告

生成 UTC：2026-10-14T03:12:54.280202+00:00
生成 Asia/Shanghai：2026-10-14T11:12:54.280232+08:00

## 结论与执行状态

执行状态：completed；9组benchmark是否全覆盖：True；累计单卡GPU分配时间：10.3787/24小时。

本实验使用开源Qwen2.5-VL替代闭源Buffalo VLM，不是官方编辑权重或官方指标复现。

**当前近似方法没有复现可靠的未见物体指令编辑：LoRA对目标的整体几何分数低于直接保留源物体。**

正确与打乱指令生成网格：共同有效样本120，物体40；输出间F-score@0.05=0.989973，CD平方和=0.000743。相似度高支持指令敏感性弱，但不是完整语义成功率。

接口通过、velocity loss下降、几何拟合、未见物体编辑有效性必须分别判断。网格有效率只代表可加载且坐标/索引合法，不保证闭合、连通或语义正确。
原计划要求8样本过拟合诊断先于扩大训练；先前实际gate仅要求固定噪声velocity loss下降10%，未在正式训练前验证几何记忆。这是验收口径的限制，本轮补做几何检查；不能把此前通过loss gate称为完整过拟合成功。

原eval_shuffled异常退出码-6(SIGABRT)，退出前记录EGL全黑帧错误；日志没有足够证据锁定原生崩溃根因。恢复采用生成与渲染分离、每网格独立EGL进程，保留原39条记录和失败时间。原报告因旧overfit报告缺final_validation_entries失败；现在按当时脚本实际16条验证处理，原始报告未改写。

## 数据、环境与训练

源码固定82920d643c0dc2f7bfd7255f45f62d386edfe60c；独立Shape/VLM环境，PyTorch2.5.1+cu124。权重revision、下载大小/SHA和依赖锁文件留存。Nano3D3000候选中1798独立add配对通过自动QC；train1024条是512独立配对正反向，val128条是64另配对。AI检查46候选、排除16，最终30配对视觉检查，其余482配对只有自动QC。

Qwen输入源四视角+指令，使用最后层所有有效输入token隐藏状态，不生成文本回答。MLP3584→1024→1024，pool1370；源Shape-VAE token与噪声目标拼接，只预测目标速度。VLM/VAE/DiT原权重冻结；attention LoRA rank8/alpha16，LoRA阶段同时训练连接器。10%保持不变训练。

正式首训练源VAE gate F-score@0.05=0.996386，只验证该例。编辑VAE使用FP32权重/输入+FP16 autocast；DiT BF16/adapters FP32。

|阶段|训练步数|初始train probe|最终train probe|初始val|最终val|最终验证条目|
|---|---:|---:|---:|---:|---:|---:|
|overfit|200|0.992637|0.629340|1.030458|0.662861|16|
|connector|500|0.965803|0.638640|0.923168|0.615929|128|
|lora|2000|0.638640|0.410058|0.615929|0.427614|128|

正式connector/LoRA最终val覆盖128条；best选择使用固定16条probe。benchmark和几何诊断使用best，LoRA best为step1920，训练末尾为step2000。overfit_best为step200。

## 8对训练样本的实际几何

同一8个独立原始add训练配对，每模型单seed20261013，50步采样；不追加训练。对比实际目标和不编辑源物体。

|checkpoint|完成/期望|有效|生成CD↓|源不编辑CD↓|生成F-score↑|源不编辑F-score↑|
|---|---:|---:|---:|---:|---:|---:|
|overfit|8/8|8|0.001361|0.001281|0.959653|0.964831|
|lora|8/8|8|0.002356|0.001281|0.918637|0.964831|

overfit：未优于不编辑源物体，不能认定学会几何记忆。


lora：未优于不编辑源物体，不能认定学会几何记忆。


AI检查全部8对四视角，观察如下（小细节无法据此确认）：

- uid_66677_add: Both outputs resemble source octopus; target protruding mast/back addition is absent. Verdict: not achieved.

- uid_859_add: Carriage is retained; target front mask/horns are not clearly reproduced across views. Verdict: not clearly achieved.

- uid_33733_add: Target legs/head appendages around base are absent; both outputs keep source-like round base. Verdict: not achieved.

- uid_66061_add: Target wide hat canopy is absent; source carriage ornament is retained. LoRA loses thin geometry. Verdict: not achieved.

- uid_33740_add: Overfit lacks target front spider legs. LoRA adds irregular front protrusions, but does not reproduce three target mechanical legs. Verdict: partial irregular change, not target fit.

- uid_33784_add: Target long blade extending from sides is absent; generated dragon sculpture remains source-like. Verdict: not achieved.

- uid_120_add: Target wrist/forearm thickness change is not clearly reproduced; LoRA introduces unrelated shoulder/head fragments. Bracelet runes are below visual resolution, so semantic detail cannot be judged. Verdict: not clearly achieved.

- uid_66911_add: Target bottom faceted whale-tail/rudder protrusion is absent; hull remains source-like. Verdict: not achieved.

overfit逐样本：F-score优于源基线0/8；CD优于源基线2/8。单seed小诊断不是泛化结果。

lora逐样本：F-score优于源基线0/8；CD优于源基线0/8。单seed小诊断不是泛化结果。

## 未见物体benchmark与对照

Edit3D-Bench40物体，添加20/删除20，每物体3seeds，50步，网格解码分辨率256。CD/F-score以16384表面点、源导出的共同归一化坐标计算，无ICP；native单视角CFG5，编辑四视角guidance1，输入并不完全对等。

|模式|完成/期望|有效率|CD平方和↓|F-score@0.05↑|评分参照|
|---|---:|---:|---:|---:|---|
|source|120/120|1.000|0.014155|0.718601|target|
|vae|120/120|1.000|0.017408|0.673173|target|
|native|120/120|1.000|0.062493|0.279547|target|
|random|120/120|1.000|0.015178|0.704222|target|
|connector|120/120|1.000|0.046906|0.656865|target|
|lora|120/120|1.000|0.016409|0.663540|target|
|no_source|120/120|1.000|0.167191|0.145880|target|
|noop|120/120|1.000|0.026651|0.865324|source|
|shuffled|120/120|1.000|0.016749|0.665361|target|

noop summary以source为参照，其余以target为参照；不能把noop该列直接与编辑目标分数比较。

### 同样本成对比较

下表delta=前者-后者，CD为负/F-score为正有利于前者。仅共同有效样本，三seed先按物体平均，1000次物体bootstrap计算95%区间。无效网格没有填零。

|比较|有效样本|物体|ΔCD及95%CI|ΔF-score及95%CI|
|---|---:|---:|---|---|
|connector vs random|120|40|0.031728 [0.004682027273652941, 0.07877411152453832]|-0.047357 [-0.09075817487197876, -0.015361379520696513]|
|lora vs connector|120|40|-0.030498 [-0.07767568567405926, -0.003490606630589572]|0.006675 [-0.036001546378945495, 0.05975730135042698]|
|lora vs native|120|40|-0.046084 [-0.059312266969071555, -0.03533008533010377]|0.383994 [0.3127311669112698, 0.4597664620155939]|
|lora vs source|120|40|0.002253 [0.0005731046472303502, 0.00391679886328916]|-0.055060 [-0.0801077631143326, -0.03224005176769085]|
|lora vs no_source|120|40|-0.150783 [-0.17563968706109073, -0.12668246443601827]|0.517661 [0.4421058691680098, 0.5895995340279709]|
|lora vs noop|120|40|-0.022598 [-0.05737657692098895, -0.0030353272662795084]|0.005526 [-0.007597697627186017, 0.01840422567148015]|
|lora vs shuffled|120|40|-0.000340 [-0.0009906185097951577, 8.695853354900376e-05]|-0.001821 [-0.00424509271402526, 0.00025361168910420794]|

变化区域/保留区域是ground-truth辅助表面距离proxy，并非人工部件mask。删除任务容易因大部分几何不变获得高整体分；必须检查被删部件是否仍在。no_source推理时去条件还引入分布变化。

## 视觉检查与查看入口

预先选定add/remove各3例，seed20261013，AI查看四视角，不是用户本人审核，不能估计全体语义成功率。

- bench_add_45d32f8e2df34ce0b9a71cca74f79078 / Add a power cord extending from the base：Power cord is absent; lamp stem lost, base and shade disconnected, shade rim has holes. Correct and shuffled outputs are visually very similar. Verdict: failed.

- bench_add_8ca6b76e676b4504b5f9c5580a751ec3 / Add a crafted lid with a small knob on top of the vase：Crafted lid and knob are absent; vase opening/rim is distorted. Shuffled also lacks the lid; small rim differences do not implement requested addition. Verdict: failed.

- bench_add_9bc76559bccb409d9a8e6dc69b50977b / Add a roof luggage rack on top of the limousine：No coherent roof luggage rack is visible; car surface has holes and roof/body damage. Correct and shuffled outputs look similar; noop also has damage. Verdict: failed.

- bench_remove_81615480737c495785a6aecee0184e10 / Remove topknot hair：Head retains a raised hair-like/top protrusion; does not clearly match the bald target across views. Body identity mostly retained. Correct, noop, shuffled head silhouettes remain similar. Verdict: not clearly achieved.

- bench_remove_dd841d02b6164337a486de13c545b15d / Remove the grip：Grip remains visible; body and magazine have extensive holes/fragments. Correct and shuffled outputs show similar retained grip and breakage. Verdict: failed.

- bench_remove_82d7a79c62f94c979f02c2ced6126780 / Remove lid from the container：Container still has a closed lid/disc-like top; small knob removed but target opening not produced. Correct and shuffled outputs are near identical. Verdict: failed.

渲染恢复：检查336组四视角，修复26组，剩余缺图0组。原始render_error保留为历史。

查看全部40张图：/root/userdata/zc/hunyuan3d-buffalo-1.0-editing/results/visual_comparisons；图行依次为source、target、vae、native、random、connector、lora、no_source、noop、shuffled。
训练8对比较图：/root/userdata/zc/hunyuan3d-buffalo-1.0-editing/results/training_geometry_comparisons。GLB在results/evaluation/<mode>/和results/training_geometry/<checkpoint>/，可用Blender旋转查看。

## 复现、日志与限制

- environment/run_state.json保存每阶段命令、代码提交、GPU、时间、退出码、预算；environment/recovery_before.json保留恢复前状态。
- logs/stage_*.log、train_*.jsonl、recovery_*.log保留原失败和恢复记录。
- results/summaries/checkpoint_integrity.json、evaluation_integrity.json为权重和全部导出网格审计。
- controlled_comparisons.json与instruction_sensitivity.json包含完整量化对照。
- 不改变权重、manifest、样本划分或已完成baseline。
- 恢复命令：environment/venvs/shape/bin/python scripts/recover_experiment.py（只能在现有进程退出后执行，同一pilot锁/累计24h预算）；已成功GPU阶段跳过。
- 人工复核应同时检查指令执行、非编辑结构保留、noop/shuffled差异；不要把loss拟合推断为编辑成功。

## 各模式含义



所有模式使用固定的40个未见物体；每个物体三个seed。编辑模式使用相同50步Euler采样与源坐标系。源和目标没有独立归一化或ICP。

|模式|实际执行|用途|
|---|---|---|
|source|直接输出原始源网格，不通过模型|不编辑基线；判断模型是否优于复制输入|
|vae|源网格编码为Shape-VAE latent，再直接解码|检查几何压缩/重建误差；不运行DiT，不做指令编辑|
|native|官方Hunyuan3D2.1 image-to-shape，源物体第一个视角，官方图像条件，CFG5|原始生成基线；不接Qwen或指令，不是同等输入的编辑模型|
|random|预训练冻结DiT＋未训练随机MLP连接器＋源VAE latent；输入正确Qwen特征|编辑接口在没有适配训练时的基线；随机的是连接器，不是整个模型|
|connector|训练500步MLP连接器，DiT保持冻结；正确指令＋源latent|检查只训练语义连接器的贡献|
|lora|以connector_best初始化，再训练连接器和DiT注意力LoRA最多2000步；评估lora_best(step1920)|完整近似编辑方法；VLM/VAE/DiT原始主干权重冻结|
|no_source|同lora_best，移除直接拼入DiT的源VAE latent tokens|检查直接几何条件的作用；Qwen仍看源四视角，因此并非完全没有源物体信息|
|noop|同lora_best，源物体不变，将指令替换为Keep the geometry unchanged.|检查保持不变能力；summary主要对源网格评分，其余模式主要对编辑目标评分|
|shuffled|同lora_best，保留源四视角和源latent，将指令替换为manifest下一物体的指令（循环映射）|检验是否利用指令；不是打乱图片、几何或VLM token，也不是新的训练模型|

Qwen2.5-VL-7B-Instruct的条件由源四视角与指令共同前向得到，取最后一层所有有效输入token隐藏状态。它不生成文本回答。本实验没有使用闭源Buffalo VLM或官方编辑权重。

正确指令优于shuffled/noop并且对应添加/删除实际可见，才能支持指令有效性。与native相比更好、velocity loss下降、网格能导出，都不足以单独证明编辑成功。no_source为推理时去条件消融，存在分布变化；不能解释为另一个单独训练模型。
