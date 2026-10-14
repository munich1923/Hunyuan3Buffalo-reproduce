"""Generate final Chinese report using current summaries and honest schema compatibility."""
import datetime,json
from remote_runtime import ROOT
def read(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else None
def fmt(x):return '未完成' if x is None else f'{x:.6f}'
def main():
    state=read('environment/run_state.json');comp=read('results/summaries/controlled_comparisons.json');sens=read('results/summaries/instruction_sensitivity.json')
    summaries={m:read('results/evaluation/'+m+'/summary.json') for m in ['source','vae','native','random','connector','lora','no_source','noop','shuffled']}
    complete=all(r and r['complete'] for r in summaries.values())
    lines=['# Hunyuan3D Buffalo 编辑能力近似复现实验报告','','生成 UTC：'+datetime.datetime.now(datetime.timezone.utc).isoformat(),'生成 Asia/Shanghai：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'','## 结论与执行状态','',f"执行状态：{state['status']}；9组benchmark是否全覆盖：{complete}；累计单卡GPU分配时间：{state['gpu_seconds']/3600:.4f}/24小时。",'','本实验使用开源Qwen2.5-VL替代闭源Buffalo VLM，不是官方编辑权重或官方指标复现。']
    pairs=comp.get('paired_comparisons',[]) if comp else []
    vs_source=next((x for x in pairs if x['first']=='lora' and x['second']=='source'),{})
    if vs_source.get('mean_delta_fscore_005',1)<0 and vs_source.get('mean_delta_cd_squared_sum',-1)>0:
        lines+=['','**当前近似方法没有复现可靠的未见物体指令编辑：LoRA对目标的整体几何分数低于直接保留源物体。**']
    if sens:
        x=sens['comparisons'].get('shuffled',{}).get('all',{})
        lines+=['',f"正确与打乱指令生成网格：共同有效样本{x.get('both_valid',0)}，物体{x.get('paired_objects',0)}；输出间F-score@0.05={fmt(x.get('fscore_005'))}，CD平方和={fmt(x.get('cd_squared_sum'))}。相似度高支持指令敏感性弱，但不是完整语义成功率。"]
    lines+=['','接口通过、velocity loss下降、几何拟合、未见物体编辑有效性必须分别判断。网格有效率只代表可加载且坐标/索引合法，不保证闭合、连通或语义正确。','原计划要求8样本过拟合诊断先于扩大训练；先前实际gate仅要求固定噪声velocity loss下降10%，未在正式训练前验证几何记忆。这是验收口径的限制，本轮补做几何检查；不能把此前通过loss gate称为完整过拟合成功。','',"原eval_shuffled异常退出码-6(SIGABRT)，退出前记录EGL全黑帧错误；日志没有足够证据锁定原生崩溃根因。恢复采用生成与渲染分离、每网格独立EGL进程，保留原39条记录和失败时间。原报告因旧overfit报告缺final_validation_entries失败；现在按当时脚本实际16条验证处理，原始报告未改写。",'','## 数据、环境与训练','',"源码固定82920d643c0dc2f7bfd7255f45f62d386edfe60c；独立Shape/VLM环境，PyTorch2.5.1+cu124。权重revision、下载大小/SHA和依赖锁文件留存。Nano3D3000候选中1798独立add配对通过自动QC；train1024条是512独立配对正反向，val128条是64另配对。AI检查46候选、排除16，最终30配对视觉检查，其余482配对只有自动QC。",'','Qwen输入源四视角+指令，使用最后层所有有效输入token隐藏状态，不生成文本回答。MLP3584→1024→1024，pool1370；源Shape-VAE token与噪声目标拼接，只预测目标速度。VLM/VAE/DiT原权重冻结；attention LoRA rank8/alpha16，LoRA阶段同时训练连接器。10%保持不变训练。','',"正式首训练源VAE gate F-score@0.05=0.996386，只验证该例。编辑VAE使用FP32权重/输入+FP16 autocast；DiT BF16/adapters FP32。",'','|阶段|训练步数|初始train probe|最终train probe|初始val|最终val|最终验证条目|','|---|---:|---:|---:|---:|---:|---:|']
    for stage in ['overfit','connector','lora']:
        d=read('environment/training_'+stage+'.json')
        if d:
            n=d.get('final_validation_entries',16 if stage=='overfit' else 'unknown')
            lines.append(f"|{stage}|{d['steps']}|{fmt(d['initial_train'])}|{fmt(d['final_train'])}|{fmt(d['initial_val'])}|{fmt(d['final_val'])}|{n}|")
    lines+=['','正式connector/LoRA最终val覆盖128条；best选择使用固定16条probe。benchmark和几何诊断使用best，LoRA best为step1920，训练末尾为step2000。overfit_best为step200。','','## 8对训练样本的实际几何','',"同一8个独立原始add训练配对，每模型单seed20261013，50步采样；不追加训练。对比实际目标和不编辑源物体。",'','|checkpoint|完成/期望|有效|生成CD↓|源不编辑CD↓|生成F-score↑|源不编辑F-score↑|','|---|---:|---:|---:|---:|---:|---:|']
    for stage in ['overfit','lora']:
        d=read('results/summaries/training_geometry_'+stage+'.json')
        if d:
            lines.append(f"|{stage}|{d['attempted']}/8|{d['valid']}|{fmt(d.get('generated_cd_squared_sum'))}|{fmt(d.get('unchanged_source_cd_squared_sum'))}|{fmt(d.get('generated_fscore_005'))}|{fmt(d.get('unchanged_source_fscore_005'))}|")
            verdict='未优于不编辑源物体，不能认定学会几何记忆。' if d.get('generated_fscore_005',0)<=d.get('unchanged_source_fscore_005',0) else '整体F-score优于源基线；仍需逐样本检查编辑部件和非编辑区域，不能只凭一个平均分认定记忆成功。'
            lines+=['',stage+'：'+verdict,'']
        else:lines.append('|'+stage+'|未完成|未完成|未完成|未完成|未完成|未完成|')
    lines+=['','## 未见物体benchmark与对照','',"Edit3D-Bench40物体，添加20/删除20，每物体3seeds，50步，网格解码分辨率256。CD/F-score以16384表面点、源导出的共同归一化坐标计算，无ICP；native单视角CFG5，编辑四视角guidance1，输入并不完全对等。",'','|模式|完成/期望|有效率|CD平方和↓|F-score@0.05↑|评分参照|','|---|---:|---:|---:|---:|---|']
    for m,d in summaries.items():
        if d:
            v=d.get('all',{});lines.append(f"|{m}|{d['attempted']}/{d['expected']}|{d['valid_rate']:.3f}|{fmt(v.get('cd_squared_sum'))}|{fmt(v.get('fscore_005'))}|{d['reference']}|")
    lines+=['','noop summary以source为参照，其余以target为参照；不能把noop该列直接与编辑目标分数比较。','',"### 同样本成对比较",'',"下表delta=前者-后者，CD为负/F-score为正有利于前者。仅共同有效样本，三seed先按物体平均，1000次物体bootstrap计算95%区间。无效网格没有填零。",'','|比较|有效样本|物体|ΔCD及95%CI|ΔF-score及95%CI|','|---|---:|---:|---|---|']
    for d in pairs:
        lines.append(f"|{d['first']} vs {d['second']}|{d['both_valid']}|{d.get('paired_objects',0)}|{fmt(d.get('mean_delta_cd_squared_sum'))} {d.get('object_bootstrap_95ci_cd_squared_sum')}|{fmt(d.get('mean_delta_fscore_005'))} {d.get('object_bootstrap_95ci_fscore_005')}|")
    lines+=['','变化区域/保留区域是ground-truth辅助表面距离proxy，并非人工部件mask。删除任务容易因大部分几何不变获得高整体分；必须检查被删部件是否仍在。no_source推理时去条件还引入分布变化。','',"## 视觉检查与查看入口",'',"预先选定add/remove各3例，seed20261013，AI查看四视角，不是用户本人审核，不能估计全体语义成功率。"]
    review=read('results/summaries/benchmark_visual_observations.json')
    if review:
        for d in review['observations']:lines+=['',f"- {d['id']} / {d['instruction']}：{d['requested_edit']} {d['instruction_controls']} Verdict: {d['verdict']}."]
    rg=read('results/summaries/render_recovery.json')
    if rg:lines+=['',f"渲染恢复：检查{rg['jobs']}组四视角，修复{len(rg['repaired'])}组，剩余缺图{len(rg.get('missing_after',[]))}组。原始render_error保留为历史。"]
    lines+=['','查看全部40张图：'+str(ROOT/'results/visual_comparisons')+'；图行依次为source、target、vae、native、random、connector、lora、no_source、noop、shuffled。',"训练8对比较图："+str(ROOT/'results/training_geometry_comparisons')+'。GLB在results/evaluation/<mode>/和results/training_geometry/<checkpoint>/，可用Blender旋转查看。',"","## 复现、日志与限制",'',"- environment/run_state.json保存每阶段命令、代码提交、GPU、时间、退出码、预算；environment/recovery_before.json保留恢复前状态。",'- logs/stage_*.log、train_*.jsonl、recovery_*.log保留原失败和恢复记录。','- results/summaries/checkpoint_integrity.json、evaluation_integrity.json为权重和全部导出网格审计。','- controlled_comparisons.json与instruction_sensitivity.json包含完整量化对照。','- 不改变权重、manifest、样本划分或已完成baseline。','- 恢复命令：environment/venvs/shape/bin/python scripts/recover_experiment.py（只能在现有进程退出后执行，同一pilot锁/累计24h预算）；已成功GPU阶段跳过。','- 人工复核应同时检查指令执行、非编辑结构保留、noop/shuffled差异；不要把loss拟合推断为编辑成功。','']
    lines+=['## 各模式含义','',(ROOT/'results/summaries/evaluation_modes.md').read_text()]
    (ROOT/'results/summaries/final_report.md').write_text('\n'.join(lines)+'\n')
    print('Final evidence report written, complete benchmark:',complete)
if __name__=='__main__':main()
