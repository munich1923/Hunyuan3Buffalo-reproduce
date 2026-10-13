"""Durable evidence-based Markdown report from final summaries."""
import datetime,json
from remote_runtime import ROOT
def read(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else None
def fmt(x):return "pending" if x is None else f"{x:.6f}"
def main():
    state=read("environment/run_state.json")
    lines=["# Hunyuan3D Buffalo approximate editing pilot", "", "Generated: "+datetime.datetime.now(datetime.timezone.utc).isoformat(), "", "## Scope and execution", "", "Qwen2.5-VL-7B contextual input hidden states + MLP + frozen Hunyuan3D2.1 backbone with source VAE tokens and attention LoRA. This is an open replacement experiment, not official Buffalo weights or scores.", "", f"Run status: {state['status']}; allocated GPU hours for completed stages: {state['gpu_seconds']/3600:.4f} /24. Stop reason: {state.get('error','none')}", "", "## Fitting", "", "Velocity MSE is a diagnostic and does not establish geometry editing.", "", "|Stage|Steps|Initial train probe|Final train probe|Initial val|Final val|Val entries|", "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in ["overfit","connector","lora"]:
        r=read("environment/training_"+stage+".json")
        if r:lines.append(f"|{stage}|{r['steps']}|{fmt(r['initial_train'])}|{fmt(r['final_train'])}|{fmt(r['initial_val'])}|{fmt(r['final_val'])}|{r['final_validation_entries']}|")
    lines.extend(["", "Formal connector/LoRA validation covers128 entries; best-checkpoint selection used a fixed16-entry probe. Train1024 entries represent512 independent add pairs and their reverse tasks. Val128 represents64 separate pairs. AI reviewed30 final training pairs; remaining482 have automatic geometry QC only.", "", "## Held-out benchmark", "", "Edit3D-Bench40 objects, add20/remove20,3seeds,50steps. Native baseline uses one source view without instruction and CFG5; edited models use four views and guidance1. All distances are in the source-derived shared coordinates, without ICP.", "", "|Mode|Attempted/expected|Valid rate|CD squared sum|F-score .05|Reference|", "|---|---:|---:|---:|---:|---|"])
    summaries={}
    for mode in ["source","vae","native","random","connector","lora","no_source","noop","shuffled"]:
        r=read("results/evaluation/"+mode+"/summary.json")
        if r:
            summaries[mode]=r;m=r.get("all",{})
            lines.append(f"|{mode}|{r['attempted']}/{r['expected']}|{r['valid_rate']:.3f}|{fmt(m.get('cd_squared_sum'))}|{fmt(m.get('fscore_005'))}|{r['reference']}|")
        else:lines.append(f"|{mode}|pending|pending|pending|pending|pending|")
    lines.extend(["", "Per-mode scores may use different partial coverage. Controlled comparisons in controlled_comparisons.json restrict pairs to shared attempted samples and retain invalid-mesh rates. Distances are never filled with zero for failed meshes.", "", "## Paired evidence", "", "Negative CD delta and positive F-score delta favor the first model against edited targets. Confidence intervals use object-level bootstrap,1000 draws; the three seeds are averaged per object.", "", "|Comparison|Both-valid samples|Objects|Delta CD (95% CI)|Delta F-score (95% CI)|", "|---|---:|---:|---|---|"])
    comp=read("results/summaries/controlled_comparisons.json")
    if comp:
        for r in comp['paired_comparisons']:
            lines.append(f"|{r['first']} vs {r['second']}|{r['both_valid']}|{r.get('paired_objects',0)}|{fmt(r.get('mean_delta_cd_squared_sum'))} {r.get('object_bootstrap_95ci_cd_squared_sum')}|{fmt(r.get('mean_delta_fscore_005'))} {r.get('object_bootstrap_95ci_fscore_005')}|")
    lines.extend(["", "## Training geometry", "", "Eight original training add pairs, one seed, same50-step sampler. This separately checks whether velocity fitting produced target geometry."])
    for stage in ['overfit','lora']:
        r=read('results/summaries/training_geometry_'+stage+'.json')
        if r:lines.extend(["", f"{stage}: attempted{r['attempted']}/8, valid{r['valid']}; generated F-score={fmt(r.get('generated_fscore_005'))}, unchanged-source F-score={fmt(r.get('unchanged_source_fscore_005'))}; generated CD={fmt(r.get('generated_cd_squared_sum'))}."])
    lines.extend(["", "## Conclusions and limits", "", "Interface: official-weight forward/backward, frozen-backbone and LoRA gradient contracts passed. Formal first training source VAE gate passed; synthetic pure-FP16 box failure was retained separately.", "", "Fitting: report velocity-loss fitting separately from the training geometry above.", "", "Held-out editing: metrics and instruction controls require interpretation together with visual observations. Successful mesh export or lower velocity MSE alone is insufficient. Part preservation/addition/deletion metrics are ground-truth-assisted surface proxies, not human part masks. Sample inspection is AI review, not user signoff.", "", "## Evidence and rerun", "", "- Source commit82920d643c0dc2f7bfd7255f45f62d386edfe60c; source checkout src/Hunyuan3D-2.1.", "- environment/run_state.json records stage command, commit, GPU, UTC/Shanghai timestamps, duration, exit code and cumulative budget.", "- logs/stage_*.log and train_*.jsonl preserve run records; logs/downloads contain verified receipts.", "- results/summaries contain authoritative manifests, review decisions, cache and checkpoint/evaluation audits.", "- results/evaluation contains generated GLB, four-view renders, raw metrics and protocol.json; results/visual_comparisons has contact sheets.", "- results/checkpoints contain adapter initial/best/latest files; large files remain remote and ignored by Git.", "- Execute environment/venvs/shape/bin/python scripts/finalize_experiment.py only after the pilot runner has exited; it uses the same budget lock for optional training geometry. CPU analysis scripts can be rerun separately."])
    (ROOT/'results/summaries/final_report.md').write_text(chr(10).join(lines)+chr(10))
    print('Report written; semantic conclusions still require visual interpretation.')
if __name__=='__main__':main()
