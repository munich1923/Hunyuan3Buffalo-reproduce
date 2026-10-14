"""Complete existing evaluations/geometry diagnostics under original lock and 24h budget."""
import datetime,fcntl,json,os,subprocess,traceback
from remote_runtime import ROOT,configure
from run_pilot import run,readstate,atomic,utc
configure()
PY=ROOT/'environment/venvs/shape/bin/python'
def main():
    lock=(ROOT/'environment/pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=readstate();state.pop('error',None);state.pop('waiting_for',None)
    status={'started_utc':utc(),'commands':[],'ok':False}
    path=ROOT/'environment/recovery_status.json'
    try:
        for attempt in range(3):
            name='eval_shuffled' if attempt==0 else 'eval_shuffled_resume_'+str(attempt)
            run(state,name,PY,'evaluate_editor.py',['--mode','shuffled','--seconds','3500','--skip-render'],gpu=True,cap=3600)
            summary=json.loads((ROOT/'results/evaluation/shuffled/summary.json').read_text())
            if summary['complete']:break
        if not summary['complete']:raise RuntimeError('Shuffled evaluation incomplete after capped resumes')
        for checkpoint in ['overfit','lora']:
            run(state,'train_geometry_'+checkpoint,PY,'evaluate_training_fit.py',['--checkpoint',checkpoint,'--seconds','1500','--skip-render'],gpu=True,cap=1800)
            data=json.loads((ROOT/'results/summaries'/('training_geometry_'+checkpoint+'.json')).read_text())
            if not data['complete']:raise RuntimeError('Training geometry diagnostic incomplete')
        run(state,'repair_result_renders',PY,'repair_result_renders.py',gpu=True,cap=7200)
        state.update(status='completed',training_geometry_diagnostics=True,updated_utc=utc());atomic(state)
        for script in ['audit_checkpoints.py','audit_evaluations.py','compare_evaluations.py','analyze_instruction_sensitivity.py','build_comparison_sheets.py','plot_training.py','summarize_pilot.py','write_final_report.py']:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            with (ROOT/'logs'/('recovery_'+script.replace('.py','.log'))).open('a') as log:
                ret=subprocess.run([str(PY),str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            status['commands'].append({'script':script,'exit_code':ret});path.write_text(json.dumps(status,indent=2)+'\n')
            if ret:raise RuntimeError(script+' failed')
        status['ok']=True
    except Exception as error:
        status['error']=repr(error);state.update(status='stopped',error=repr(error),updated_utc=utc());atomic(state);traceback.print_exc();raise
    finally:
        status['ended_utc']=utc();path.write_text(json.dumps(status,indent=2)+'\n')
        (ROOT/'environment/finalization_status.json').write_text(json.dumps(status,indent=2)+'\n')
if __name__=='__main__':main()
