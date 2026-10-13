"""Finish durable post-run diagnostics/CPU summaries after the pilot exits."""
import datetime,json,os,subprocess,time
from pathlib import Path
from remote_runtime import ROOT
from run_pilot import readstate
PY=ROOT/'environment/venvs/shape/bin/python'
def alive(pid):
    path=Path('/proc')/str(pid)
    try:return path.exists() and ') Z' not in (path/'stat').read_text()
    except FileNotFoundError:return False
def main():
    pid=int((ROOT/'environment/pilot_pid.txt').read_text())
    while alive(pid):time.sleep(30)
    state=readstate();record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pilot_status':state['status'],'commands':[]}
    scripts=[]
    if state['status']=='completed':scripts.append('run_training_geometry.py')
    scripts.extend(['audit_checkpoints.py','audit_evaluations.py','compare_evaluations.py','analyze_instruction_sensitivity.py','build_comparison_sheets.py','plot_training.py','summarize_pilot.py','write_final_report.py'])
    for script in scripts:
        env=os.environ.copy()
        if script!='run_training_geometry.py':env['CUDA_VISIBLE_DEVICES']=''
        cmd=[str(PY),str(ROOT/'scripts'/script)]
        print('START',script,flush=True)
        with (ROOT/'logs'/('final_'+script.replace('.py','.log'))).open('a') as out:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        record['commands'].append({'script':script,'exit_code':r.returncode})
        (ROOT/'environment/finalization_status.json').write_text(json.dumps(record,indent=2)+chr(10))
        print('END',script,r.returncode,flush=True)
    record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();record['ok']=all(x['exit_code']==0 for x in record['commands'])
    (ROOT/'environment/finalization_status.json').write_text(json.dumps(record,indent=2)+chr(10))
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
