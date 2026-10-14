"""Repair missing result views using isolated EGL workers; called under GPU budget lock."""
import json,subprocess,time,os
from pathlib import Path
from PIL import Image
from remote_runtime import ROOT,configure
configure()
PY=ROOT/'environment/venvs/shape/bin/python'
def complete(folder):
    try:
        for az in [0,90,180,270]:
            with Image.open(folder/(str(az)+'.png')) as im:
                im.load()
                if im.mode!='RGBA' or im.getextrema()[3][1]==0:return False
        return True
    except (OSError,FileNotFoundError):return False
def main():
    jobs=[]
    for mode in ['vae','native','random','connector','lora','no_source','noop','shuffled']:
        f=ROOT/'results/evaluation'/mode/'metrics.jsonl'
        for row in map(json.loads,f.read_text().splitlines()):
            if row['seed']==20261013 and row['valid']:
                jobs.append((mode,row['id'],Path(row['path']),f.parent/(row['id']+'_views')))
    for stage in ['overfit','lora']:
        f=ROOT/'results/training_geometry'/stage/'metrics.jsonl'
        for row in map(json.loads,f.read_text().splitlines()):
            if row['valid']:jobs.append(('train_'+stage,row['id'],Path(row['path']),f.parent/(row['id']+'_views')))
    report={'jobs':len(jobs),'repaired':[],'errors':[],'isolation':'one mesh per fresh EGL subprocess; geometry and original render errors preserved'}
    for mode,ident,mesh,folder in jobs:
        if complete(folder):continue
        for attempt in range(2):
            with (ROOT/'logs/render_recovery_workers.log').open('a') as log:
                try:
                    ret=subprocess.run([str(PY),str(ROOT/'scripts/render_one_mesh.py'),str(mesh),str(folder)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
                except subprocess.TimeoutExpired:ret='timeout'
            if ret==0 and complete(folder):break
        dest=report['repaired'] if ret==0 and complete(folder) else report['errors']
        dest.append({'mode':mode,'id':ident,'exit_code':ret});print(mode,ident,ret,flush=True)
        (ROOT/'results/summaries/render_recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    report['missing_after']=[{'mode':m,'id':i} for m,i,p,f in jobs if not complete(f)]
    report['ok']=not report['missing_after']
    (ROOT/'results/summaries/render_recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['ok']:raise RuntimeError('Result views remain missing')
if __name__=='__main__':main()
