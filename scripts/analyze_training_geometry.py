"""Per-pair training geometry comparisons; small diagnostic, not generalization estimate."""
import json,hashlib,numpy as np,trimesh
from remote_runtime import ROOT
from train_editor import load_records
rows={r['id']:r for r in load_records('train') if r['operation']=='add'}
report={'expected_pairs':8,'note':'One seed,8 known add training pairs. F-score/CD deltas compare generated to target against unchanged source to target; region values are surface proxies, not semantic part masks.','checkpoints':{}}
for stage in ['overfit','lora']:
    folder=ROOT/'results/training_geometry'/stage
    records=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
    seen=set();pairs=[]
    for r in records:
        assert r['id'] not in seen;seen.add(r['id'])
        row=rows[r['id']];assert r['seed']==20261013
        p={'id':r['id'],'instruction':row['instruction'],'valid':r['valid']}
        if r['valid']:
            mesh=trimesh.load(r['path'],force='mesh');assert len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
            p.update(mesh_sha256=hashlib.sha256(open(r['path'],'rb').read()).hexdigest(),vertices=len(mesh.vertices),faces=len(mesh.faces))
            for m in ['fscore_005','cd_squared_sum']:
                p['delta_'+m]=r['target_metrics'][m]-r['unchanged_source_target_metrics'][m]
            p['source_similarity']=r['source_metrics'];p['region_proxies']=r['preservation_metrics']
        pairs.append(p)
    assert len(records)==8
    valid=[p for p in pairs if p['valid']]
    report['checkpoints'][stage]={'attempted':len(records),'valid':len(valid),'better_fscore_pairs':sum(p['delta_fscore_005']>0 for p in valid),'better_cd_pairs':sum(p['delta_cd_squared_sum']<0 for p in valid),'pairs':pairs}
(ROOT/'results/summaries/training_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='pairs'} for k,v in report['checkpoints'].items()}))
