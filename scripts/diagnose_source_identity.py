import json,sys,numpy as np,trimesh
from pathlib import Path
p=Path('/root/userdata/zc/hunyuan3d-buffalo-1.0-editing');sys.path.insert(0,str(p/'scripts'))
from metrics import compare,preservation
rows=[json.loads(x) for x in (p/'assets/manifests/test_prepared.jsonl').read_text().splitlines()]
result=[]
for r in rows:
 src=trimesh.load(r['source_normalized'],force='mesh');tar=trimesh.load(r['target_normalized'],force='mesh')
 result.append({'id':r['id'],'operation':r['operation'],'target_metrics':compare(src,tar),'preservation_metrics':preservation(src,src,tar)})
summary={'note':'CPU identity-source baseline at first seed; final 3-seed benchmark is separate.','count':len(result),'by_operation':{}}
for op in ['add','remove','all']:
 x=[r for r in result if op=='all' or r['operation']==op]
 summary['by_operation'][op]={m:float(np.mean([r['target_metrics'][m] for r in x])) for m in ['cd_squared_sum','fscore_005']}
summary['records']=result
(p/'results/summaries/source_identity_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
