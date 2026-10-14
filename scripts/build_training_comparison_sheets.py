"""CPU comparison sheets for the8 original training-add diagnostic objects."""
import json
from PIL import Image,ImageDraw
from remote_runtime import ROOT
from train_editor import load_records
from repair_result_renders import complete
rows=[r for r in load_records('train') if r['operation']=='add'][:8]
out=ROOT/'results/training_geometry_comparisons';out.mkdir(exist_ok=True)
index=[]
for row in rows:
    entries=[('source',row['source_views']),('target',row['target_views'])]
    for stage in ['overfit','lora']:
        f=ROOT/'results/training_geometry'/stage/(row['id']+'_views')
        entries.append((stage,[str(f/(str(a)+'.png')) for a in [0,90,180,270]]))
    canvas=Image.new('RGB',(1154,1210),'white');d=ImageDraw.Draw(canvas)
    d.text((8,5),row['id']+' '+row['instruction'],fill='black')
    for i,(label,paths) in enumerate(entries):
        y=90+i*280;d.text((8,y+100),label,fill='black')
        for j,p in enumerate(paths):
            im=Image.open(p).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
            canvas.paste(bg.convert('RGB').resize((256,256)),(130+j*256,y))
    path=out/(row['id']+'.jpg');canvas.save(path,quality=90);index.append({'id':row['id'],'instruction':row['instruction'],'path':str(path)})
(ROOT/'results/summaries/training_geometry_visual_index.json').write_text(json.dumps(index,indent=2)+'\n')
print('Training contact sheets:',len(index))
