"""CPU contact sheets from existing benchmark renders; never creates GPU work."""
import json,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from remote_runtime import ROOT
MODES=["source","vae","native","random","connector","lora","no_source","noop","shuffled"]
def main():
    rows=[json.loads(x) for x in (ROOT/"assets/manifests/test_prepared.jsonl").read_text().splitlines()]
    out=ROOT/"results/visual_comparisons";out.mkdir(parents=True,exist_ok=True)
    index=[]
    size=256;label=130;top=90;height=280
    try:font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    except OSError:font=ImageFont.load_default()
    for row in rows:
        entries=[("input source",row["source_views"]),("edited target",row["target_views"])]
        for mode in MODES[1:]:
            folder=ROOT/"results/evaluation"/mode/(row["id"]+"_views")
            entries.append((mode,[str(folder/(str(a)+".png")) for a in [0,90,180,270]]))
        canvas=Image.new("RGB",(label+size*4,top+height*len(entries)),"white");d=ImageDraw.Draw(canvas)
        d.text((8,5),row["id"]+" / "+row["operation"],fill="black",font=font)
        for n,line in enumerate(textwrap.wrap(row["instruction"],width=115)):
            d.text((8,25+n*18),line,fill="black",font=font)
        available=[]
        for i,(name,paths) in enumerate(entries):
            y=top+i*height;d.text((8,y+100),name,fill="black",font=font)
            found=False
            for j,path in enumerate(paths):
                if Path(path).exists():
                    im=Image.open(path).convert("RGBA")
                    bg=Image.new("RGBA",im.size,"white");bg.alpha_composite(im)
                    canvas.paste(bg.convert("RGB").resize((size,size)),(label+j*size,y))
                    found=True
                else:d.text((label+j*size+60,y+100),"pending",fill="gray",font=font)
            if found:available.append(name)
        path=out/(row["id"]+".jpg");canvas.save(path,quality=90)
        index.append({"id":row["id"],"operation":row["operation"],"instruction":row["instruction"],"path":str(path),"available":available,"seed":20261013})
    (ROOT/"results/summaries/visual_comparison_index.json").write_text(json.dumps(index,indent=2)+"\n")
    print("Contact sheets:",len(index))
if __name__=="__main__":main()
