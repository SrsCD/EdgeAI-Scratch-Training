"""Small validation-only scratch-checkpoint demo audit; no labels are changed."""
import csv, hashlib, json, random, sys
from collections import Counter
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stage5a_detect_video import CHECKPOINT, EXPECTED_SHA256, NAMES, check_checkpoint
DATA=ROOT/'data/experimental/stage3c_v1'
OUT=ROOT/'reports/stage5e'
TARGET=(5,3,2,1)
COLORS={0:'#47bb6b',1:'#18a5db',2:'#ff9800',3:'#9c27b0',4:'#bc4892',5:'#e32939'}

def boxes(path,w,h):
    result=[]
    for line in path.read_text().splitlines():
        c,cx,cy,bw,bh=map(float,line.split())
        result.append((int(c),(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h,1.0))
    return result

def iou(a,b):
    x=max(0,min(a[3],b[3])-max(a[1],b[1])); y=max(0,min(a[4],b[4])-max(a[2],b[2]))
    u=(a[3]-a[1])*(a[4]-a[2])+(b[3]-b[1])*(b[4]-b[2])-x*y
    return x*y/u if u else 0

def draw_pair(path,truth,pred,out):
    im=Image.open(path).convert('RGB'); w,h=im.size
    scale=min(500/w,400/h,1.5); sz=(round(w*scale),round(h*scale))
    panels=[]
    for entries,word in ((truth,'PROVISIONAL LABELS'),(pred,'PREDICTIONS 0.25')):
        panel=im.resize(sz); d=ImageDraw.Draw(panel)
        for c,x1,y1,x2,y2,conf in entries:
            color=COLORS[c]; d.rectangle((x1*scale,y1*scale,x2*scale,y2*scale),outline=color,width=3)
            d.text((max(0,x1*scale),max(0,y1*scale-12)),f'{NAMES[c]} {conf:.2f}' if word.startswith('PRED') else NAMES[c],fill=color,stroke_width=1,stroke_fill='white')
        panels.append(panel)
    canvas=Image.new('RGB',(max(2*sz[0],650),sz[1]+42),'white')
    d=ImageDraw.Draw(canvas);canvas.paste(panels[0],(0,32));canvas.paste(panels[1],(sz[0],32))
    d.text((4,5),'LEFT: PROVISIONAL LABELS    RIGHT: MODEL PREDICTIONS 0.25',fill='black')
    canvas.save(out,quality=90)

def main():
    check_checkpoint(); OUT.mkdir(parents=True,exist_ok=True);(OUT/'examples').mkdir(exist_ok=True)
    ledger=[r for r in csv.DictReader((DATA/'ledger.csv').open(encoding='utf-8-sig')) if r['status']=='included' and r['assigned_split']=='val']
    byid={r['id']:r for r in ledger}; chosen=[]; seen=set()
    for c in TARGET:
        candidates=[]
        for r in ledger:
            if r['id'] in seen:continue
            label=DATA/'labels/val'/f"{r['id']}.txt"
            if any(int(line.split()[0])==c for line in label.read_text().splitlines()):candidates.append(r)
        random.Random(500+c).shuffle(candidates)
        selected=[]; groups=set()
        for r in candidates:
            g=r['provisional_component_id']
            if g in groups:continue
            groups.add(g);selected.append(r)
            if len(selected)==12:break
        if len(selected)<12:raise RuntimeError(f'Insufficient class {c}')
        chosen.extend((r,c) for r in selected);seen.update(r['id'] for r in selected)
    model=YOLO(str(CHECKPOINT))
    if tuple(model.names[i] for i in range(6))!=NAMES:raise RuntimeError('Wrong class map')
    device=0 if torch.cuda.is_available() else 'cpu'
    rows=[]
    for r,focus in chosen:
        path=DATA/'images/val'/f"{r['id']}.jpg"; frame=cv2.imread(str(path));h,w=frame.shape[:2]
        truth=boxes(DATA/'labels/val'/f"{r['id']}.txt",w,h)
        result=model.predict(source=frame,imgsz=640,device=device,conf=.25,iou=.70,verbose=False)[0]
        pred=[]
        for box in result.boxes:
            c=int(box.cls[0]);x1,y1,x2,y2=map(float,box.xyxy[0].tolist())
            pred.append((c,x1,y1,x2,y2,float(box.conf[0])))
        focus_t=[x for x in truth if x[0]==focus];focus_p=[x for x in pred if x[0]==focus]
        pairs=sorted(((iou(a,b),a,b) for a in focus_p for b in focus_t),key=lambda x:-x[0])
        best=pairs[0][0] if pairs else 0
        largest=max(( ((p[3]-p[1])*(p[4]-p[2])/(w*h)) for p in pred),default=0)
        out=OUT/'examples'/f"{NAMES[focus]}_{r['id']}.jpg"
        draw_pair(path,truth,pred,out)
        rows.append({'id':r['id'],'focus_class_id':focus,'focus_class':NAMES[focus],
            'source':r['source'],'source_group':r['source_group_hint'],
            'provisional_focus_boxes':len(focus_t),'predicted_focus_boxes':len(focus_p),
            'best_focus_iou':f'{best:.4f}','all_prediction_count':len(pred),
            'largest_predicted_area_fraction':f'{largest:.4f}',
            'prediction_classes':';'.join(NAMES[x[0]] for x in pred),
            'prediction_scores':';'.join(f'{x[5]:.3f}' for x in pred),
            'image':str(path.relative_to(ROOT)),'comparison':str(out.relative_to(ROOT)),
            'provisional_only':'Yes'})
    with (OUT/'audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    for focus in TARGET:
        subset=[r for r in rows if r['focus_class_id']==focus]
        sheet=Image.new('RGB',(1800,((len(subset)+2)//3)*470),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(subset):
            img=Image.open(ROOT/r['comparison']);img.thumbnail((590,425));x=j%3*600;y=j//3*470
            sheet.paste(img,(x,y));d.text((x,y+430),f"{r['id']} IoU={r['best_focus_iou']} area={r['largest_predicted_area_fraction']}",fill='black')
        sheet.save(OUT/f'{NAMES[focus]}_contact_sheet.jpg',quality=88)
    (OUT/'provenance.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT.relative_to(ROOT)),
        'checkpoint_sha256':EXPECTED_SHA256,'split':'stage3c_v1/val','confidence':.25,
        'iou_nms':.70,'imgsz':640,'seed_per_class':'500+class_id','sample_per_class':12,
        'classes':[NAMES[c] for c in TARGET],'annotation_status':'provisional, not human verified'},indent=2)+'\n')
    print(Counter((r['focus_class'],round(float(r['best_focus_iou']),1)) for r in rows))

if __name__=='__main__':main()
