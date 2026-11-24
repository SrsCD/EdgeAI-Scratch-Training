"""Read-only direct-vs-Stage5A comparison on four existing CCTV GIF frames."""
import csv,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stage5a_detect_video import CHECKPOINT,EXPECTED_SHA256,NAMES,check_checkpoint,frame_source,result_detections
from ultralytics import YOLO
import torch

SOURCE=ROOT/'data/demo_videos/cctv_test.gif'
RUN=ROOT/'runs/stage5a/mentor_demo_conf010'
OUT=ROOT/'reports/stage5b'
FRAMES=(5,7,11,19)

def csv_rows():
    with (RUN/'detections.csv').open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def normalized(row):
    return (int(row['class_id']),float(row['confidence']),
            *(float(row[k]) for k in ('x1','y1','x2','y2')))

def main():
    check_checkpoint()
    meta=json.loads((RUN/'run_metadata.json').read_text())
    if meta['input_sha256']!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or meta['checkpoint_sha256']!=EXPECTED_SHA256:
        raise SystemExit('Existing Stage 5A run provenance differs from source/checkpoint')
    saved=csv_rows()
    fps,hint,frames,_=frame_source(SOURCE)
    selected={i:frame for i,_,frame in frames if i in FRAMES}
    if set(selected)!=set(FRAMES):raise SystemExit('Representative frame missing')
    model=YOLO(str(CHECKPOINT))
    if tuple(model.names[i] for i in range(6))!=NAMES:raise SystemExit('Class order changed')
    device=0 if torch.cuda.is_available() else 'cpu'
    summary=[];low_person=[];direct_rows=[]
    for i in FRAMES:
        frame=selected[i];height,width=frame.shape[:2]
        result=model.predict(source=frame,imgsz=640,device=device,conf=.10,iou=.70,verbose=False)[0]
        direct=[(d['class_id'],d['confidence'],*(d[k] for k in ('x1','y1','x2','y2')))
                for d in result_detections(result,width,height)]
        rows=[normalized(r) for r in saved if int(r['frame_number'])==i]
        direct.sort();rows.sort()
        if len(direct)!=len(rows):raise SystemExit(f'Frame {i}: direct/CSV count differs')
        max_delta=max((abs(a-b) for d,s in zip(direct,rows) for a,b in zip(d[1:],s[1:])),default=0)
        class_agree=all(d[0]==s[0] for d,s in zip(direct,rows))
        if not class_agree or max_delta>0.0001:raise SystemExit(f'Frame {i}: direct/CSV box or score differs: {max_delta}')
        for d in direct:
            direct_rows.append({'frame_number':i,'class_id':d[0],'class_name':NAMES[d[0]],
                                'confidence':f'{d[1]:.8f}',
                                **{k:f'{v:.6f}' for k,v in zip(('x1','y1','x2','y2'),d[2:])}})
        low=model.predict(source=frame,imgsz=640,device=device,conf=.001,iou=.70,verbose=False)[0]
        people=[d for d in result_detections(low,width,height) if d['class_id']==0]
        people.sort(key=lambda d:-d['confidence'])
        for d in people:
            low_person.append({'frame_number':i,'confidence':f"{d['confidence']:.8f}",
                               **{k:f"{d[k]:.4f}" for k in ('x1','y1','x2','y2')}})
        summary.append({'frame':i,'direct_at_0_10':len(direct),'csv_at_0_10':len(rows),
                        'max_absolute_numeric_delta':max_delta,'person_at_0_10':sum(d[0]==0 for d in direct),
                        'person_at_0_001':len(people),
                        'max_person_confidence_at_0_001':people[0]['confidence'] if people else None})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'direct_comparison.json').write_text(json.dumps({'source':str(SOURCE),'checkpoint_sha256':EXPECTED_SHA256,
        'source_fps':fps,'source_frame_count':hint,'frames':summary},indent=2)+'\n')
    with (OUT/'low_confidence_person_boxes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['frame_number','confidence','x1','y1','x2','y2']);w.writeheader();w.writerows(low_person)
    with (OUT/'direct_predictions_0_10.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['frame_number','class_id','class_name','confidence','x1','y1','x2','y2']);w.writeheader();w.writerows(direct_rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
