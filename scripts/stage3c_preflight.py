"""Mechanical gate for provisional Stage 3C data; never implies approval."""
import csv,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/experimental/stage3c_v1'
NAMES=['person','helmet','safety_vest','backpack','knife','pistol']
def main():
    train=yaml.safe_load((ROOT/'configs/stage3c_train.yaml').read_text())
    data=yaml.safe_load((ROOT/'configs/stage3c_data.yaml').read_text())
    if train['model']!='configs/yolo11n.yaml' or train['pretrained'] is not False or train['resume'] is not False or train['amp'] is not False or train['device']!=0 or train['seed']!=42 or train['batch']!=2 or train['imgsz']!=640:
        raise SystemExit('Stage 3C scratch contract changed')
    if data['nc']!=6 or [data['names'][i] for i in range(6)]!=NAMES:raise SystemExit('Stage 3C classes changed')
    if not (BASE/'STATUS.txt').read_text().startswith('EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED'):
        raise SystemExit('Experimental status missing')
    with (BASE/'ledger.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)!=9441:raise SystemExit('Incomplete candidate ledger')
    included=[r for r in rows if r['status']=='included']
    if not included:raise SystemExit('No included images')
    if any(r['human_review']!='No' or r['annotation_completeness']!='unverified_six_class' for r in included):
        raise SystemExit('Provisional review status changed')
    images_by_split={s:defaultdict(list) for s in ('train','test','val')}
    for s in images_by_split:
        for p in (BASE/'images'/s).iterdir():
            if p.is_file():images_by_split[s][p.stem].append(p)
    groups=defaultdict(set);ids=set();stats=defaultdict(Counter)
    for r in included:
        split=r['assigned_split'];gid=r['provisional_component_id'];stem=r['id']
        if split not in ('train','test','val') or not gid or stem in ids:raise SystemExit('Invalid split, group or duplicate ID')
        ids.add(stem);groups[gid].add(split)
        paths=images_by_split[split].get(stem,[])
        label=BASE/'labels'/split/(stem+'.txt')
        if len(paths)!=1 or not label.is_file():raise SystemExit('Missing image/label pair '+stem)
        with paths[0].open('rb') as f:digest=hashlib.sha256(f.read()).hexdigest()
        if digest!=r['image_sha256']:raise SystemExit('Image hash changed '+stem)
        present=set();n=0
        for line in label.read_text().splitlines():
            parts=line.split()
            if len(parts)!=5:raise SystemExit('Bad label '+stem)
            cid=int(parts[0]);x,y,w,h=map(float,parts[1:])
            if cid not in range(6) or not all(map(math.isfinite,(x,y,w,h))) or not (0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1):raise SystemExit('Bad box '+stem)
            present.add(cid);stats[split][(cid,'boxes')]+=1;n+=1
        if n==0:raise SystemExit('Empty unverified label '+stem)
        for cid in present:stats[split][(cid,'images')]+=1
    if any(len(s)!=1 for s in groups.values()):raise SystemExit('Provisional group crosses split')
    for split in ('train','test','val'):
        im=[p for paths in images_by_split[split].values() for p in paths]
        lab=list((BASE/'labels'/split).glob('*.txt'))
        expected=sum(r['assigned_split']==split for r in included)
        if len(im)!=expected or len(lab)!=expected:raise SystemExit('Extra or missing processed pair '+split)
        if any(stats[split][(cid,'images')]==0 for cid in range(6)):raise SystemExit('Class missing from split')
    print(json.dumps({'mechanically_ready':True,'status':'EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED',
                      'included':len(included),'groups':len(groups),
                      'split_images':dict(Counter(r['assigned_split'] for r in included)),
                      'warning':'Test independence and annotation completeness are not established'},indent=2))
if __name__=='__main__':main()
