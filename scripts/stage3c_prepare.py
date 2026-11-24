"""Materialize a separate, explicitly provisional six-class experiment."""
import csv, hashlib, json, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from stage3b_prepare_review import PILOT_SCREEN_OUT

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'data/interim/stage3_candidates_v0/selection_and_conversion_manifest.csv'
COMP=ROOT/'reports/stage3/provisional_component_members.csv'
OUT=ROOT/'data/experimental/stage3c_v1'
NAMES=['person','helmet','safety_vest','backpack','knife','pistol']
SOURCES={'construction_site_safety','construction_ppe_roboflow','backpack_roboflow','knife_source','pistols_roboflow'}

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def valid_label(path):
    lines=path.read_text(encoding='utf-8').splitlines()
    if not lines:return None
    boxes=[]
    for line in lines:
        parts=line.split()
        if len(parts)!=5:return None
        try:cid=int(parts[0]);x,y,w,h=map(float,parts[1:])
        except ValueError:return None
        if cid not in range(6) or not (0<=x-w/2<x+w/2<=1 and 0<=y-h/2<y+h/2<=1):return None
        boxes.append((cid,x,y,w,h))
    return boxes

def main():
    if (OUT/'ledger.csv').exists():raise SystemExit('Stage 3C dataset already materialized; refusing overwrite')
    rows=read(MAN);components=read(COMP)
    group={(r['source'],r['image_path_in_zip']):r['provisional_component_id'] for r in components}
    if len(group)!=8007:raise SystemExit('Priority component manifest unexpected size')
    ledger=[];accepted=[];archives={}
    try:
        for r in rows:
            key=(r['source'],r['image_path_in_zip'])
            reason='';boxes=None
            if r['source'] not in SOURCES:reason='incompatible_or_excluded_source'
            elif r['selection_status']!='mapped_candidate_quarantine':reason=r['selection_status']+';'+r['selection_reason']
            elif r['image_path_in_zip'] in PILOT_SCREEN_OUT:reason='stage3b_known_disqualifying_or_unresolved:'+PILOT_SCREEN_OUT[r['image_path_in_zip']]
            else:
                label=ROOT/r['provisional_label_path']
                boxes=valid_label(label) if label.is_file() else None
                if boxes is None:reason='missing_empty_or_invalid_provisional_label'
            if not reason:
                archive=ROOT/'data/raw/manual_imports'/r['archive']
                if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
                try:
                    data=archives[archive].read(r['image_path_in_zip'])
                    if hashlib.sha256(data).hexdigest()!=r['image_sha256']:raise ValueError('image_hash_mismatch')
                    from io import BytesIO
                    with Image.open(BytesIO(data)) as im:
                        im.verify()
                except Exception as exc:reason='invalid_image_or_hash:'+type(exc).__name__
            status='included' if not reason else 'excluded_or_quarantined'
            record={'id':r['image_sha256'][:20], 'source':r['source'],'archive':r['archive'],
                    'image_path_in_zip':r['image_path_in_zip'],'image_sha256':r['image_sha256'],
                    'provisional_component_id':group.get(key,''), 'source_group_hint':r['source_group_hint'],
                    'original_source_split':r['source_split'],'status':status,'reason':reason,
                    'annotation_warning':r['review_flags'],'rights_status':r['rights_status'],
                    'annotation_completeness':'unverified_six_class','human_review':'No',
                    'polygon_box_conversions':r['converted_polygon_rows'],'assigned_split':''}
            ledger.append(record)
            if not reason:accepted.append((record,boxes))
    finally:
        for a in archives.values():a.close()
    if len(accepted)<100:raise SystemExit('Insufficient mechanically valid candidates')
    # Conservative provisional components stay intact. This cannot prove semantic independence.
    counts=defaultdict(Counter)
    for rec,boxes in accepted:
        gid=rec['provisional_component_id']
        counts[gid]['images']+=1
        for cid in {b[0] for b in boxes}:counts[gid][NAMES[cid]]+=1
    assignments={}
    for gid in sorted(counts):
        digest=int(hashlib.sha256(('stage3c_seed42:'+gid).encode()).hexdigest()[:16],16)/16**16
        assignments[gid]='train' if digest<.7 else 'test' if digest<.9 else 'val'
    for rec,_ in accepted:rec['assigned_split']=assignments[rec['provisional_component_id']]
    stats=[]
    for cid,name in enumerate(NAMES):
        entry={'class_id':cid,'class_name':name}
        for split in ('train','test','val'):
            subset=[boxes for rec,boxes in accepted if rec['assigned_split']==split]
            entry[split+'_positive_images']=sum(any(b[0]==cid for b in boxes) for boxes in subset)
            entry[split+'_boxes']=sum(b[0]==cid for boxes in subset for b in boxes)
        stats.append(entry)
    if any(row[s+'_positive_images']==0 for row in stats for s in ('train','test','val')):
        raise SystemExit('Provisional group split lacks a positive class in a split')
    archives={}
    try:
        for rec,boxes in accepted:
            split=rec['assigned_split'];stem=rec['id']
            image=OUT/'images'/split/(stem+Path(rec['image_path_in_zip']).suffix.lower())
            label=OUT/'labels'/split/(stem+'.txt')
            image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            archive=ROOT/'data/raw/manual_imports'/rec['archive']
            if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
            image.write_bytes(archives[archive].read(rec['image_path_in_zip']))
            label.write_text(''.join(f'{cid} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n' for cid,x,y,w,h in boxes),encoding='utf-8')
    finally:
        for a in archives.values():a.close()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    with (OUT/'split_stats.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    (OUT/'STATUS.txt').write_text('EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED\nProvisional annotations and source-group components; test is not proven independent.\n',encoding='utf-8')
    print(json.dumps({'included':len(accepted),'excluded':len(ledger)-len(accepted),
                      'split_images':dict(Counter(r['assigned_split'] for r,_ in accepted)),
                      'stats':stats},indent=2))

if __name__=='__main__':main()
