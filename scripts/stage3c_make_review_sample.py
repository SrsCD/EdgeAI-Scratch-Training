"""Exactly ten deliberate human-review examples; no approval is inferred."""
import csv,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/interim/stage3b_review_v1'
OUT=ROOT/'data/experimental/stage3c_v1/review_sample'
q=json.loads((SOURCE/'queue.json').read_text(encoding='utf-8'))['items']
if any((OUT/'reviews').glob('P*.json')):raise SystemExit('Review decisions exist; refusing sample overwrite')
def take(source,predicate,n):
    rows=[r for r in q if r['source']==source and predicate(r)]
    if len(rows)<n:raise ValueError(source)
    return rows[:n]
present=lambda r,c:any(b['class_id']==c for b in r['provisional_boxes'])
selected=(take('construction_site_safety',lambda r:all(present(r,c) for c in (0,1,2)),2)+
          take('construction_ppe_roboflow',lambda r:'polygon_enclosure' in r['review_flags'],2)+
          take('backpack_roboflow',lambda r:True,2)+
          take('knife_source',lambda r:True,2)+
          take('pistols_roboflow',lambda r:True,2))
reasons=['person_helmet_vest_multiclass','person_helmet_vest_multiclass',
         'polygon_to_box_and_possible_missing_person','polygon_to_box_and_possible_missing_person',
         'backpack_and_possible_missing_person','backpack_and_possible_missing_person',
         'knife_and_possible_missing_person','knife_and_possible_missing_person',
         'real_pistol_vs_illustration_and_missing_person','real_pistol_vs_illustration_and_missing_person']
(OUT/'assets').mkdir(parents=True,exist_ok=True);(OUT/'reviews').mkdir(exist_ok=True)
for r,reason in zip(selected,reasons):
    shutil.copyfile(SOURCE/r['asset_path'],OUT/r['asset_path'])
    r['sample_reason']=reason
    r['decision']='Pending'
(OUT/'queue.json').write_text(json.dumps({'version':'stage3c_ten_image_review','items':selected},indent=2)+'\n',encoding='utf-8')
with (OUT/'queue.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['id','source','image_path_in_zip','sample_reason','review_flags','decision']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(selected)
sheet=Image.new('RGB',(1500,460),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(selected):
    with Image.open(OUT/r['asset_path']) as im:thumb=im.convert('RGB')
    thumb.thumbnail((290,190));x=(i%5)*300;y=(i//5)*230
    sheet.paste(thumb,(x,y))
    for b in r['provisional_boxes']:
        a=(b['cx']-b['w']/2)*thumb.width+x;c=(b['cy']-b['h']/2)*thumb.height+y
        d=(b['cx']+b['w']/2)*thumb.width+x;e=(b['cy']+b['h']/2)*thumb.height+y
        draw.rectangle((a,c,d,e),outline='red',width=2)
    draw.text((x,y+194),r['id']+' '+r['source'][:25],fill='black')
    draw.text((x,y+209),'UNREVIEWED',fill='red')
sheet.save(OUT/'contact_sheet.jpg',quality=90)
print('Selected exactly',len(selected),'pending review images:',','.join(r['id'] for r in selected))
