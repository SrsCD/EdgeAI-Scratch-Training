"""Select a diverse offline review pilot; copy source images, never approve them."""
from __future__ import annotations

import csv
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/interim/stage3_candidates_v0/selection_and_conversion_manifest.csv"
COMPONENTS = ROOT / "reports/stage3/provisional_component_members.csv"
OUT = ROOT / "data/interim/stage3b_review_v1"
ASSETS = OUT / "assets"
SHEETS = OUT / "contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(exist_ok=True)
SHEETS.mkdir(exist_ok=True)
QUOTAS = {"construction_site_safety":240,"construction_ppe_roboflow":180,
          "backpack_roboflow":160,"knife_source":150,"pistols_roboflow":170}
NAMES = ["person","helmet","safety_vest","backpack","knife","pistol"]
SEED = 42
COLORS = ["#29bf70","#d59b22","#ee731d","#31a7d7","#d83a9f","#9b58df"]
SOURCE_URLS = {
    "construction_site_safety":"https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30",
    "construction_ppe_roboflow":"https://universe.roboflow.com/gaos-workspace/construction-ppe-qofi4/dataset/1",
    "backpack_roboflow":"https://universe.roboflow.com/vrk2/backpack-y9y87/dataset/1",
    "knife_source":"https://universe.roboflow.com/tuan-tran/knifes-lfu4v/dataset/1",
    "pistols_roboflow":"https://universe.roboflow.com/joseph-nelson/pistols/dataset/1",
}
SOURCE_LICENSE = {s:("Public Domain claim" if s=="pistols_roboflow" else "CC BY 4.0 claim") for s in QUOTAS}
PILOT_SCREEN_OUT = {
    "export/images/armas (730)_jpg.rf.48fab75bda99526e9c41877de517d8cd.jpg":"toy_or_staged_miniature",
    "export/images/armas (1429)_jpg.rf.83bfc05ab9fad72f189690c5f7666241.jpg":"ambiguous_replica",
    "export/images/armas (1790)_jpg.rf.90c5969b29d8071f309a7aa4e06e35cd.jpg":"gun_depiction_on_phone",
    "export/images/armas (2831)_jpg.rf.5f0242c68e483ae76d6d99abcc5c5d91.jpg":"printed_gun_photo",
    "export/images/armas (745)_jpg.rf.132810e77748ede9f12d13ab926dba7b.jpg":"graphic_poster",
    "export/images/armas (989)_jpg.rf.78f8061f78eeb57618818c319ee5200b.jpg":"gun_symbol_on_sign",
    "export/images/armas (1740)_jpg.rf.e38e85eea500826071131451a44db301.jpg":"orange_tip_replica_candidate",
    "export/images/armas (292)_jpg.rf.0e7b9340b08fabfaf91d94e68e05cd82.jpg":"clipart_illustration",
    "export/images/armas (173)_jpg.rf.e57f25e25d90cdbcea12f494ed554d48.jpg":"silhouette_illustration",
    "export/images/armas (1354)_jpg.rf.c5afd33e584c940dc6686f22c1a92743.jpg":"cross_source_phash10_pair_unresolved_defer_from_pilot",
}


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    if any((OUT/"reviews").glob("P*.json")):
        raise SystemExit("Pilot reviews exist; refusing to regenerate queue IDs and assets")
    rows=read_csv(SELECTION)
    components=read_csv(COMPONENTS)
    sizes={(r["source"],r["image_path_in_zip"]):int(r["component_size"]) for r in components}
    rng=random.Random(SEED)
    selected=[]
    global_phashes=[]
    for source,quota in QUOTAS.items():
        pool=[r for r in rows if r["source"]==source and r["selection_status"]=="mapped_candidate_quarantine"
              and sizes[(source,r["image_path_in_zip"]) ]==1 and r["image_path_in_zip"] not in PILOT_SCREEN_OUT]
        # Seeded tie-breaker plus class coverage. Avoid overly similar views
        # within the selected pilot without deleting anything from the source.
        rng.shuffle(pool)
        chosen=[]
        selected_phashes=[]
        covered=Counter()
        remaining=pool[:]
        while len(chosen)<quota and remaining:
            def score(r):
                c=json.loads(r["provisional_target_counts"])
                if source=="construction_site_safety":
                    return 5*(c["safety_vest"]>0)/(1+covered["safety_vest"]) + 4*(c["helmet"]>0)/(1+covered["helmet"]) + 3*(c["person"]>0)/(1+covered["person"])
                if source=="construction_ppe_roboflow":
                    return 4*(c["safety_vest"]>0)/(1+covered["safety_vest"]) + 4*(c["helmet"]>0)/(1+covered["helmet"])
                return 1
            remaining.sort(key=score,reverse=True)
            picked=None
            for i,r in enumerate(remaining):
                ph=int(r["phash"],16) if r["phash"] else None
                if ph is None or (all((ph^h).bit_count()>10 for h in selected_phashes) and
                                  all((ph^h).bit_count()>10 for h in global_phashes)):
                    picked=remaining.pop(i)
                    if ph is not None:
                        selected_phashes.append(ph)
                        global_phashes.append(ph)
                    break
            if picked is None: break
            chosen.append(picked)
            for name,count in json.loads(picked["provisional_target_counts"]).items():
                covered[name]+=count>0
        if len(chosen)<quota:
            print(f"{source}: selected {len(chosen)} of requested {quota}; similarity filter limited pool")
        selected+=chosen
    zips={}
    queue=[]
    try:
        for n,r in enumerate(selected,1):
            archive=ROOT/"data/raw/manual_imports"/r["archive"]
            if archive not in zips: zips[archive]=zipfile.ZipFile(archive)
            data=zips[archive].read(r["image_path_in_zip"])
            if sha256(data).hexdigest()!=r["image_sha256"]:
                raise ValueError(f"Source image hash mismatch: {r['image_path_in_zip']}")
            image_id=f"P{n:04d}"
            ext=Path(r["image_path_in_zip"]).suffix.lower()
            asset=ASSETS/f"{image_id}{ext}"
            asset.write_bytes(data)
            label=ROOT/r["provisional_label_path"]
            boxes=[]
            for line in label.read_text(encoding="utf-8").splitlines():
                cid,*vals=line.split()
                boxes.append({"class_id":int(cid),"cx":float(vals[0]),"cy":float(vals[1]),
                              "w":float(vals[2]),"h":float(vals[3])})
            with Image.open(BytesIO(data)) as im:
                width,height=im.size
            queue.append({"id":image_id,"source":r["source"],"archive":r["archive"],
                          "image_path_in_zip":r["image_path_in_zip"],"image_sha256":r["image_sha256"],
                          "phash":r["phash"],
                          "asset_path":asset.relative_to(OUT).as_posix(),"width":width,"height":height,
                          "source_split":r["source_split"],"source_group_hint":r["source_group_hint"],
                          "provisional_label_path":r["provisional_label_path"],
                          "provisional_boxes":boxes,"review_flags":r["review_flags"],
                          "source_url":SOURCE_URLS[r["source"]],"source_license_claim":SOURCE_LICENSE[r["source"]],
                          "decision":"Pending"})
    finally:
        for z in zips.values():z.close()
    (OUT/"queue.json").write_text(json.dumps({"version":"stage3b_review_v1","seed":SEED,"items":queue},indent=2)+"\n",encoding="utf-8")
    with (OUT/"pre_screen_exclusions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["source","image_path_in_zip","reason","status"]);w.writeheader()
        for path,reason in PILOT_SCREEN_OUT.items():
            w.writerow({"source":"pistols_roboflow","image_path_in_zip":path,"reason":reason,
                        "status":"deferred_near_pair_review" if "phash" in reason else
                                 "excluded_from_pilot_by_agent_visual_screen_not_human_approval"})
    fields=["id","source","archive","image_path_in_zip","image_sha256","phash","asset_path","width","height",
            "source_split","source_group_hint","provisional_label_path","review_flags","source_url","source_license_claim","decision"]
    with (OUT/"queue.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(queue)
    # Every image has an offline visual batch index. The boxes are source
    # proposals only; contact sheets never represent reviewed ground truth.
    for source in QUOTAS:
        subset=[r for r in queue if r["source"]==source]
        for offset in range(0,len(subset),30):
            sheet=Image.new("RGB",(1500,1080),"white")
            draw=ImageDraw.Draw(sheet)
            for pos,r in enumerate(subset[offset:offset+30]):
                with Image.open(OUT/r["asset_path"]) as orig:
                    im=orig.convert("RGB")
                im.thumbnail((242,175))
                x0,y0=(pos%6)*250+4,(pos//6)*216+4
                sheet.paste(im,(x0,y0))
                for box in r["provisional_boxes"]:
                    cid=box["class_id"]
                    x1=x0+(box["cx"]-box["w"]/2)*im.width
                    y1=y0+(box["cy"]-box["h"]/2)*im.height
                    x2=x0+(box["cx"]+box["w"]/2)*im.width
                    y2=y0+(box["cy"]+box["h"]/2)*im.height
                    draw.rectangle((x1,y1,x2,y2),outline=COLORS[cid],width=2)
                    draw.text((x1,max(y0,y1-10)),NAMES[cid],fill=COLORS[cid])
                draw.text((x0,y0+178),r["id"]+" "+Path(r["image_path_in_zip"]).name[:23],fill="black")
                draw.text((x0,y0+193),"UNREVIEWED SOURCE LABELS",fill="#a00000")
            sheet.save(SHEETS/f"{source}_{offset//30+1:02d}.jpg",quality=88)
    print(json.dumps({"selected":len(queue),"by_source":dict(Counter(r["source"] for r in queue)),
                      "review_decisions":"all pending","contact_sheets":len(list(SHEETS.glob('*.jpg')))},indent=2))


if __name__=="__main__":main()
