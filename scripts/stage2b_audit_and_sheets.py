"""Mechanical cross-source audit and annotated pilot contact sheets (no model)."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/stage2b_v1"
OUT = ROOT / "reports/stage2b_v1"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
MIDS = {"/m/01g317": 0, "/m/0zvk5": 1, "/m/01940j": 3, "/m/04ctx": 4, "/m/0gxl3": 5}
COLORS = ["#20da5d", "#edc846", "#ff8a26", "#54c5ff", "#f04c8b", "#af63ff"]


def main():
    construction = json.loads((RAW / "construction_pilot_manifest.json").read_text())
    oi = json.loads((RAW / "open_images_pilot_manifest.json").read_text())
    records = []
    for r in construction["records"]:
        if r["status"] == "rejected":
            continue
        rec = {"source": "construction_ppe", "id": r["source_image_id"],
               "path": r["raw_image_path"], "sha256": r["sha256"], "phash": r["phash"],
               "status": "quarantined", "reason": r["reason"], "group_id": None,
               "source_box_counts": r["mapped_source_box_counts"],
               "boxes": []}
        label = ROOT / r["raw_image_path"]
        for line in label.with_suffix(".txt").read_text().splitlines():
            src, x, y, w, h = line.split()
            target = {0: 1, 2: 2, 5: 0}.get(int(src))
            if target is not None:
                rec["boxes"].append([target, *map(float, (x, y, w, h))])
        records.append(rec)
    for r in oi["records"]:
        if not r.get("sha256"):
            continue
        p = ROOT / r["raw_path"]
        try:
            with Image.open(p) as im:
                im.load()
                width, height = im.size
                phash = str(imagehash.phash(im.convert("RGB")))
            boxes = []
            non_target_source_boxes = 0
            depiction_source_boxes = 0
            for src in r["source_boxes"]:
                xmin, xmax, ymin, ymax = (float(src[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
                if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                    raise ValueError("invalid source box extent")
                depiction_source_boxes += src.get("IsDepiction") == "1"
                target = MIDS.get(src["LabelName"])
                if target is None:
                    non_target_source_boxes += 1
                    continue
                boxes.append([target, (xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin])
            if not boxes:
                raise ValueError("no target source boxes")
            counts = Counter(NAMES[b[0]] for b in boxes)
            rec = {"source": "open_images_v7", "id": r["image_id"], "path": r["raw_path"],
                   "sha256": r["sha256"], "phash": phash, "width": width, "height": height,
                   "status": "quarantined", "reason": "six-class visual completeness and taxonomy pending",
                   "group_id": None, "source_box_counts": dict(counts), "boxes": boxes,
                   "all_source_boxes_validated": True,
                   "non_target_source_boxes": non_target_source_boxes,
                   "depiction_source_boxes": depiction_source_boxes,
                   "license_url": r["metadata"]["License"],
                   "landing_url": r["metadata"]["OriginalLandingURL"],
                   "author": r["metadata"]["Author"], "title": r["metadata"]["Title"]}
            records.append(rec)
        except Exception as error:
            r["mechanical_error"] = str(error)
    hashes = defaultdict(list)
    for r in records:
        hashes[r["sha256"]].append(f"{r['source']}:{r['id']}")
    exact = [v for v in hashes.values() if len(v) > 1]
    near = []
    for i, a in enumerate(records):
        h1 = imagehash.hex_to_hash(a["phash"])
        for b in records[i+1:]:
            d = h1-imagehash.hex_to_hash(b["phash"])
            if d <= 4 and a["sha256"] != b["sha256"]:
                near.append({"a": f"{a['source']}:{a['id']}", "b": f"{b['source']}:{b['id']}",
                             "phash_distance": d, "review_status": "pending_human"})
    source_images = Counter()
    source_boxes = Counter()
    for r in records:
        for name, count in r["source_box_counts"].items():
            source_images[name] += 1
            source_boxes[name] += count
    summary = {"candidate_images_downloaded": len(records),
               "by_source": dict(Counter(r["source"] for r in records)),
               "accepted": 0, "rejected_after_download": 0, "quarantined": len(records),
               "verified_positive_images": {x: 0 for x in NAMES},
               "verified_boxes": {x: 0 for x in NAMES},
               "provisional_source_positive_images": dict(source_images),
               "provisional_source_boxes": dict(source_boxes),
               "independent_source_groups_verified": 0,
               "exact_duplicate_groups": exact,
               "near_duplicate_review_candidates": near,
               "near_duplicate_threshold": "64-bit pHash Hamming distance <=4; candidate only",
               "records": records}
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Contact sheets use only the original source annotations and identify their
    # unresolved status. They are review aids, not accepted labels.
    groups = {"construction_ppe": [r for r in records if r["source"] == "construction_ppe"],
              "open_images_v7": [r for r in records if r["source"] == "open_images_v7"]}
    for source, group in groups.items():
        for start in range(0, len(group), 30):
            chunk = group[start:start+30]
            sheet = Image.new("RGB", (6*240, 5*216), "#f4f4f4")
            canvas = ImageDraw.Draw(sheet)
            for n, rec in enumerate(chunk):
                image = Image.open(ROOT / rec["path"]).convert("RGB")
                image.thumbnail((232, 175))
                x0 = n%6*240+4
                y0 = n//6*216+4
                sheet.paste(image, (x0, y0))
                for cls, x, y, w, h in rec["boxes"]:
                    box = [(x-w/2)*image.width+x0, (y-h/2)*image.height+y0,
                           (x+w/2)*image.width+x0, (y+h/2)*image.height+y0]
                    canvas.rectangle(box, outline=COLORS[cls], width=2)
                    canvas.text((box[0]+2, box[1]+2), NAMES[cls], fill=COLORS[cls])
                canvas.text((x0, y0+178), rec["id"][:26], fill="black")
                canvas.text((x0, y0+192), "QUARANTINED - source boxes", fill="#b01919")
            path = OUT / f"{source}_contact_{start//30+1:02d}.jpg"
            sheet.save(path, quality=88)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
