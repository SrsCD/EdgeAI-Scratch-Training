"""Read-only audit of manual ZIPs; writes review artifacts, never accepted labels."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import imagehash
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/manual_imports"
OUT = ROOT / "reports/stage2d_manual"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "images for EL MAIN.zip": "knife_source",
    "Motorcycle Helmet.v1i.yolov8.zip": "motorcycle_helmet",
    "guns.zip": "people_with",
    "EL.zip": "robbery_headwear",
}
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLORS = ["#f44336", "#00bcd4", "#ff9800", "#4caf50", "#9c27b0", "#3f51b5"]


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def split_of(path):
    p = PurePosixPath(path).parts
    return p[0] if p and p[0] in {"train", "test", "valid", "val"} else "unknown"


def group_of(filename):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(filename).stem, flags=re.I)
    stem = re.sub(r"_jpg$|_png$|_jpeg$", "", stem, flags=re.I)
    m = re.match(r"(Robbery\d+)[_-]", stem, re.I)
    if m:
        return m.group(1).lower(), "recording_prefix"
    m = re.match(r"(Call-of-Duty)[_-]", stem, re.I)
    if m:
        return "call_of_duty", "game_title_prefix"
    m = re.match(r"(knifeDefenseTechnique)[_-]?", stem, re.I)
    if m:
        return m.group(1).lower(), "filename_series"
    if re.match(r"knife[_-]\d+$|BikesHelmets\d+$", stem, re.I):
        return stem.lower(), "numbered_collection_unresolved"
    m = re.match(r"(.+?)[_-]\d{2,}(?:[_-]|$)", stem)
    if m:
        return m.group(1).lower(), "filename_series"
    return stem.lower(), "single_or_unresolved"


def parse_labels(raw, nc):
    boxes, polygons, issues = [], [], []
    if raw is None:
        return boxes, polygons, ["missing_label"]
    for line_no, line in enumerate(raw.decode("utf-8-sig", "replace").splitlines(), 1):
        if not line.strip():
            continue
        words = line.split()
        try:
            cid = int(words[0])
            vals = [float(x) for x in words[1:]]
            if cid < 0 or cid >= nc or not all(math.isfinite(x) for x in vals):
                raise ValueError("class_or_nonfinite")
            if len(vals) == 4:
                x, y, w, h = vals
                if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0 and
                        x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6 and
                        y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6):
                    raise ValueError("invalid_box_coordinates")
                boxes.append((cid, (x-w/2, y-h/2, x+w/2, y+h/2)))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                pts = list(zip(vals[::2], vals[1::2]))
                if any(v < 0 or v > 1 for v in vals) or len(set(pts)) < 3:
                    raise ValueError("invalid_polygon_coordinates")
                polygons.append((cid, pts))
            else:
                raise ValueError("invalid_field_count")
        except Exception as e:
            issues.append(f"line_{line_no}:{e}")
    if not boxes and not polygons and not issues:
        issues.append("empty_label")
    return boxes, polygons, issues


def main():
    records, stats, metadata = [], [], {}
    previews = defaultdict(list)
    for zipname, source in SOURCES.items():
        archive = INPUT / zipname
        if not archive.exists():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as z:
            entries = {i.filename: i for i in z.infolist() if not i.is_dir()}
            yamls = [n for n in entries if PurePosixPath(n).name == "data.yaml"]
            if not yamls:
                raise ValueError(f"No data.yaml in {zipname}")
            yaml = z.read(yamls[0]).decode("utf-8-sig", "replace")
            names_match = re.search(r"^names:\s*\[(.*?)\]", yaml, re.M)
            if not names_match:
                raise ValueError(f"Unrecognized class names in {zipname}")
            names = [x.strip().strip("'\"") for x in names_match.group(1).split(",")]
            metadata[source] = {"archive": zipname, "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
                                "yaml": yaml, "names": names,
                                "metadata_files": {n: z.read(n).decode("utf-8-sig", "replace")
                                                   for n in entries if PurePosixPath(n).name.lower()
                                                   in {"readme.dataset.txt", "readme.roboflow.txt", "license.txt", "license.md"}},
                                "duplicate_zip_member_names": [n for n, c in Counter(i.filename for i in z.infolist()).items() if c > 1]}
            imgs = sorted(n for n in entries if PurePosixPath(n).suffix.lower() in IMAGE_EXT and "/images/" in n)
            labels = {n for n in entries if n.endswith(".txt") and "/labels/" in n}
            expected = {n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for n in imgs}
            c = Counter()
            class_images = Counter()
            class_boxes = Counter()
            for imgpath in imgs:
                labelpath = imgpath.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                raw_label = z.read(labelpath) if labelpath in labels else None
                boxes, polys, issues = parse_labels(raw_label, len(names))
                raw = z.read(imgpath)
                im = None
                try:
                    im = Image.open(BytesIO(raw))
                    im.load()
                    if im.width <= 0 or im.height <= 0:
                        raise ValueError("zero_dimensions")
                    phash = str(imagehash.phash(ImageOps.exif_transpose(im).convert("RGB")))
                    size = (im.width, im.height)
                except Exception as e:
                    issues.append("unreadable_image:" + str(e))
                    phash = ""
                    size = (0, 0)
                group, group_basis = group_of(PurePosixPath(imgpath).name)
                rec = {"source": source, "archive": zipname, "split": split_of(imgpath),
                       "image_path_in_zip": imgpath, "label_path_in_zip": labelpath if labelpath in labels else "",
                       "width": size[0], "height": size[1], "image_sha256": sha256(raw).hexdigest(),
                       "phash": phash, "source_group": f"{source}:{group}", "group_basis": group_basis,
                       "box_count": len(boxes), "polygon_count": len(polys),
                       "source_class_counts": json.dumps(dict(Counter(names[cid] for cid, _ in boxes+polys)), ensure_ascii=False),
                       "issues": ";".join(issues), "review_decision": "Pending",
                       "candidate_status": "quarantine_unverified"}
                records.append(rec)
                c["images"] += 1
                c["boxes"] += len(boxes)
                c["polygons"] += len(polys)
                c["missing_labels"] += "missing_label" in issues
                c["empty_labels"] += "empty_label" in issues
                c["invalid_images"] += any(i.startswith("unreadable_image") for i in issues)
                c["invalid_label_images"] += any(i.startswith("line_") for i in issues)
                for cid in {cid for cid, _ in boxes+polys}:
                    class_images[names[cid]] += 1
                for cid, _ in boxes+polys:
                    class_boxes[names[cid]] += 1
                # Retain a compact, reproducible stratified visual sample.
                if im is not None:
                    key = tuple(sorted({cid for cid, _ in boxes+polys}))
                    rank = int(sha256(imgpath.encode()).hexdigest()[:12], 16)
                    previews[source].append((rank, key, imgpath, boxes, polys))
            for split in ["train", "test", "valid", "val", "unknown"]:
                subset = [r for r in records if r["source"] == source and r["split"] == split]
                if subset:
                    stats.append({"source": source, "archive": zipname, "split": split,
                                  "images": len(subset), "paired_labels": sum(bool(r["label_path_in_zip"]) for r in subset),
                                  "missing_labels": sum("missing_label" in r["issues"] for r in subset),
                                  "empty_labels": sum("empty_label" in r["issues"] for r in subset),
                                  "invalid_image_count": sum("unreadable_image" in r["issues"] for r in subset),
                                  "invalid_label_image_count": sum("line_" in r["issues"] for r in subset),
                                  "valid_box_rows": sum(r["box_count"] for r in subset),
                                  "valid_polygon_rows": sum(r["polygon_count"] for r in subset),
                                  "orphan_labels_in_archive": len(labels-expected) if split == "train" else "",
                                  "source_class_positive_images": json.dumps(dict(Counter(k for r in subset for k in json.loads(r["source_class_counts"]))), ensure_ascii=False),
                                  "source_class_objects": json.dumps(dict(Counter({k:sum(json.loads(r["source_class_counts"]).get(k,0) for r in subset) for k in names})), ensure_ascii=False)})
            metadata[source]["orphan_label_paths"] = sorted(labels-expected)
            metadata[source]["class_positive_images"] = dict(class_images)
            metadata[source]["class_objects"] = dict(class_boxes)

    # Exact content matches and pHash distance <=4 are candidates; inspect visually.
    hashes = defaultdict(list)
    for r in records:
        hashes[r["image_sha256"]].append(r)
    exact = [v for v in hashes.values() if len(v) > 1]
    for i, cluster in enumerate(exact, 1):
        for r in cluster:
            r["exact_duplicate_cluster"] = i
    near_edges = []
    for i, a in enumerate(records):
        if not a["phash"]:
            continue
        h = int(a["phash"], 16)
        for b in records[i+1:]:
            if b["phash"] and a["image_sha256"] != b["image_sha256"]:
                d = (h ^ int(b["phash"], 16)).bit_count()
                if d <= 4:
                    near_edges.append({"source_a": a["source"], "image_a": a["image_path_in_zip"],
                                       "split_a": a["split"], "source_b": b["source"],
                                       "image_b": b["image_path_in_zip"], "split_b": b["split"],
                                       "phash_distance": d, "review_decision": "Pending"})
                    a["near_duplicate_candidate_count"] = a.get("near_duplicate_candidate_count", 0)+1
                    b["near_duplicate_candidate_count"] = b.get("near_duplicate_candidate_count", 0)+1
    groups = defaultdict(list)
    for r in records:
        groups[r["source_group"]].append(r)
    group_rows = [{"source_group": k, "basis": v[0]["group_basis"], "image_count": len(v),
                   "splits": ";".join(sorted({r["split"] for r in v})),
                   "cross_split": len({r["split"] for r in v}) > 1}
                  for k, v in groups.items() if len(v) > 1]

    for source, items in previews.items():
        # Samples from each source class and annotation type plus deterministic random remainder.
        chosen = []
        for cid in range(len(metadata[source]["names"])):
            chosen += sorted([x for x in items if cid in x[1]], key=lambda x:x[0])[:8]
        chosen += sorted([x for x in items if any(len(p[1]) >= 3 for p in x[4])], key=lambda x:x[0])[:8]
        chosen += sorted(items, key=lambda x:x[0])[:36]
        unique = list(dict((x[2], x) for x in chosen).values())[:60]
        with zipfile.ZipFile(INPUT / metadata[source]["archive"]) as z:
            for page in range(0, len(unique), 30):
                chunk = unique[page:page+30]
                sheet = Image.new("RGB", (1500, 1080), "white")
                draw = ImageDraw.Draw(sheet)
                for i, (_, _, name, boxes, polys) in enumerate(chunk):
                    im = Image.open(BytesIO(z.read(name))).convert("RGB")
                    im.thumbnail((242, 175))
                    x0, y0 = (i%6)*250+4, (i//6)*216+4
                    sheet.paste(im, (x0, y0))
                    for cid, xyxy in boxes:
                        draw.rectangle((x0+xyxy[0]*im.width, y0+xyxy[1]*im.height,
                                        x0+xyxy[2]*im.width, y0+xyxy[3]*im.height),
                                       outline=COLORS[cid % len(COLORS)], width=2)
                    for cid, pts in polys:
                        draw.line([(x0+x*im.width, y0+y*im.height) for x,y in pts+[pts[0]]],
                                  fill=COLORS[cid % len(COLORS)], width=2)
                    draw.text((x0,y0+177), name.split("/")[-1][:34], fill="black")
                    draw.text((x0,y0+192), "SOURCE ANNOTATIONS - UNVERIFIED", fill="#a00000")
                sheet.save(OUT / f"{source}_review_{page//30+1:02}.jpg", quality=90)

    fields = ["source","archive","split","image_path_in_zip","label_path_in_zip","width","height",
              "image_sha256","phash","source_group","group_basis","box_count","polygon_count",
              "source_class_counts","issues","exact_duplicate_cluster","near_duplicate_candidate_count",
              "candidate_status","review_decision"]
    for r in records:
        r.setdefault("exact_duplicate_cluster", "")
        r.setdefault("near_duplicate_candidate_count", 0)
    write_csv("per_image_audit_ledger.csv", records, fields)
    write_csv("source_statistics.csv", stats, list(stats[0]))
    write_csv("near_duplicate_candidates.csv", near_edges, list(near_edges[0]) if near_edges else ["source_a","image_a","split_a","source_b","image_b","split_b","phash_distance","review_decision"])
    write_csv("related_source_groups.csv", group_rows, list(group_rows[0]) if group_rows else ["source_group","basis","image_count","splits","cross_split"])
    queue = [{"source":r["source"],"image_path_in_zip":r["image_path_in_zip"],
              "source_group":r["source_group"],"source_class_counts":r["source_class_counts"],
              "mechanical_issues":r["issues"],"near_duplicate_candidate_count":r["near_duplicate_candidate_count"],
              "decision_Approve_Reject_Correct":"Pending","corrected_label_path":"",
              "six_class_completeness_checked":"No","taxonomy_checked":"No",
              "source_rights_checked":"No","reviewer":"","review_date":"","notes":""} for r in records]
    write_csv("human_review_queue.csv", queue, list(queue[0]))
    (OUT / "archive_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    summary = {"total_images":len(records), "source_images":dict(Counter(r["source"] for r in records)),
               "exact_duplicate_clusters":len(exact),"exact_duplicate_images":sum(len(v) for v in exact),
               "near_duplicate_edges":len(near_edges), "related_groups":len(group_rows),
               "cross_split_related_groups":sum(g["cross_split"] for g in group_rows),
               "cross_split_near_edges":sum(e["split_a"] != e["split_b"] for e in near_edges)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
