"""Make reviewable six-class candidate labels; never promote them to training data."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/stage2e_unified_v0/unified_candidate_manifest_v0.csv"
OUT = ROOT / "data/interim/stage3_candidates_v0"
LABELS = OUT / "provisional_labels"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
TARGET = {
    "construction_site_safety": {2: 1, 8: 0, 11: 2},
    "construction_ppe_roboflow": {0: 1, 1: 1, 2: 2},
    "backpack_roboflow": {0: 3},
    "knife_source": {0: 4},
    "pistols_roboflow": {0: 5},
}
SOURCE_COMPLETENESS_FLAGS = {
    "construction_site_safety": "backpack_knife_pistol_not_in_source_taxonomy",
    "construction_ppe_roboflow": "person_backpack_knife_pistol_not_in_source_taxonomy",
    "backpack_roboflow": "person_helmet_vest_knife_pistol_not_in_source_taxonomy",
    "knife_source": "person_helmet_vest_backpack_pistol_not_in_source_taxonomy",
    "pistols_roboflow": "person_helmet_vest_backpack_knife_not_in_source_taxonomy",
}
EXCLUDE_SOURCES = {
    "motorcycle_helmet": "motorcycle_or_bicycle_helmet_taxonomy_mismatch",
    "people_with": "video_game_hud_and_avatars_not_real_world_targets",
    "robbery_headwear": "unrelated_headwear_taxonomy_and_unreviewed_person_labels",
}
VISUALLY_CONFIRMED_NONPHOTO = {
    "export/images/armas (1814)_jpg.rf.c03a8cd9399b5aec76f5a0e21e3d0dc2.jpg",
    "export/images/armas (2952)_jpg.rf.38b6d635c18d87a081fac6fe439625b2.jpg",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parse_mapped(raw, mapping, image_id):
    lines, polygons, source_count, omitted_count = [], 0, 0, 0
    for line_number, line in enumerate(raw.decode("utf-8-sig", "replace").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        vals = [float(x) for x in parts[1:]]
        if not all(math.isfinite(x) for x in vals):
            raise ValueError(f"Nonfinite annotation {image_id}:{line_number}")
        source_count += 1
        if cid not in mapping:
            omitted_count += 1
            continue
        if len(vals) == 4:
            x, y, w, h = vals
            x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
            origin = "source_box"
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            polygons += 1
            origin = "polygon_enclosure"
        else:
            raise ValueError(f"Invalid annotation format {image_id}:{line_number}")
        if not (-1e-6 <= x1 < x2 <= 1+1e-6 and -1e-6 <= y1 < y2 <= 1+1e-6):
            raise ValueError(f"Invalid mapped bounds {image_id}:{line_number}")
        x1, y1, x2, y2 = max(0.0,x1), max(0.0,y1), min(1.0,x2), min(1.0,y2)
        cx, cy, bw, bh = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        lines.append((mapping[cid], cx, cy, bw, bh, cid, line_number, origin))
    return lines, polygons, source_count, omitted_count


def main():
    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        original = list(csv.DictReader(f))
    inventory, candidate_labels, conversion = [], [], []
    zips = {}
    try:
        for rec in original:
            source = rec["source"]
            row = {"source": source, "archive": rec["archive"],
                   "image_path_in_zip": rec["image_path_in_zip"], "image_sha256": rec["sha256"],
                   "phash": rec["phash"],
                   "source_split": rec["source_split"], "source_group_hint": rec["source_group_hint"],
                   "source_class_id_counts": rec["source_class_id_counts"],
                   "source_box_count": rec["source_box_count"], "source_polygon_count": rec["source_polygon_count"],
                   "provisional_label_path": "", "provisional_target_counts": "",
                   "unmapped_source_rows": "", "converted_polygon_rows": "",
                   "selection_status": "", "selection_reason": "", "annotation_completeness": "unreviewed",
                   "review_flags": "",
                   "rights_status": "unverified", "duplicate_group_status": "unresolved",
                   "accepted": "No", "final_group_id": "", "assigned_split": ""}
            if source in EXCLUDE_SOURCES:
                row["selection_status"] = "exclude_source"
                row["selection_reason"] = EXCLUDE_SOURCES[source]
                inventory.append(row)
                continue
            if source == "pistols_roboflow" and rec["image_path_in_zip"] in VISUALLY_CONFIRMED_NONPHOTO:
                row["selection_status"] = "exclude_image"
                row["selection_reason"] = "visually_confirmed_cartoon_in_stage2e_review_sheet"
                inventory.append(row)
                continue
            if rec["mechanical_issues"] and rec["mechanical_issues"] != "empty_label":
                row["selection_status"] = "exclude_mechanical"
                row["selection_reason"] = rec["mechanical_issues"]
                inventory.append(row)
                continue
            archive = ROOT / rec["archive_relative_path"]
            if archive not in zips:
                zips[archive] = zipfile.ZipFile(archive)
            z = zips[archive]
            raw = z.read(rec["label_path_in_zip"])
            mapped, poly_count, source_rows, unmapped = parse_mapped(raw, TARGET[source], rec["image_path_in_zip"])
            if source_rows != int(rec["source_box_count"]) + int(rec["source_polygon_count"]):
                raise ValueError(f"Source row count differs from Stage 2E manifest: {rec['image_path_in_zip']}")
            counts = Counter(NAMES[x[0]] for x in mapped)
            row["provisional_target_counts"] = json.dumps({n: counts.get(n, 0) for n in NAMES}, sort_keys=True)
            row["unmapped_source_rows"] = unmapped
            row["converted_polygon_rows"] = poly_count
            flags = [SOURCE_COMPLETENESS_FLAGS[source]]
            if not mapped:
                flags.append("no_mapped_target_box_not_verified_negative")
            if poly_count:
                flags.append("polygon_enclosure_requires_object_review")
            if int(rec["near_duplicate_candidates"]):
                flags.append("unresolved_near_duplicate_candidates")
            if source == "pistols_roboflow":
                flags.append("real_handgun_vs_cartoon_toy_or_other_gun_unreviewed")
            row["review_flags"] = ";".join(flags)
            if mapped:
                dest = LABELS / source / f"{rec['sha256']}.txt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("".join(f"{cid} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n" for cid,x,y,w,h,*_ in mapped), encoding="utf-8")
                row["provisional_label_path"] = dest.relative_to(ROOT).as_posix()
                row["selection_status"] = "mapped_candidate_quarantine"
                row["selection_reason"] = "source_mapped_labels_only; six_class_completeness_rights_and_groups_unreviewed"
                candidate_labels.append(row)
            else:
                row["selection_status"] = "no_mapped_target_quarantine"
                row["selection_reason"] = "no_qualifying_source_label; unannotated_target_objects_not_ruled_out"
            for cid, cx, cy, w, h, src_id, line_no, origin in mapped:
                if origin == "polygon_enclosure":
                    conversion.append({"source":source,"image_path_in_zip":rec["image_path_in_zip"],
                                       "source_class_id":src_id,"source_line":line_no,"target_class_id":cid,
                                       "box_cx":f"{cx:.10f}","box_cy":f"{cy:.10f}","box_w":f"{w:.10f}",
                                       "box_h":f"{h:.10f}","status":"provisional_requires_visual_review"})
            inventory.append(row)
    finally:
        for z in zips.values():
            z.close()
    write_csv(OUT / "selection_and_conversion_manifest.csv", inventory)
    write_csv(OUT / "polygon_to_box_log.csv", conversion or [{"source":"","image_path_in_zip":"","source_class_id":"","source_line":"","target_class_id":"","box_cx":"","box_cy":"","box_w":"","box_h":"","status":""}])
    class_rows = []
    for source in ["construction_site_safety","construction_ppe_roboflow","backpack_roboflow","knife_source","pistols_roboflow"]:
        subset = [r for r in inventory if r["source"] == source and r["selection_status"] in
                  {"mapped_candidate_quarantine","no_mapped_target_quarantine"}]
        counts = [json.loads(r["provisional_target_counts"]) for r in subset]
        for name in NAMES:
            class_rows.append({"source":source,"target_class":name,
                               "candidate_positive_images":sum(c[name]>0 for c in counts),
                               "candidate_boxes":sum(c[name] for c in counts),
                               "verified_positive_images":0,"verified_boxes":0})
    write_csv(OUT / "candidate_class_statistics.csv", class_rows)
    summary = {"total_source_images": len(inventory), "selection_status": dict(Counter(r["selection_status"] for r in inventory)),
               "provisional_label_files":len(candidate_labels),"provisional_polygon_boxes":len(conversion),
               "accepted_images":0,"final_splits_created":False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
