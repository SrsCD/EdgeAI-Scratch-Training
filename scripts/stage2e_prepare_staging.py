"""Assemble a versioned candidate manifest; promote no unreviewed pixels."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/stage2d_manual"
NEW = ROOT / "reports/stage2e_unified"
STAGE = ROOT / "data/interim/stage2e_unified_v0"
STAGE.mkdir(parents=True, exist_ok=True)
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
MAPPING = {
    ("construction_site_safety", "Hardhat"): (1, "conditional_hardhat_object_review"),
    ("construction_site_safety", "Person"): (0, "conditional_full_person_review"),
    ("construction_site_safety", "Safety Vest"): (2, "conditional_vest_definition_review"),
    ("construction_ppe_roboflow", "Helmet"): (1, "conditional_industrial_helmet_review_polygon_conversion"),
    ("construction_ppe_roboflow", "hardhat"): (1, "conditional_hardhat_object_review"),
    ("construction_ppe_roboflow", "vest"): (2, "conditional_vest_definition_review"),
    ("backpack_roboflow", "backpack"): (3, "conditional_backpack_object_review"),
    ("pistols_roboflow", "pistol"): (5, "conditional_real_handgun_review"),
}
REVIEW_ACTION = {
    "construction_site_safety": "Inspect all six target classes; check hardhat, vest and person completeness; inspect empty labels and video/scene groups",
    "construction_ppe_roboflow": "Inspect all six classes; add person boxes; verify helmets and vests; review polygon-to-box conversion and scene groups",
    "backpack_roboflow": "Verify actual backpack, all co-occurring people and target objects; inspect product-image rights and same-scene reuse",
    "pistols_roboflow": "Exclude cartoons, toys, unrelated guns and ambiguous cases; add people; verify underlying image rights and pistol boxes",
    "knife_source": "Verify knife scope and all visible people; resolve source rights and near-duplicate groups",
    "motorcycle_helmet": "Do not map motorcycle helmets; full six-class reannotation only if retained",
    "people_with": "Reject game imagery for real-world person/pistol; retain source audit reference",
    "robbery_headwear": "Reference only until full reannotation, source-video grouping and rights review",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    old_meta = json.loads((OLD / "archive_metadata.json").read_text(encoding="utf-8"))
    new_meta = json.loads((NEW / "archive_metadata.json").read_text(encoding="utf-8"))
    old = read_csv(OLD / "per_image_audit_ledger.csv")
    new = read_csv(NEW / "per_image_audit_ledger.csv")
    old_map = read_csv(OLD / "class_mapping_proposal.csv")
    maps = []
    for r in old_map:
        maps.append({"source":r["source"],"archive":old_meta[r["source"]]["archive"],
                     "source_class_id":r["source_class_id"],"source_class_name":r["source_class_name"],
                     "proposed_target_id":r["proposed_target_id"],"proposed_target_name":r["proposed_target_name"],
                     "status":r["disposition"],"reason":r["reason"],"required_action":r["required_action"]})
    for source, meta in new_meta.items():
        for cid, name in enumerate(meta["names"]):
            mapped = MAPPING.get((source,name))
            maps.append({"source":source,"archive":meta["archive"],"source_class_id":cid,
                         "source_class_name":name,"proposed_target_id":mapped[0] if mapped else "",
                         "proposed_target_name":NAMES[mapped[0]] if mapped else "",
                         "status":mapped[1] if mapped else "no_target_mapping",
                         "reason":"Visual and six-class review pending" if mapped else "Source category outside fixed six-class taxonomy",
                         "required_action":REVIEW_ACTION[source]})
    write_csv(NEW / "unified_class_mapping_proposal.csv", maps, list(maps[0]))

    source_class_stats = []
    for source, meta in {**old_meta, **new_meta}.items():
        typed = Counter()
        with zipfile.ZipFile(ROOT / "data/raw/manual_imports" / meta["archive"]) as z:
            for path in z.namelist():
                if "/labels/" not in path or not path.endswith(".txt"):
                    continue
                for line in z.read(path).decode("utf-8-sig", "replace").splitlines():
                    words = line.split()
                    if words:
                        typed[(int(words[0]), "box" if len(words)==5 else "polygon")] += 1
        for cid, name in enumerate(meta["names"]):
            source_class_stats.append({"source":source,"archive":meta["archive"],
                                       "source_class_id":cid,"source_class_name":name,
                                       "positive_images":meta["class_positive_images"].get(name,0),
                                       "box_rows":typed[(cid,"box")],"polygon_rows":typed[(cid,"polygon")],
                                       "total_annotation_rows":meta["class_objects"].get(name,0),
                                       "verified_target_positive_images":0})
    write_csv(NEW / "source_class_statistics.csv", source_class_stats, list(source_class_stats[0]))

    manifest, queue = [], []
    for r in old + new:
        source = r["source"]
        source_names = (new_meta.get(source) or old_meta[source])["names"]
        named_counts = json.loads(r["source_class_counts"])
        id_counts = {str(cid):named_counts[name] for cid,name in enumerate(source_names) if name in named_counts}
        status = "source_reject_real_world" if source == "people_with" else "quarantine_pending_review"
        row = {"dataset_version":"stage2e_unified_v0", "source":source, "archive":r["archive"],
               "archive_relative_path":f"data/raw/manual_imports/{r['archive']}",
               "source_split":r["split"], "image_path_in_zip":r["image_path_in_zip"],
               "label_path_in_zip":r["label_path_in_zip"], "original_id_hint":r.get("original_id_hint", ""),
               "width":r["width"],"height":r["height"],"sha256":r["image_sha256"],"phash":r["phash"],
               "source_group_hint":r["source_group"],"source_group_basis":r["group_basis"],
               "source_class_counts":r["source_class_counts"],
               "source_class_id_counts":json.dumps(id_counts,sort_keys=True),
               "source_box_count":r["box_count"],
               "source_polygon_count":r["polygon_count"],"mechanical_issues":r["issues"],
               "near_duplicate_candidates":r["near_duplicate_candidate_count"],
               "cross_stage2d_near_candidates":r.get("cross_archive_near_candidate_count", ""),
               "candidate_classification":status,"review_decision":"Pending",
               "rights_verified":"No","six_class_complete":"No","taxonomy_verified":"No",
               "duplicate_group_resolved":"No","target_label_path":"","target_class_counts":"",
               "final_group_id":"","accepted":"No","assigned_split":""}
        manifest.append(row)
        queue.append({"source":source,"image_path_in_zip":r["image_path_in_zip"],
                      "source_group_hint":r["source_group"],"candidate_classification":status,
                      "mechanical_issues":r["issues"],"source_class_counts":r["source_class_counts"],
                      "priority_action":REVIEW_ACTION[source],"decision_Approve_Reject_Correct":"Pending",
                      "corrected_label_path":"","rights_evidence":"","reviewer":"","review_date":"","notes":""})
    write_csv(STAGE / "unified_candidate_manifest_v0.csv", manifest, list(manifest[0]))
    write_csv(NEW / "unified_correction_review_queue.csv", queue, list(queue[0]))
    (STAGE / "README.md").write_text(
        "# Stage 2E unified candidate staging v0\n\n"
        "This area contains references to eight original ZIP archives, not approved training pixels. "
        "Every manifest row is unaccepted. Do not place images in data/processed or assign "
        "train/test/val until image-level taxonomy, six-class completeness, rights and duplicate "
        "review gates are documented and passed. Source split is historical metadata only.\n",
        encoding="utf-8")
    classification = []
    for source in sorted({r["source"] for r in manifest}):
        subset = [r for r in manifest if r["source"] == source]
        classification.append({"source":source,"candidate_images":len(subset),
                               "mechanically_unreadable_or_invalid":sum("unreadable_image" in r["mechanical_issues"] or "line_" in r["mechanical_issues"] for r in subset),
                               "candidate_quarantined":sum(r["candidate_classification"]=="quarantine_pending_review" for r in subset),
                               "source_reject_real_world":sum(r["candidate_classification"]=="source_reject_real_world" for r in subset),
                               "accepted":0,"six_class_reviewed":0,"rights_verified":0,
                               "omission_rate":"NA - no full six-class image reviews yet"})
    write_csv(NEW / "candidate_acceptance_and_omission_status.csv", classification, list(classification[0]))
    print(json.dumps({"manifest_rows":len(manifest),"review_queue_rows":len(queue),
                      "mapping_rows":len(maps),"accepted":0,"source_counts":dict(Counter(r["source"] for r in manifest))},indent=2))


if __name__ == "__main__":
    main()
