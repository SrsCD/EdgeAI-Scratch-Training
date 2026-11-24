"""Create group-aware 70/20/10 manifests only after all review gates pass.

This script does not copy images. It refuses an empty or incompletely approved
pool and writes no split artifact unless every target class appears in all three.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/stage2e_unified_v0/unified_candidate_manifest_v0.csv"
OUTPUT = ROOT / "data/manifests"
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
RATIOS = {"train":0.7,"test":0.2,"val":0.1}
SEED = 42


def main():
    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    approved = [r for r in rows if r["accepted"] == "Yes"]
    if not approved:
        raise SystemExit("No approved images: split not created. Complete human review first.")
    required = ("rights_verified","six_class_complete","taxonomy_verified","duplicate_group_resolved")
    for r in approved:
        image = r["image_path_in_zip"]
        if r["review_decision"] != "Approve" or any(r[k] != "Yes" for k in required):
            raise SystemExit(f"Review gate incomplete: {image}")
        if not r["final_group_id"] or not r["target_label_path"] or not (ROOT / r["target_label_path"]).is_file():
            raise SystemExit(f"Group or reviewed label missing: {image}")
        try:
            counts = json.loads(r["target_class_counts"])
            if set(counts) != set(NAMES) or any(not isinstance(counts[k],int) or counts[k] < 0 for k in NAMES):
                raise ValueError("invalid class-count vector")
        except (json.JSONDecodeError, ValueError) as e:
            raise SystemExit(f"Reviewed target counts invalid for {image}: {e}")
        r["_counts"] = counts
    groups = defaultdict(list)
    for r in approved:
        groups[r["final_group_id"]].append(r)
    class_groups = {name:sum(any(r["_counts"][name] > 0 for r in g) for g in groups.values()) for name in NAMES}
    if any(v < 3 for v in class_groups.values()):
        raise SystemExit(f"Insufficient independent positive groups for all three splits: {class_groups}")

    rng = random.Random(SEED)
    group_list = list(groups.items())
    rng.shuffle(group_list)
    rarity = {name:1/class_groups[name] for name in NAMES}
    group_list.sort(key=lambda item:-sum(rarity[name] for name in NAMES if any(r["_counts"][name]>0 for r in item[1])))
    assignments = {}
    sizes = Counter()
    support = defaultdict(Counter)
    total = len(approved)
    for gid, members in group_list:
        present = [name for name in NAMES if any(r["_counts"][name]>0 for r in members)]
        def score(split):
            missing_bonus = sum(100*rarity[name] for name in present if support[split][name]==0)
            size_penalty = abs((sizes[split]+len(members))-RATIOS[split]*total) / total
            overflow = max(0,(sizes[split]+len(members))-RATIOS[split]*total) / total
            return missing_bonus - size_penalty - 2*overflow
        split = max(RATIOS, key=score)
        assignments[gid] = split
        sizes[split] += len(members)
        for name in present:
            support[split][name] += sum(r["_counts"][name]>0 for r in members)
    if any(support[split][name]==0 for split in RATIOS for name in NAMES):
        raise SystemExit("Group allocation could not cover all classes in each split; no split written.")

    output_rows = []
    distribution = []
    for r in approved:
        output_rows.append({"source":r["source"],"archive":r["archive"],
                            "image_path_in_zip":r["image_path_in_zip"],"target_label_path":r["target_label_path"],
                            "sha256":r["sha256"],"final_group_id":r["final_group_id"],
                            "split":assignments[r["final_group_id"]],"target_class_counts":json.dumps(r["_counts"],sort_keys=True)})
    for name in NAMES:
        row = {"class_name":name}
        for split in RATIOS:
            subset = [r for r in approved if assignments[r["final_group_id"]]==split]
            row[f"{split}_positive_images"] = sum(r["_counts"][name]>0 for r in subset)
            row[f"{split}_objects"] = sum(r["_counts"][name] for r in subset)
        row["total_positive_images"] = sum(r["_counts"][name]>0 for r in approved)
        row["total_objects"] = sum(r["_counts"][name] for r in approved)
        distribution.append(row)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    def write(path, content):
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(content[0]))
            w.writeheader(); w.writerows(content)
    write(OUTPUT / "stage2e_approved_group_splits_v1.csv",output_rows)
    write(OUTPUT / "stage2e_approved_class_distribution_v1.csv",distribution)
    print(json.dumps({"images":total,"actual_split_counts":dict(sizes),
                      "actual_split_percentages":{k:round(100*sizes[k]/total,2) for k in RATIOS},
                      "class_positive_groups":class_groups,"seed":SEED},indent=2))


if __name__ == "__main__":
    main()
