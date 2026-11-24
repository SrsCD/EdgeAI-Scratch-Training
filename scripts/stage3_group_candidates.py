"""Build conservative, provisional leakage components from audit hints."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/stage3_candidates_v0/selection_and_conversion_manifest.csv"
OLD = ROOT / "reports/stage2d_manual/near_duplicate_candidates.csv"
NEW = ROOT / "reports/stage2e_unified/near_duplicate_candidates.csv"
OUT = ROOT / "reports/stage3"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def main():
    candidate = [r for r in read_csv(INPUT) if r["selection_status"] in
                 {"mapped_candidate_quarantine","no_mapped_target_quarantine"}]
    key = lambda r: r["source"] + "|" + r["image_path_in_zip"]
    index = {key(r):r for r in candidate}
    parent = {k:k for k in index}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    group_hints=defaultdict(list)
    for r in candidate:
        group_hints[r["source_group_hint"]].append(key(r))
    for members in group_hints.values():
        for k in members[1:]:
            union(members[0],k)
    edges=[]
    for path in (OLD,NEW):
        for e in read_csv(path):
            a = e.get("source_a",e.get("source", "")) + "|" + e.get("image_a", "")
            b = e.get("source_b", "") + "|" + e.get("image_b", "")
            if a in index and b in index:
                edges.append({"source_a":e["source_a"],"image_a":e["image_a"],
                              "source_b":e["source_b"],"image_b":e["image_b"],
                              "split_a":e["split_a"],"split_b":e["split_b"],
                              "phash_distance":e["phash_distance"],
                              "status":"unresolved_near_candidate"})
                union(a,b)
    components=defaultdict(list)
    for k,r in index.items():
        components[find(k)].append(r)
    ids={root:f"C{i:05d}" for i,root in enumerate(sorted(components),1)}
    members=[]
    summaries=[]
    for root, group in components.items():
        cid=ids[root]
        splits={r["source_split"] for r in group}
        sources={r["source"] for r in group}
        summaries.append({"provisional_component_id":cid,"image_count":len(group),
                          "sources":";".join(sorted(sources)),"source_splits":";".join(sorted(splits)),
                          "cross_source_split":len(splits)>1,
                          "status":"unresolved_candidate_not_final_group"})
        for r in group:
            members.append({"provisional_component_id":cid,"source":r["source"],
                            "image_path_in_zip":r["image_path_in_zip"],
                            "source_group_hint":r["source_group_hint"],
                            "source_split":r["source_split"],
                            "component_size":len(group),"status":"needs_adjudication_before_final_split"})
    write_csv(OUT/"provisional_component_members.csv",members,list(members[0]))
    write_csv(OUT/"provisional_component_summary.csv",summaries,list(summaries[0]))
    write_csv(OUT/"unresolved_near_pairs.csv",edges,
              list(edges[0]) if edges else ["source_a","image_a","source_b","image_b","split_a","split_b","phash_distance","status"])
    summary={"candidate_images":len(candidate),"source_group_hints_with_multiple_images":sum(len(v)>1 for v in group_hints.values()),
             "source_group_hints_cross_split":sum(len({index[k]["source_split"] for k in v})>1 for v in group_hints.values()),
             "unresolved_near_pairs":len(edges),
             "near_pairs_cross_source_split":sum(e["split_a"]!=e["split_b"] for e in edges),
             "provisional_components":len(components),
             "components_with_multiple_images":sum(len(v)>1 for v in components.values()),
             "components_cross_source_split":sum(len({r["source_split"] for r in v})>1 for v in components.values()),
             "images_in_multi_image_components":sum(len(v) for v in components.values() if len(v)>1),
             "exact_duplicates_found_in_stage2d_2e":0}
    (OUT/"duplicate_group_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
