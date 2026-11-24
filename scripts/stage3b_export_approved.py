"""Export only fully reviewed pilot images and freeze a group-separated split."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/interim/stage3b_review_v1"
REVIEWS = BASE / "reviews"
OUT = ROOT / "data/processed"
MANIFESTS = ROOT / "data/manifests"
NAMES = ["person","helmet","safety_vest","backpack","knife","pistol"]
RATIOS = {"train":.7,"test":.2,"val":.1}
MIN_SUPPORT = {"train":10,"test":5,"val":5}


def h(path):
    with path.open("rb") as f:
        return sha256(f.read()).hexdigest()


def validate_box(b):
    cid=b["class_id"]
    vals=[float(b[k]) for k in ("cx","cy","w","h")]
    x,y,w,bh=vals
    if type(cid) is not int or cid not in range(6) or not all(map(math.isfinite,vals)) or not (
        -1e-9<=x-w/2<x+w/2<=1+1e-9 and -1e-9<=y-bh/2<y+bh/2<=1+1e-9):
        raise ValueError("invalid reviewed box")
    return cid,x,y,w,bh


def choose_splits(items):
    groups=defaultdict(list)
    for r in items: groups[r["review"]["final_group_id"]].append(r)
    support_groups={name:sum(any(any(b["class_id"]==cid for b in x["review"]["boxes"]) for x in group)
                             for group in groups.values()) for cid,name in enumerate(NAMES)}
    if any(v<3 for v in support_groups.values()):
        raise ValueError(f"fewer than three independent positive groups for a class: {support_groups}")
    all_support=Counter(name for r in items for name in {NAMES[b["class_id"]] for b in r["review"]["boxes"]})
    if any(all_support[name]<sum(MIN_SUPPORT.values()) for name in NAMES):
        raise ValueError(f"pilot lacks minimum positive-image support per class: {dict(all_support)}")
    best=None
    keys=list(groups)
    present={gid:Counter(name for r in members for name in
                         {NAMES[b["class_id"]] for b in r["review"]["boxes"]})
             for gid,members in groups.items()}
    lengths={gid:len(members) for gid,members in groups.items()}
    for trial in range(24):
        rng=random.Random(42+trial)
        rng.shuffle(keys)
        assignment={gid:"train" for gid in keys}
        sizes=Counter({"train":len(items),"test":0,"val":0})
        positives={s:Counter() for s in RATIOS}
        positives["train"].update(all_support)
        feasible=True
        # Reserve minimum reviewed positive support before balancing ratios.
        for split in ("val","test"):
            reserve={name:MIN_SUPPORT["train"]+(MIN_SUPPORT["test"] if split=="val" else 0)
                     for name in NAMES}
            while any(positives[split][name]<MIN_SUPPORT[split] for name in NAMES):
                deficits={name:max(0,MIN_SUPPORT[split]-positives[split][name]) for name in NAMES}
                options=[]
                for gid in keys:
                    if assignment[gid]!="train":continue
                    gain=sum(min(deficits[name],present[gid][name]) for name in NAMES)
                    if not gain or any(positives["train"][name]-present[gid][name]<reserve[name] for name in NAMES):
                        continue
                    options.append((gain/max(1,lengths[gid]),gain,-lengths[gid],rng.random(),gid))
                if not options:
                    feasible=False;break
                gid=max(options)[-1]
                assignment[gid]=split
                sizes["train"]-=lengths[gid];sizes[split]+=lengths[gid]
                positives["train"].subtract(present[gid]);positives[split].update(present[gid])
            if not feasible:break
        if not feasible:continue
        def loss():
            size_loss=sum(abs(sizes[s]/len(items)-RATIOS[s]) for s in RATIOS)
            class_loss=sum(abs(positives[s][name]/all_support[name]-RATIOS[s])
                           for s in RATIOS for name in NAMES)/18
            return size_loss+class_loss
        for gid in keys:
            if assignment[gid]!="train" or any(
                positives["train"][name]-present[gid][name]<MIN_SUPPORT["train"] for name in NAMES):
                continue
            options=[]
            for split in ("test","val"):
                image_deficit=RATIOS[split]*len(items)-sizes[split]
                if image_deficit<=0:continue
                class_deficit=sum(max(0,RATIOS[split]*all_support[name]-positives[split][name])
                                  *present[gid][name]/all_support[name] for name in NAMES)
                options.append((image_deficit/max(1,RATIOS[split]*len(items))+class_deficit,split))
            if not options:break
            split=max(options)[1]
            assignment[gid]=split
            sizes["train"]-=lengths[gid];sizes[split]+=lengths[gid]
            positives["train"].subtract(present[gid]);positives[split].update(present[gid])
        if any(positives[split][name]<MIN_SUPPORT[split] for split in RATIOS for name in NAMES):
            continue
        score=loss()
        if best is None or score<best[0]: best=(score,assignment.copy(),sizes.copy(),{s:positives[s].copy() for s in RATIOS})
    if best is None:
        raise ValueError("could not create group-separated splits with minimum per-class support")
    return best[1],best[2],best[3]


def main():
    queue=json.loads((BASE/"queue.json").read_text(encoding="utf-8"))["items"]
    approved=[]
    for item in queue:
        path=REVIEWS/f"{item['id']}.json"
        if not path.exists():continue
        review=json.loads(path.read_text(encoding="utf-8"))
        if review["decision"]!="Approve":continue
        if review["id"]!=item["id"] or review["image_sha256"]!=item["image_sha256"]:
            raise ValueError(f"review/source identity mismatch: {item['id']}")
        if not all(review.get(k) is True for k in ("taxonomy_checked","six_class_checked","rights_verified","group_resolved")):
            raise ValueError(f"approval gate incomplete: {item['id']}")
        if not review.get("reviewer") or not review.get("rights_evidence") or not review.get("final_group_id"):
            raise ValueError(f"approval evidence missing: {item['id']}")
        if not review["boxes"] and not review.get("verified_negative"):
            raise ValueError(f"unverified empty label: {item['id']}")
        for b in review["boxes"]: validate_box(b)
        if h(BASE/item["asset_path"])!=item["image_sha256"]:
            raise ValueError(f"review image content changed: {item['id']}")
        approved.append({"item":item,"review":review})
    if not approved:
        raise SystemExit("No human-approved images. Review pilot images before export.")
    # The pilot selector avoided known near pairs. A final hash/pHash check
    # still protects against changed approvals or accidentally mixed groups.
    for i,a in enumerate(approved):
        for b in approved[i+1:]:
            if a["item"]["image_sha256"]==b["item"]["image_sha256"] and a["review"]["final_group_id"]!=b["review"]["final_group_id"]:
                raise ValueError("exact duplicates assigned different groups")
            pa,pb=a["item"].get("phash"),b["item"].get("phash")
            if pa and pb and (int(pa,16)^int(pb,16)).bit_count()<=4 and a["review"]["final_group_id"]!=b["review"]["final_group_id"]:
                raise ValueError("near-duplicate candidates assigned different groups; adjudicate before export")
    assignment,sizes,support=choose_splits(approved)
    MANIFESTS.mkdir(exist_ok=True)
    manifest=MANIFESTS/"stage3b_pilot_v1_splits.csv"
    freeze=MANIFESTS/"stage3b_pilot_v1_test_freeze.json"
    if manifest.exists() or freeze.exists():
        raise SystemExit("Pilot v1 split is already frozen; do not overwrite it")
    for split in RATIOS:
        for kind in ("images","labels"):
            existing=[p for p in (OUT/kind/split).iterdir() if p.is_file() and p.name!=".gitkeep"]
            if existing: raise SystemExit(f"Processed {kind}/{split} is nonempty; export refused")
    rows=[]
    for r in approved:
        item,review=r["item"],r["review"]
        split=assignment[review["final_group_id"]]
        counts=Counter(NAMES[b["class_id"]] for b in review["boxes"])
        rows.append({"id":item["id"],"source":item["source"],"archive":item["archive"],
                     "image_path_in_zip":item["image_path_in_zip"],"sha256":item["image_sha256"],
                     "final_group_id":review["final_group_id"],"split":split,
                     "target_class_counts":json.dumps({name:counts.get(name,0) for name in NAMES},sort_keys=True),
                     "reviewer":review["reviewer"],"reviewed_at_utc":review["reviewed_at_utc"],
                     "rights_evidence":review["rights_evidence"],"source_url":item["source_url"],
                     "source_license_claim":item["source_license_claim"]})
    # All checks above precede this materialization step. Original ZIPs and
    # review assets stay untouched. Filenames use stable pilot IDs.
    for r in approved:
        item,review=r["item"],r["review"]
        split=assignment[review["final_group_id"]]
        src=BASE/item["asset_path"]
        dest=OUT/"images"/split/f"{item['id']}{src.suffix.lower()}"
        shutil.copyfile(src,dest)
        label=OUT/"labels"/split/f"{item['id']}.txt"
        label.write_text("".join(f"{b['class_id']} {b['cx']:.10f} {b['cy']:.10f} {b['w']:.10f} {b['h']:.10f}\n"
                                 for b in review["boxes"]),encoding="utf-8")
    with manifest.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    test_rows=sorted((r["id"],r["sha256"],r["final_group_id"]) for r in rows if r["split"]=="test")
    freeze.write_text(json.dumps({"version":"stage3b_pilot_v1","test_rows":test_rows,
                                  "test_rows_sha256":sha256(json.dumps(test_rows,sort_keys=True).encode()).hexdigest(),
                                  "split_manifest_sha256":h(manifest)},indent=2)+"\n",encoding="utf-8")
    stats=[]
    for cid,name in enumerate(NAMES):
        stat={"class_id":cid,"class_name":name}
        for split in ("train","test","val"):
            members=[r for r in approved if assignment[r["review"]["final_group_id"]]==split]
            stat[f"{split}_positive_images"]=sum(any(b["class_id"]==cid for b in r["review"]["boxes"]) for r in members)
            stat[f"{split}_boxes"]=sum(b["class_id"]==cid for r in members for b in r["review"]["boxes"])
        stats.append(stat)
    stats_path=ROOT/"reports/stage3b/pilot_split_stats.csv"
    stats_path.parent.mkdir(parents=True,exist_ok=True)
    with stats_path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    print(json.dumps({"accepted":len(rows),"split_images":dict(sizes),
                      "class_positive_images":{s:dict(support[s]) for s in RATIOS},
                      "frozen_test_images":len(test_rows),"manifest":str(manifest),
                      "per_class_stats":str(stats_path)},indent=2))


if __name__=="__main__":main()
