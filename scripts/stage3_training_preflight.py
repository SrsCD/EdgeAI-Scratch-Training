"""Read-only gate for a future scratch run. Never constructs or trains a model."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "configs/train.yaml"
DATA = ROOT / "configs/stage3_unified_data.yaml"
ARCH = ROOT / "configs/yolo11n.yaml"
PILOT_SPLITS = ROOT / "data/manifests/stage3b_pilot_v1_splits.csv"
PILOT_FREEZE = ROOT / "data/manifests/stage3b_pilot_v1_test_freeze.json"
LEGACY_SPLITS = ROOT / "data/manifests/stage2e_approved_group_splits_v1.csv"
SPLITS = PILOT_SPLITS if PILOT_SPLITS.exists() else LEGACY_SPLITS
PROCESSED = ROOT / "data/processed"
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_hash(path):
    with path.open("rb") as f:
        return sha256(f.read()).hexdigest()


def main():
    train = yaml.safe_load(TRAIN.read_text(encoding="utf-8"))
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if train["pretrained"] is not False or train["resume"] is not False:
        raise SystemExit("Scratch gate failed: pretrained and resume must both be false")
    if Path(train["model"]).suffix != ".yaml" or train["model"] != "configs/yolo11n.yaml":
        raise SystemExit("Scratch gate failed: initialize from the six-class architecture YAML")
    if train["amp"] is not False or train["device"] != 0 or int(train["seed"]) != 42:
        raise SystemExit("Scratch gate failed: check AMP, CUDA device and seed")
    if [data["names"][i] for i in range(6)] != NAMES or data["nc"] != 6:
        raise SystemExit("Six-class dataset contract changed")
    if not SPLITS.is_file():
        raise SystemExit("No frozen approved split manifest; training is blocked")
    with SPLITS.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Approved split manifest is empty")
    if SPLITS == PILOT_SPLITS:
        if not PILOT_FREEZE.is_file():
            raise SystemExit("Pilot independent test freeze is missing")
        freeze=json.loads(PILOT_FREEZE.read_text(encoding="utf-8"))
        test_rows=sorted((r["id"],r["sha256"],r["final_group_id"]) for r in rows if r["split"]=="test")
        test_digest=sha256(json.dumps(test_rows,sort_keys=True).encode()).hexdigest()
        if (freeze.get("split_manifest_sha256")!=file_hash(SPLITS) or
            freeze.get("test_rows_sha256")!=test_digest or freeze.get("test_rows")!=[list(r) for r in test_rows]):
            raise SystemExit("Pilot test freeze differs from approved split manifest")
    groups=defaultdict(set)
    hashes=defaultdict(set)
    counts=defaultdict(Counter)
    for r in rows:
        split=r["split"]
        if split not in {"train","test","val"} or not r["final_group_id"]:
            raise SystemExit("Missing split or final source group")
        groups[r["final_group_id"]].add(split)
        hashes[r["sha256"]].add(split)
        class_counts=json.loads(r["target_class_counts"])
        if set(class_counts) != set(NAMES):
            raise SystemExit("Target class count vector is incomplete")
        for name in NAMES:
            counts[split][name]+=int(class_counts[name])>0
    if any(len(s)>1 for s in groups.values()) or any(len(s)>1 for s in hashes.values()):
        raise SystemExit("Source-group or exact-image leakage between splits")
    if any(counts[split][name] == 0 for split in ("train","test","val") for name in NAMES):
        raise SystemExit("A class lacks positive images in a split")
    image_counts={}
    for split in ("train","test","val"):
        images=[p for p in (PROCESSED/"images"/split).iterdir() if p.suffix.lower() in EXT]
        labels=[p for p in (PROCESSED/"labels"/split).glob("*.txt")]
        if {p.stem for p in images} != {p.stem for p in labels} or len(images)!=len(labels):
            raise SystemExit(f"Processed image-label pairing invalid in {split}")
        expected=sum(r["split"]==split for r in rows)
        if len(images)!=expected:
            raise SystemExit(f"Processed {split} count {len(images)} differs from manifest {expected}")
        expected_hashes=Counter(r["sha256"] for r in rows if r["split"]==split)
        actual_hashes=Counter(file_hash(p) for p in images)
        if actual_hashes!=expected_hashes:
            raise SystemExit(f"Processed {split} image content differs from approved manifest")
        actual_boxes=Counter()
        for label in labels:
            for line_no,line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(),1):
                if not line.strip():
                    continue
                parts=line.split()
                if len(parts)!=5:
                    raise SystemExit(f"Invalid YOLO label row: {label}:{line_no}")
                cid=int(parts[0]); x,y,w,h=map(float,parts[1:])
                if cid not in range(6) or not all(map(math.isfinite,(x,y,w,h))) or not (
                    -1e-9<=x-w/2<x+w/2<=1+1e-9 and -1e-9<=y-h/2<y+h/2<=1+1e-9):
                    raise SystemExit(f"Invalid YOLO label values: {label}:{line_no}")
                actual_boxes[NAMES[cid]]+=1
        expected_boxes=Counter({name:sum(int(json.loads(r["target_class_counts"])[name]) for r in rows if r["split"]==split) for name in NAMES})
        if actual_boxes!=expected_boxes:
            raise SystemExit(f"Processed {split} class boxes differ from approved manifest")
        image_counts[split]=len(images)
    print(json.dumps({"ready_for_future_training":True,"split_images":image_counts,
                      "architecture_sha256":file_hash(ARCH),"train_config_sha256":file_hash(TRAIN),
                      "data_yaml_sha256":file_hash(DATA),"split_manifest_sha256":file_hash(SPLITS)},indent=2))


if __name__ == "__main__":
    main()
