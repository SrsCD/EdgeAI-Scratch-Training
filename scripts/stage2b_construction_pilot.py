"""Extract and mechanically audit 150 original Construction-PPE pilot images."""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import shutil
from zipfile import ZipFile

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/stage2b_v1"
ZIP = RAW / "construction-ppe.zip"
OUT = RAW / "construction_ppe_subset"
MAP = {0: 1, 2: 2, 5: 0}  # candidate mapping only; vest semantic review pending
NAMES = ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"]


def main():
    if shutil.disk_usage(ROOT).free < 31 * 1024**3:
        raise RuntimeError("30 GiB free-space floor")
    OUT.mkdir(exist_ok=True)
    with ZipFile(ZIP) as z:
        archive = {item.filename: item for item in z.infolist()}
        source = [json.loads(line) for line in z.read("construction-ppe/manifest.jsonl").decode().splitlines()]
        rng = random.Random(42)
        chosen = []
        for prefix in ("source-", "ultralytics-"):
            pool = sorted((r for r in source if r["file"].startswith(prefix)), key=lambda r: r["file"])
            chosen += rng.sample(pool, 75)
        records = []
        phashes = {}
        counts = Counter()
        image_counts = Counter()
        sha_groups = defaultdict(list)
        for r in chosen:
            image_name = f"construction-ppe/images/{r['split']}/{r['file']}"
            label_name = f"construction-ppe/labels/{r['split']}/{Path(r['file']).stem}.txt"
            rec = {"source_id": "ultralytics_construction_ppe_2026", "source_image_id": r["file"],
                   "source_split": r["split"], "archive_image_path": image_name,
                   "archive_label_path": label_name, "license": "CC BY 4.0 (publisher assertion)",
                   "source_group_id": None, "synthetic_status": "unknown_per_image",
                   "status": "quarantined", "reason": "six-class visual completeness, vest semantics, provenance and synthetic status pending"}
            try:
                data = z.read(image_name)
                labels = z.read(label_name).decode().splitlines()
                digest = hashlib.sha256(data).hexdigest()
                if digest != r["sha256"]:
                    raise ValueError("publisher manifest SHA-256 mismatch")
                image = Image.open(BytesIO(data))
                image.verify()
                image = Image.open(BytesIO(data)).convert("RGB")
                rec["width"], rec["height"] = image.size
                rec["sha256"] = digest
                rec["phash"] = str(imagehash.phash(image))
                rec["original_bytes"] = len(data)
                rec["source_box_count"] = len(labels)
                if len(labels) != r["annotations"]:
                    raise ValueError("publisher annotation count mismatch")
                mapped = Counter()
                for line in labels:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError("label does not contain five fields")
                    cls = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                    if cls not in range(12) or not all(0 <= x <= 1 for x in vals):
                        raise ValueError("invalid class or coordinate")
                    x, y, w, h = vals
                    if not (w > 0 and h > 0 and x-w/2 >= -1e-5 and y-h/2 >= -1e-5 and
                            x+w/2 <= 1+1e-5 and y+h/2 <= 1+1e-5):
                        raise ValueError("invalid box extent")
                    if cls in MAP:
                        mapped[MAP[cls]] += 1
                rec["mapped_source_box_counts"] = {NAMES[k]: v for k, v in mapped.items()}
                for k, v in mapped.items():
                    counts[NAMES[k]] += v
                    image_counts[NAMES[k]] += 1
                sha_groups[digest].append(r["file"])
                phashes[r["file"]] = rec["phash"]
                target = OUT / r["split"]
                target.mkdir(exist_ok=True)
                (target / r["file"]).write_bytes(data)
                (target / f"{Path(r['file']).stem}.txt").write_bytes(z.read(label_name))
                rec["raw_image_path"] = str((target / r["file"]).relative_to(ROOT))
            except Exception as error:
                rec["status"] = "rejected"
                rec["reason"] = f"mechanical validation failed: {type(error).__name__}: {error}"
            records.append(rec)
    names = sorted(phashes)
    near = []
    for i, a in enumerate(names):
        h1 = imagehash.hex_to_hash(phashes[a])
        for b in names[i+1:]:
            d = h1 - imagehash.hex_to_hash(phashes[b])
            if d <= 4:
                near.append({"a": a, "b": b, "phash_distance": d, "status": "review_candidate"})
    report = {"selection": "seed 42, 75 source-* and 75 ultralytics-* from full owner manifest",
              "records": records, "mapped_source_positive_images": dict(image_counts),
              "mapped_source_boxes": dict(counts),
              "exact_duplicate_groups": [v for v in sha_groups.values() if len(v) > 1],
              "near_duplicate_candidates": near}
    (RAW / "construction_pilot_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mapped_source_positive_images", "mapped_source_boxes",
                       "exact_duplicate_groups", "near_duplicate_candidates")}, indent=2))
    print("chosen", len(chosen), "quarantined", sum(r["status"] == "quarantined" for r in records),
          "rejected", sum(r["status"] == "rejected" for r in records))


if __name__ == "__main__":
    main()
