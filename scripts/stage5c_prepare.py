"""Prepare Stage 5C review artifacts without accepting labels or starting training."""
from __future__ import annotations

import csv
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/experimental/stage3c_v1"
OUT = ROOT / "data/experimental/stage5c_v1"
REPORT = ROOT / "reports/stage5c"
CLASSES = ("person", "helmet", "safety_vest", "backpack", "knife", "pistol")
QUOTAS = {"construction_ppe_roboflow": 40, "backpack_roboflow": 4,
          "knife_source": 12, "pistols_roboflow": 6, "construction_site_safety": 10}
FRAMES = (5, 7, 11, 19)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_boxes(path: Path) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, w, h = line.split()
        boxes.append({"class_id": int(cls), "cx": float(cx), "cy": float(cy),
                      "w": float(w), "h": float(h)})
    return boxes


def score(row: dict, boxes: list[dict]) -> int:
    filename = Path(row["image_path_in_zip"]).name.lower()
    source = row["source"]
    if source == "construction_ppe_roboflow":
        return 10 * len({b["class_id"] for b in boxes} & {1, 2}) + min(len(boxes), 6)
    if source == "backpack_roboflow":
        return 20 * any(word in filename for word in
                        ("model", "man", "woman", "people", "person", "wear", "hiking"))
    if source == "knife_source":
        return 20 * any(word in filename for word in
                        ("defense", "attack", "fight", "combat", "human"))
    if source == "pistols_roboflow":
        return 0  # Taxonomy omits people; no filename proof of a person.
    return min(len(boxes), 8) + 4 * sum(b["class_id"] == 0 for b in boxes)


def select_audit(rows: list[dict]) -> list[dict]:
    old = {(r["source"], r["image_path_in_zip"]) for r in csv.DictReader(
        (SOURCE / "review_sample/queue.csv").open(encoding="utf-8-sig", newline=""))}
    selected = []
    for source, quota in QUOTAS.items():
        candidates = []
        for row in rows:
            if row["source"] != source or (source, row["image_path_in_zip"]) in old:
                continue
            boxes = label_boxes(SOURCE / "labels/train" / f"{row['id']}.txt")
            if source != "construction_site_safety" and any(b["class_id"] == 0 for b in boxes):
                continue
            candidates.append((row, boxes))
        candidates.sort(key=lambda pair: (-score(*pair),
            sha256(pair[0]["id"].encode()).hexdigest()))
        groups = set()
        picked = []
        for row, boxes in candidates:
            group = row["provisional_component_id"]
            if group in groups:
                continue
            groups.add(group)
            picked.append((row, boxes))
            if len(picked) == quota:
                break
        if len(picked) != quota:
            raise RuntimeError(f"Could only select {len(picked)} from {source}")
        selected.extend({**row, "boxes": boxes, "triage_score": score(row, boxes)}
                        for row, boxes in picked)
    return selected


def make_sheets(items: list[dict]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    for page in range((len(items) + 11) // 12):
        sheet = Image.new("RGB", (1200, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(items[page * 12:(page + 1) * 12]):
            image = Image.open(SOURCE / "images/train" / f"{item['id']}.jpg").convert("RGB")
            image.thumbnail((290, 250))
            x, y = (j % 4) * 300, (j // 4) * 300
            sheet.paste(image, (x, y))
            for box in item["boxes"]:
                x1 = x + (box["cx"] - box["w"] / 2) * image.width
                y1 = y + (box["cy"] - box["h"] / 2) * image.height
                x2 = x + (box["cx"] + box["w"] / 2) * image.width
                y2 = y + (box["cy"] + box["h"] / 2) * image.height
                draw.rectangle((x1, y1, x2, y2), outline="#f44336", width=2)
                draw.text((x1, y1), CLASSES[box["class_id"]], fill="#f44336")
            draw.text((x, y + 253), item["id"] + "  " + item["source"], fill="black")
            draw.text((x, y + 270), "UNREVIEWED: red = original label", fill="#b71c1c")
        sheet.save(REPORT / f"training_audit_{page + 1:02d}.jpg", quality=88)


def copy_baseline() -> dict:
    review_dir = OUT / "review_queue/reviews"
    if review_dir.exists() and any(review_dir.iterdir()):
        raise RuntimeError("Review files exist; refusing to regenerate Stage 5C")
    if (OUT / "review_queue/queue.json").exists():
        raise RuntimeError("Review queue exists; refusing to reset Stage 5C proposals")
    accepted = OUT / "accepted_corrections.csv"
    if accepted.exists() and len(accepted.read_text(encoding="utf-8-sig").splitlines()) > 1:
        raise RuntimeError("Accepted corrections exist; refusing to regenerate Stage 5C")
    totals = Counter()
    for split in ("train", "val"):
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            src, dst = SOURCE / kind / split, OUT / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            originals = sorted(src.glob(f"*{suffix}"))
            if any(dst.iterdir()) and len(list(dst.glob(f"*{suffix}"))) != len(originals):
                raise RuntimeError(f"Partial destination: {dst}")
            for original in originals:
                target = dst / original.name
                if not target.exists():
                    shutil.copy2(original, target)
                elif target.stat().st_size != original.stat().st_size:
                    raise RuntimeError(f"Changed copy: {target}")
            totals[f"{split}_{kind}"] = len(originals)
    return dict(totals)


def main() -> None:
    ledger = list(csv.DictReader((SOURCE / "ledger.csv").open(encoding="utf-8-sig", newline="")))
    train = [r for r in ledger if r["status"] == "included" and r["assigned_split"] == "train"]
    audit = select_audit(train)
    assert len(audit) == 72 and all(r["assigned_split"] == "train" for r in audit)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    copies = copy_baseline()
    make_sheets(audit)

    audit_rows = []
    for rank, item in enumerate(audit, 1):
        image = SOURCE / "images/train" / f"{item['id']}.jpg"
        label = SOURCE / "labels/train" / f"{item['id']}.txt"
        audit_rows.append({"rank": rank, "id": item["id"], "source": item["source"],
            "archive": item["archive"], "image_path_in_zip": item["image_path_in_zip"],
            "image_sha256": item["image_sha256"], "image_copy_sha256": digest(image),
            "source_group": item["source_group_hint"],
            "component_id": item["provisional_component_id"],
            "original_split": item["original_source_split"],
            "original_label_sha256": digest(label),
            "original_class_ids": ";".join(str(b["class_id"]) for b in item["boxes"]),
            "triage_score": item["triage_score"],
            "proposal": "inspect_for_omitted_person_and_other_six_class_targets",
            "suggested_box_count": 0, "decision": "unreviewed",
            "contact_sheet": f"training_audit_{(rank - 1) // 12 + 1:02d}.jpg"})
    write_csv(OUT / "training_audit_72.csv", audit_rows, list(audit_rows[0]))

    gif = ROOT / "data/demo_videos/cctv_test.gif"
    source_hash = digest(gif)
    cctv_dir = OUT / "cctv_review"
    cctv_dir.mkdir(exist_ok=True)
    with Image.open(gif) as image:
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            if index in FRAMES:
                frame.convert("RGB").save(cctv_dir / f"frame_{index:02d}.jpg", quality=100)
    cctv_rows = []
    for index in FRAMES:
        image = cctv_dir / f"frame_{index:02d}.jpg"
        cctv_rows.append({"id": f"P5C{index:02d}", "frame_number": index,
            "timestamp_seconds": f"{index / 2:.3f}", "source_gif": str(gif.relative_to(ROOT)),
            "source_sha256": source_hash, "image_sha256": digest(image),
            "source_group": "cctv_test.gif:single_scene", "split": "cctv_review_holdout",
            "decision": "pending_explicit_review", "training_eligible": "No",
            "image": str(image.relative_to(ROOT))})
    write_csv(cctv_dir / "queue.csv", cctv_rows, list(cctv_rows[0]))

    original_counts = {split: Counter() for split in ("train", "val")}
    for split in original_counts:
        for label in (OUT / "labels" / split).glob("*.txt"):
            original_counts[split].update(b["class_id"] for b in label_boxes(label))
    distribution = [{"split": split, "class_id": cid, "class_name": name,
                     "original_boxes": original_counts[split][cid],
                     "proposed_boxes": 0, "accepted_delta": 0,
                     "current_stage5c_boxes": original_counts[split][cid]}
                    for split in ("train", "val") for cid, name in enumerate(CLASSES)]
    write_csv(OUT / "class_distribution.csv", distribution, list(distribution[0]))
    write_csv(OUT / "accepted_corrections.csv", [],
              ["id", "split", "source_group", "review_record", "original_label_sha256",
               "new_label_sha256", "class_id", "change", "status"])
    write_csv(OUT / "unresolved_annotations.csv", [
        {"scope": "training_audit", "id": row["id"], "source_group": row["source_group"],
         "issue": row["proposal"], "status": "unreviewed"} for row in audit_rows] + [
        {"scope": "cctv_review", "id": row["id"], "source_group": row["source_group"],
         "issue": "complete_six_class_annotation_and_object_identity", "status": "pending_review"}
        for row in cctv_rows], ["scope", "id", "source_group", "issue", "status"])
    (OUT / "dataset_manifest.json").write_text(json.dumps({
        "version": "stage5c_v1", "status": "PREPARATION_ONLY_NO_ACCEPTED_CORRECTIONS",
        "source_dataset": str(SOURCE.relative_to(ROOT)), "source_ledger_sha256": digest(SOURCE / "ledger.csv"),
        "classes": list(CLASSES), "copied_baseline": copies,
        "train_candidates": len(audit), "accepted_corrections": 0,
        "cctv_frames_excluded_from_train_and_val": list(FRAMES),
        "test_not_copied_or_modified": True, "training_ready": False
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATUS.txt").write_text(
        "STAGE 5C PREPARATION ONLY / NOT TRAINING READY\n"
        "Train and val are untouched copies of Stage 3C provisional data. No proposed box has been accepted.\n"
        "CCTV frames are review holdout only. Do not create a training YAML until reviewed corrections and a validation protocol are ready.\n",
        encoding="utf-8")
    print(json.dumps({"audit_sample": len(audit), "cctv_frames": list(FRAMES),
                      "copied_baseline": copies, "accepted_corrections": 0}, indent=2))


if __name__ == "__main__":
    main()
