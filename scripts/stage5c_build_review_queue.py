"""Build six-image offline review queue; visual drafts are never ground truth."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/experimental/stage5c_v1"
SOURCE = ROOT / "data/experimental/stage3c_v1"
QUEUE = DATA / "review_queue"
REPORT = ROOT / "reports/stage5c"

# Pixel xyxy, drafted by visual inspection of the source image. Reviewer must
# correct every boundary and verify every visible six-class object.
CCTV_PEOPLE = {
    5: [(96, 59, 172, 184), (158, 36, 187, 113)],
    7: [(71, 60, 145, 207), (149, 35, 186, 130)],
    11: [(0, 73, 83, 228), (76, 51, 144, 181), (129, 39, 171, 153)],
    19: [(0, 64, 60, 230), (58, 75, 134, 230), (117, 27, 170, 177)],
}
TRAIN_PEOPLE = {
    "f74370649bf23e45af01": [(265, 185, 505, 579)],
    "e4a11c4fcafff67184c3": [(174, 84, 730, 720), (674, 75, 1280, 720)],
}
TRAIN_NOTES = {
    "f74370649bf23e45af01": "Foreground worker draft only. Distant workers and all other target objects need explicit review.",
    "e4a11c4fcafff67184c3": "Two partly clipped people. Verify full extents and the original knife label.",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def normalized(box: tuple[int, int, int, int], width: int, height: int) -> dict:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError((box, width, height))
    return {"class_id": 0, "cx": (x1 + x2) / (2 * width),
            "cy": (y1 + y2) / (2 * height), "w": (x2 - x1) / width,
            "h": (y2 - y1) / height}


def original_boxes(path: Path) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cid, cx, cy, w, h = line.split()
        boxes.append({"class_id": int(cid), "cx": float(cx), "cy": float(cy),
                      "w": float(w), "h": float(h)})
    return boxes


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (QUEUE / "reviews").exists() and any((QUEUE / "reviews").iterdir()):
        raise RuntimeError("Existing human review records must never be overwritten")
    (QUEUE / "assets").mkdir(parents=True, exist_ok=True)
    (QUEUE / "reviews").mkdir(exist_ok=True)
    audit = {r["id"]: r for r in csv.DictReader(
        (DATA / "training_audit_72.csv").open(encoding="utf-8-sig", newline=""))}
    items, proposals = [], []
    for frame, pixel_boxes in CCTV_PEOPLE.items():
        src = DATA / "cctv_review" / f"frame_{frame:02d}.jpg"
        ident = f"P5C{frame:02d}"
        dst = QUEUE / "assets" / f"{ident}.jpg"
        shutil.copy2(src, dst)
        with Image.open(src) as im:
            width, height = im.size
        draft = [normalized(b, width, height) for b in pixel_boxes]
        items.append({"id": ident, "source": "cctv_test.gif", "asset_path": f"assets/{ident}.jpg",
            "image_sha256": digest(src), "frame_number": frame, "source_group": "cctv_test.gif:single_scene",
            "source_group_hint": "cctv_test.gif:single_scene", "archive": "local cctv_test.gif",
            "image_path_in_zip": f"frame {frame}", "source_split": "CCTV review holdout",
            "source_license_claim": "Unverified local demo footage; confirm rights before approval.",
            "source_url": "#",
            "assigned_split": "cctv_review_holdout", "original_boxes": [],
            "provisional_boxes": draft,
            "review_flags": "UNVERIFIED visual person drafts; verify all people and six classes; handgun-shaped regions are unresolved, not pistol labels",
            "decision": "Pending"})
        for i, (box, xyxy) in enumerate(zip(draft, pixel_boxes), 1):
            proposals.append({"id": ident, "split": "cctv_review_holdout", "source_group": "cctv_test.gif:single_scene",
                "class_id": 0, "class_name": "person", "proposal_number": i,
                "pixel_xyxy": ",".join(map(str, xyxy)),
                "normalized_xywh": ",".join(f"{box[k]:.6f}" for k in ("cx", "cy", "w", "h")),
                "basis": "assistant_visual_draft_from_source_frame", "status": "unverified_not_ground_truth"})
    for source_id, pixel_boxes in TRAIN_PEOPLE.items():
        row = audit[source_id]  # Enforces training-only and the 72-image sample.
        if row["source"] not in {"construction_ppe_roboflow", "knife_source"}:
            raise RuntimeError("Unexpected selected source")
        src = SOURCE / "images/train" / f"{source_id}.jpg"
        label = SOURCE / "labels/train" / f"{source_id}.txt"
        ident = "P5CT1" if source_id.startswith("f743") else "P5CT2"
        dst = QUEUE / "assets" / f"{ident}.jpg"
        shutil.copy2(src, dst)
        with Image.open(src) as im:
            width, height = im.size
        draft = [normalized(b, width, height) for b in pixel_boxes]
        original = original_boxes(label)
        items.append({"id": ident, "source": row["source"], "source_image_id": source_id,
            "asset_path": f"assets/{ident}.jpg", "image_sha256": digest(src),
            "source_group": row["source_group"], "component_id": row["component_id"],
            "source_group_hint": row["source_group"], "archive": row["archive"],
            "image_path_in_zip": row["image_path_in_zip"], "source_split": "train",
            "source_license_claim": "Underlying image rights unverified; confirm before approval.",
            "source_url": "#",
            "assigned_split": "train", "original_label_sha256": digest(label),
            "original_boxes": original, "provisional_boxes": original + draft,
            "review_flags": "UNVERIFIED person additions shown with original boxes; " + TRAIN_NOTES[source_id],
            "decision": "Pending"})
        for i, (box, xyxy) in enumerate(zip(draft, pixel_boxes), 1):
            proposals.append({"id": ident, "split": "train", "source_group": row["source_group"],
                "class_id": 0, "class_name": "person", "proposal_number": i,
                "pixel_xyxy": ",".join(map(str, xyxy)),
                "normalized_xywh": ",".join(f"{box[k]:.6f}" for k in ("cx", "cy", "w", "h")),
                "basis": "assistant_visual_draft_from_training_image", "status": "unverified_not_ground_truth"})
    if len(items) != 6:
        raise RuntimeError("Review workload must be exactly six images")
    (QUEUE / "queue.json").write_text(json.dumps({"version": "stage5c_six_image_pending_review",
        "warning": "Provisional boxes include unverified visual drafts, not accepted labels.",
        "items": items}, indent=2) + "\n", encoding="utf-8")
    csv_write(QUEUE / "queue.csv", [{"id": x["id"], "source": x["source"],
        "source_image_id": x.get("source_image_id", ""), "source_group": x["source_group"],
        "split": x["assigned_split"], "original_boxes": len(x["original_boxes"]),
        "unverified_person_drafts": len(x["provisional_boxes"]) - len(x["original_boxes"]),
        "decision": "Pending"} for x in items],
        ["id", "source", "source_image_id", "source_group", "split", "original_boxes",
         "unverified_person_drafts", "decision"])
    csv_write(DATA / "proposed_corrections.csv", proposals, list(proposals[0]))
    distribution_path = DATA / "class_distribution.csv"
    distribution = list(csv.DictReader(distribution_path.open(encoding="utf-8-sig", newline="")))
    for row in distribution:
        row["proposed_boxes"] = sum(p["split"] == row["split"] and
                                    str(p["class_id"]) == row["class_id"] for p in proposals)
    csv_write(distribution_path, distribution, list(distribution[0]))
    manifest_path = DATA / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"proposed_train_person_boxes": 3,
                     "proposed_cctv_person_boxes_excluded_from_training": 10,
                     "pending_manual_review_images": 6})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    csv_write(DATA / "unresolved_object_regions.csv", [
        {"id": "P5C05", "pixel_xyxy": "94,104,116,122",
         "issue": "handgun-shaped object; pistol identity requires explicit review", "class_id": "", "status": "unresolved"},
        {"id": "P5C07", "pixel_xyxy": "71,116,99,136",
         "issue": "handgun-shaped object; pistol identity requires explicit review", "class_id": "", "status": "unresolved"},
        {"id": "P5CT1", "pixel_xyxy": "202,180,380,310",
         "issue": "distant workers need individual box review", "class_id": "", "status": "unresolved"},
    ], ["id", "pixel_xyxy", "issue", "class_id", "status"])
    # Render only review proposals. This contact sheet does not confer approval.
    sheet = Image.new("RGB", (1000, 1510), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        image = Image.open(QUEUE / item["asset_path"]).convert("RGB")
        image.thumbnail((480, 450))
        x, y = (index % 2) * 500 + 10, (index // 2) * 500 + 10
        sheet.paste(image, (x, y))
        for box in item["original_boxes"]:
            draw.rectangle((x + (box["cx"] - box["w"] / 2) * image.width,
                            y + (box["cy"] - box["h"] / 2) * image.height,
                            x + (box["cx"] + box["w"] / 2) * image.width,
                            y + (box["cy"] + box["h"] / 2) * image.height),
                           outline="#ee4444", width=2)
        for box in item["provisional_boxes"][len(item["original_boxes"]):]:
            draw.rectangle((x + (box["cx"] - box["w"] / 2) * image.width,
                            y + (box["cy"] - box["h"] / 2) * image.height,
                            x + (box["cx"] + box["w"] / 2) * image.width,
                            y + (box["cy"] + box["h"] / 2) * image.height),
                           outline="#00aa44", width=3)
        draw.text((x, y + 455), f"{item['id']}  UNREVIEWED  green=draft person  red=original", fill="#b00020")
    sheet.save(REPORT / "six_image_review_contact_sheet.jpg", quality=92)
    print(f"Prepared {len(items)} pending review images and {len(proposals)} unverified person boxes")


if __name__ == "__main__":
    main()
