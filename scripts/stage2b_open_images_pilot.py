"""Selectively acquire an Open Images V7 pilot with live Flickr-license checks.

Run only within the approved Stage 2B pilot. Never imports a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/stage2b_v1"
OUT = RAW / "open_images_validation"
OUT.mkdir(exist_ok=True)
MIDS = {
    "/m/01g317": "person",
    "/m/0zvk5": "helmet",
    "/m/01940j": "backpack",
    "/m/04ctx": "knife",
    "/m/0gxl3": "pistol_candidate",
}
LICENSE = "https://creativecommons.org/licenses/by/2.0/"
CAP_PER_CLASS = 10
MAX_BYTES_PER_IMAGE = 5_000_000


def main():
    all_boxes = {}
    candidates = {mid: [] for mid in MIDS}
    with (RAW / "validation-annotations-bbox.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_boxes.setdefault(row["ImageID"], []).append(row)
            if row["LabelName"] in MIDS and row["ImageID"] not in candidates[row["LabelName"]]:
                candidates[row["LabelName"]].append(row["ImageID"])
    selected = {}
    for mid, ids in candidates.items():
        for image_id in sorted(ids)[:CAP_PER_CLASS]:
            selected.setdefault(image_id, []).append(MIDS[mid])
    metadata = {}
    with (RAW / "validation-images-with-rotation.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] in selected:
                metadata[row["ImageID"]] = row

    session = requests.Session()
    records = []
    total_bytes = 0
    for image_id in sorted(selected):
        meta = metadata.get(image_id)
        rec = {"image_id": image_id, "selected_for": selected[image_id], "source_split": "validation",
               "metadata": meta, "source_boxes": all_boxes.get(image_id, []), "status": "quarantined",
               "rights_verified": False, "transferred_bytes": 0}
        try:
            if not meta or meta["License"] != LICENSE:
                rec["reason"] = "missing metadata or license not CC BY 2.0"
            elif int(meta.get("OriginalSize") or 0) > MAX_BYTES_PER_IMAGE:
                rec["reason"] = "original exceeds per-image pilot cap"
            elif shutil.disk_usage(ROOT).free < 31 * 1024**3:
                rec["reason"] = "disk reserve floor"
            else:
                landing = session.get(meta["OriginalLandingURL"], timeout=30)
                rec["landing_http_status"] = landing.status_code
                rec["landing_checked_url"] = landing.url
                # The live Flickr page must show this precise license URI.
                found = landing.ok and bool(re.search(r"creativecommons\.org/licenses/by/2\.0/", landing.text))
                rec["rights_verified"] = found
                if not found:
                    rec["reason"] = "live landing page did not confirm CC BY 2.0"
                else:
                    path = OUT / f"{image_id}.jpg"
                    if path.exists():
                        rec["raw_path"] = str(path.relative_to(ROOT))
                        rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        rec["reason"] = "existing raw file retained; awaiting review"
                        records.append(rec)
                        print(image_id, rec["reason"], 0, flush=True)
                        continue
                    url = meta["OriginalURL"]
                    response = session.get(url, stream=True, timeout=45)
                    rec["image_http_status"] = response.status_code
                    response.raise_for_status()
                    sha = hashlib.sha256()
                    try:
                        with path.open("xb") as output:
                            for chunk in response.iter_content(1024 * 256):
                                if chunk:
                                    total_bytes += len(chunk)
                                    rec["transferred_bytes"] += len(chunk)
                                    if rec["transferred_bytes"] > MAX_BYTES_PER_IMAGE:
                                        raise RuntimeError("download exceeds per-image cap")
                                    output.write(chunk)
                                    sha.update(chunk)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                    rec["raw_path"] = str(path.relative_to(ROOT))
                    rec["sha256"] = sha.hexdigest()
                    rec["reason"] = "awaiting six-class visual and annotation review"
        except Exception as error:
            rec["error"] = f"{type(error).__name__}: {error}"
            rec["reason"] = "download/access failure"
        records.append(rec)
        print(image_id, rec["reason"], rec["transferred_bytes"], flush=True)
        time.sleep(0.2)
    manifest = RAW / "open_images_pilot_manifest.json"
    manifest.write_text(json.dumps({"selection_rule": "first 10 sorted IDs for each of five exact MID classes",
                                    "total_image_bytes_transferred": total_bytes, "records": records}, indent=2) + "\n", encoding="utf-8")
    print("selected", len(records), "downloaded", sum(bool(r.get("sha256")) for r in records),
          "image_bytes", total_bytes)


if __name__ == "__main__":
    main()
