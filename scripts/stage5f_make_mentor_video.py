"""Render a curated still-image demo from verified Stage 5E scratch predictions."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage5a_detect_video import CHECKPOINT, EXPECTED_SHA256, NAMES, check_checkpoint

OUT = ROOT / "reports/stage5f"
AUDIT = ROOT / "reports/stage5e/audit.csv"
ORDER = (
    "41485109fdf1919d0c12", "8c1075078f66e20bc039",
    "ebcb35f09a0a5e92b978", "f7c8b829136f7e776ef8",
    "3a81a64097fac7ad07eb", "44c2df5ab8b9c056c0df",
    "e55de220eea28a6db68c",
)
W, H, FPS = 1280, 720, 24
BG = "#0b1020"
WHITE = "#f3f6ff"
MUTED = "#aab6cb"
ACCENT = "#41d6b4"
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def base() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def text(draw: ImageDraw.ImageDraw, xy, words, size, fill=WHITE, bold=False):
    draw.text(xy, words, font=font(size, bold), fill=fill)


def fit(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, int, int]:
    x, y, width, height = box
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy, x + (width - copy.width) // 2, y + (height - copy.height) // 2


def model_boxes(model, path: Path, device) -> list[dict]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read {path}")
    result = model.predict(source=frame, imgsz=640, device=device, conf=0.25,
                           iou=0.70, verbose=False)[0]
    found = []
    for box in result.boxes:
        cid = int(box.cls[0])
        found.append({"class_id": cid, "class_name": NAMES[cid],
                      "confidence": float(box.conf[0]),
                      "xyxy": [float(v) for v in box.xyxy[0].tolist()]})
    return found


def annotated(source: Image.Image, boxes: list[dict]) -> Image.Image:
    image = source.copy()
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 150))
    label_font = font(max(17, round(min(image.size) / 30)), True)
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        color = "#ffca52" if box["class_name"] == "safety_vest" else ACCENT
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{box['class_name']}  {box['confidence']:.2f}"
        bounds = draw.textbbox((0, 0), label, font=label_font)
        label_w, label_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        label_y = max(0, round(y1) - label_h - 8)
        draw.rounded_rectangle((x1, label_y, min(image.width, x1 + label_w + 12),
                                label_y + label_h + 8), radius=3, fill=color)
        draw.text((x1 + 6, label_y + 2), label, font=label_font, fill="#081220")
    return image


def slide(row: dict, boxes: list[dict], number: int) -> Image.Image:
    image = base()
    draw = ImageDraw.Draw(image)
    section = "BACKPACK DETECTION" if row["focus_class"] == "backpack" else "SAFETY VEST DETECTION"
    text(draw, (54, 31), section, 20, ACCENT, True)
    text(draw, (54, 66), f"Selected validation image  {number:02d} / 07", 32, WHITE, True)
    text(draw, (54, 111), "YOLO11n trained from scratch  |  confidence threshold 0.25", 20, MUTED)
    draw.rounded_rectangle((50, 164, 621, 610), radius=16, fill="#172238")
    draw.rounded_rectangle((659, 164, 1230, 610), radius=16, fill="#172238")
    text(draw, (72, 180), "SOURCE IMAGE", 18, MUTED, True)
    text(draw, (681, 180), "ACTUAL MODEL PREDICTION", 18, ACCENT, True)
    src = Image.open(ROOT / row["image"]).convert("RGB")
    pred = annotated(src, boxes)
    for panel, bounds in ((src, (68, 216, 535, 375)), (pred, (677, 216, 535, 375))):
        fitted, x, y = fit(panel, bounds)
        image.paste(fitted, (x, y))
    counts = []
    for name in dict.fromkeys(box["class_name"] for box in boxes):
        scores = ", ".join(f"{box['confidence']:.2f}" for box in boxes if box["class_name"] == name)
        counts.append(f"{name}: {scores}")
    summary = "   |   ".join(counts)
    if len(summary) > 105:
        summary = summary[:102] + "..."
    text(draw, (54, 632), summary, 19, WHITE, True)
    text(draw, (54, 676), "Curated validation-image sequence  •  still images, not continuous video inference", 17, MUTED)
    return image


def title_slide() -> Image.Image:
    im = base(); d = ImageDraw.Draw(im)
    d.rounded_rectangle((56, 70, 1224, 650), radius=28, fill="#172238")
    d.rectangle((84, 111, 94, 590), fill=ACCENT)
    text(d, (128, 143), "YOLO11n — Object Detection", 51, WHITE, True)
    text(d, (128, 212), "Trained From Scratch", 51, WHITE, True)
    text(d, (128, 339), "Curated validation-image sequence", 29, ACCENT, True)
    text(d, (128, 390), "Backpack and safety-vest examples", 25, MUTED)
    text(d, (128, 560), "Still-image transitions • confidence threshold 0.25", 19, MUTED)
    return im


def closing_slide() -> Image.Image:
    im = base(); d = ImageDraw.Draw(im)
    text(d, (65, 62), "MODEL AND DEMO CONTEXT", 26, ACCENT, True)
    text(d, (65, 126), "Scratch-trained YOLO11n", 43, WHITE, True)
    text(d, (65, 220), "Six configured detection classes", 27, WHITE, True)
    text(d, (65, 271), "person  •  helmet  •  safety_vest  •  backpack  •  knife  •  pistol", 23, MUTED)
    text(d, (65, 371), "Confidence threshold: 0.25", 25, WHITE)
    text(d, (65, 415), "Selected validation-set examples; provisional labels", 25, WHITE)
    d.rounded_rectangle((61, 520, 1219, 641), radius=16, fill="#273044")
    text(d, (83, 543), "Independent CCTV / robbery detection remains unreliable.", 26, "#ffcf75", True)
    text(d, (83, 584), "These stills do not represent continuous video inference.", 22, MUTED)
    return im


def main() -> None:
    check_checkpoint()
    rows = {r["id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8-sig"))}
    if len(ORDER) != 7 or len(set(ORDER)) != 7:
        raise RuntimeError("Expected seven distinct Stage 5E examples")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slides").mkdir(exist_ok=True)
    model = YOLO(str(CHECKPOINT))
    if tuple(model.names[i] for i in range(6)) != NAMES:
        raise RuntimeError("Checkpoint class map changed")
    device = 0 if torch.cuda.is_available() else "cpu"
    slides = [title_slide()]
    evidence = []
    for number, ident in enumerate(ORDER, 1):
        row = rows[ident]
        if "stage3c_v1\\images\\val" not in row["image"].replace("/", "\\"):
            raise RuntimeError(f"Not a validation image: {ident}")
        found = model_boxes(model, ROOT / row["image"], device)
        saved = list(zip(row["prediction_classes"].split(";"), row["prediction_scores"].split(";")))
        current = [(b["class_name"], f"{b['confidence']:.3f}") for b in found]
        if current != saved:
            raise RuntimeError(f"Prediction differs from Stage 5E audit: {ident}: {current} != {saved}")
        slides.append(slide(row, found, number))
        evidence.append({"id": ident, "source": row["image"], "stage5e_comparison": row["comparison"],
                         "predictions": found})
    slides.append(closing_slide())
    for i, im in enumerate(slides):
        im.save(OUT / "slides" / f"slide_{i:02d}.png")
    storyboard = Image.new("RGB", (960, 540 * 3), "white")
    for i, im in enumerate(slides):
        thumb = im.resize((320, 180), Image.Resampling.LANCZOS)
        storyboard.paste(thumb, ((i % 3) * 320, (i // 3) * 180))
    storyboard = storyboard.crop((0, 0, 960, 540))
    storyboard.save(OUT / "storyboard.jpg", quality=90)
    # 4 s opening + 7 x 4 s still examples + 4 s closing = 36 s.
    durations = [4] * len(slides)
    writer = cv2.VideoWriter(str(OUT / "mentor_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    frames = [cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR) for im in slides]
    fade = 9  # 0.375 s crossfade inside each slide duration.
    try:
        for i, frame in enumerate(frames):
            count = durations[i] * FPS
            for j in range(count):
                if i and j < fade:
                    alpha = (j + 1) / fade
                    rendered = cv2.addWeighted(frames[i - 1], 1 - alpha, frame, alpha, 0)
                else:
                    rendered = frame
                writer.write(rendered)
    finally:
        writer.release()
    metadata = {"duration_seconds": sum(durations), "fps": FPS, "frames": sum(durations) * FPS,
                "resolution": [W, H], "checkpoint_sha256": EXPECTED_SHA256,
                "confidence": 0.25, "source": "Stage 3C validation still images",
                "stage5e_prediction_match": True, "examples": evidence}
    (OUT / "video_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(OUT / "mentor_demo.mp4"),
                      "duration_seconds": metadata["duration_seconds"],
                      "frames": metadata["frames"]}, indent=2))


if __name__ == "__main__":
    main()
