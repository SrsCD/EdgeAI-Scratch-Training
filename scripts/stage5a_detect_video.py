"""Offline CCTV-style inference with the project's 12-epoch scratch checkpoint."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageSequence
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt"
PROVENANCE = CHECKPOINT.parent.parent / "stage4a_provenance.json"
EXPECTED_SHA256 = {
    "1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24",  # local original
    "ef582bf60d27ba640ff9a914de2b7e01fa203fb403e66476048a135e95e58276",  # public copy
}
NAMES = ("person", "helmet", "safety_vest", "backpack", "knife", "pistol")
COLORS = ((40, 200, 80), (40, 190, 240), (20, 130, 255),
          (230, 160, 30), (170, 50, 225), (210, 70, 110))  # BGR
VIDEO_EXT = {".mp4", ".avi", ".mov"}


def file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checkpoint() -> dict:
    if not CHECKPOINT.is_file() or file_hash(CHECKPOINT) not in EXPECTED_SHA256:
        raise RuntimeError("The exact Stage 4A scratch best.pt is missing or changed")
    info = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if (info.get("initialization") != "architecture YAML random weights"
            or info.get("pretrained") is not False or info.get("resume") is not False
            or info.get("mode") != "full" or info.get("source") != "data/experimental/stage3c_v1"):
        raise RuntimeError("Scratch checkpoint provenance does not match Stage 4A")
    return info


def frame_source(path: Path):
    """Yield (zero-based index, timestamp seconds, BGR image); return FPS and count hint."""
    if path.suffix.lower() == ".gif":
        with Image.open(path) as im:
            durations = []
            for frame in ImageSequence.Iterator(im):
                durations.append(max(1, int(frame.info.get("duration", im.info.get("duration", 100)))))
        if not durations:
            raise ValueError("GIF has no readable frames")
        # GIF delays may vary. MP4 is constant-FPS; preserve frame order and source timestamps.
        fps = min(120.0, max(1.0, 1000.0 / median(durations)))
        def frames():
            timestamp_ms = 0
            with Image.open(path) as im:
                for i, frame in enumerate(ImageSequence.Iterator(im)):
                    rgb = np.asarray(frame.convert("RGB"))
                    yield i, timestamp_ms / 1000.0, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    timestamp_ms += durations[i]
        return fps, len(durations), frames(), "Pillow GIF decoder; variable delays recorded in timestamps"
    if path.suffix.lower() not in VIDEO_EXT:
        raise ValueError("Input must be MP4, AVI, MOV or GIF")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV cannot open this video; check file and installed codec")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video FPS is unavailable or invalid")
    def frames():
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise ValueError(f"Empty decoded frame at index {i}")
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = timestamp_ms / 1000.0 if np.isfinite(timestamp_ms) and timestamp_ms > 0 else i / fps
                yield i, timestamp, frame
                i += 1
        finally:
            cap.release()
    return fps, count_hint, frames(), "OpenCV VideoCapture"


def result_detections(result, width: int, height: int) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cid = int(box.cls[0])
        if cid not in range(6):
            raise ValueError(f"Unexpected model class ID {cid}")
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"class_id": cid, "class_name": NAMES[cid], "confidence": confidence,
                           "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return detections


def annotate(frame: np.ndarray, detections: list[dict], model_fps: float, process_fps: float) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(d[k])) for k in ("x1", "y1", "x2", "y2"))
        color = COLORS[d["class_id"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, top), (min(out.shape[1]-1, x1+tw+6), min(out.shape[0]-1, top+th+7)), color, -1)
        cv2.putText(out, label, (x1+3, min(out.shape[0]-1, top+th+2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    bottom = out.shape[0] - 1
    top = max(0, bottom - 43)
    cv2.rectangle(out, (0, top), (min(out.shape[1]-1, 285), bottom), (15, 15, 15), -1)
    cv2.putText(out, f"Model-only: {model_fps:.1f} FPS", (6, top+17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"End-to-end: {process_fps:.1f} FPS", (6, top+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pad_for_mp4(frame: np.ndarray) -> np.ndarray:
    """Add black pixels only at the right/bottom so mp4v keeps odd source edges."""
    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def process_video(path: Path, out_dir: Path, model, confidence: float, nms_iou: float,
                  device: int | str, checkpoint_info: dict, checkpoint_hash: str) -> dict:
    fps, count_hint, frames, decoder = frame_source(path)
    out_dir.mkdir(parents=True, exist_ok=False)
    writer = None
    detections = []
    timing = []
    counts = Counter()
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    model_ms_sum = 0.0
    width = height = encoded_width = encoded_height = 0
    status = "failed"
    error = None
    try:
        for index, timestamp, frame in frames:
            frame_start = perf_counter()
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                raise ValueError(f"Unreadable frame {index}")
            if writer is None:
                height, width = frame.shape[:2]
                encoded_width = width + width % 2
                encoded_height = height + height % 2
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (encoded_width, encoded_height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV cannot encode MP4 with installed codecs")
            elif frame.shape[:2] != (height, width):
                raise ValueError(f"Frame {index} changed dimensions")
            result = model.predict(source=frame, imgsz=640, device=device, conf=confidence,
                                   iou=nms_iou, verbose=False)[0]
            if device == 0:
                torch.cuda.synchronize(0)
            model_ms = float(result.speed["inference"])
            if not np.isfinite(model_ms) or model_ms <= 0:
                raise RuntimeError(f"Invalid inference timing at frame {index}")
            found = result_detections(result, width, height)
            model_ms_sum += model_ms
            for d in found:
                counts[d["class_name"]] += 1
                detections.append({"frame_number": index, "timestamp_seconds": f"{timestamp:.6f}",
                                   **{k: d[k] for k in ("class_id", "class_name")},
                                   "confidence": f"{d['confidence']:.8f}",
                                   **{k: f"{d[k]:.4f}" for k in ("x1", "y1", "x2", "y2")}})
            n = index + 1
            current_model_fps = 1000.0 * n / model_ms_sum
            current_process_fps = n / max(1e-9, perf_counter() - started)
            writer.write(pad_for_mp4(annotate(frame, found, current_model_fps, current_process_fps)))
            timing.append({"frame_number": index, "timestamp_seconds": f"{timestamp:.6f}",
                           "model_inference_ms": model_ms,
                           "frame_processing_ms": (perf_counter() - frame_start) * 1000.0,
                           "detections": len(found)})
        if not timing:
            raise ValueError("No readable frames in input")
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer is not None:
            writer.release()
        duration = perf_counter() - started
        with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["frame_number", "timestamp_seconds", "class_id", "class_name",
                      "confidence", "x1", "y1", "x2", "y2"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(detections)
        with (out_dir / "inference_timing.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["frame_number", "timestamp_seconds", "model_inference_ms", "frame_processing_ms", "detections"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(timing)
        with (out_dir / "class_count_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["class_id", "class_name", "detections"])
            for cid, name in enumerate(NAMES): w.writerow([cid, name, counts[name]])
        processed = len(timing)
        model_total_ms = model_ms_sum
        warnings = []
        if count_hint > 0 and processed < count_hint:
            warnings.append(f"Decoded {processed} of {count_hint} reported frames; input may be truncated")
        if path.suffix.lower() == ".gif":
            warnings.append("GIF may have variable frame delays; output MP4 uses constant nominal FPS, while CSV retains source timestamps")
        meta = {"status": status, "error": error, "experimental_provisional_model": True,
                "input_path": str(path.resolve()), "input_sha256": file_hash(path),
                "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash,
                "checkpoint_provenance": str(PROVENANCE),
                "checkpoint_initialization": checkpoint_info["initialization"],
                "fixed_classes": list(NAMES), "device": str(device), "confidence_threshold": confidence,
                "nms_iou_threshold": nms_iou, "decoder": decoder,
                "input_frame_count_hint": count_hint, "processed_frames": processed,
                "input_fps": fps, "output_fps": fps, "width": width, "height": height,
                "encoded_width": encoded_width, "encoded_height": encoded_height,
                "padding_right": encoded_width - width, "padding_bottom": encoded_height - height,
                "padding_policy": "black pixels added on right and bottom after detection and annotation",
                "total_processing_seconds": duration,
                "model_inference_total_ms": model_total_ms,
                "model_inference_mean_ms": model_total_ms / processed if processed else None,
                "model_only_fps": 1000.0 * processed / model_total_ms if model_total_ms else None,
                "end_to_end_fps": processed / duration if duration else None,
                "detections_per_class": {name: counts[name] for name in NAMES},
                "detection_count": len(detections),
                "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "warnings": warnings,
                "notes": "Confidence is a model score, not a calibrated probability. No intruder classification. FPS excludes MP4 playback."}
        (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main():
    p = argparse.ArgumentParser(description="Offline CCTV video inference using the fixed Stage 4A scratch best.pt")
    p.add_argument("input", type=Path, help="Local MP4, AVI, MOV or GIF")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--output-root", type=Path, default=ROOT / "runs/stage5a")
    p.add_argument("--name", help="Optional output subdirectory name")
    args = p.parse_args()
    if not args.input.is_file():
        p.error("Input video does not exist")
    if not (0 < args.confidence < 1 and 0 < args.nms_iou < 1):
        p.error("confidence and NMS IoU must be strictly between 0 and 1")
    if args.input.suffix.lower() not in VIDEO_EXT | {".gif"}:
        p.error("Input must be MP4, AVI, MOV or GIF")
    info = check_checkpoint()
    checkpoint_hash = file_hash(CHECKPOINT)
    from ultralytics import YOLO
    model = YOLO(str(CHECKPOINT))  # Local scratch checkpoint for inference only.
    if tuple(model.names[i] for i in range(6)) != NAMES:
        raise RuntimeError("Checkpoint class order differs from fixed six-class contract")
    device = 0 if torch.cuda.is_available() and "3050" in torch.cuda.get_device_name(0) else "cpu"
    name = args.name or f"{args.input.stem}_{file_hash(args.input)[:8]}"
    if not name or name in {".", ".."} or any(c in name for c in "\\/:*?\"<>|"):
        p.error("Unsafe output name")
    out_dir = args.output_root / name
    if out_dir.exists():
        p.error(f"Output directory already exists: {out_dir}")
    try:
        meta = process_video(args.input, out_dir, model, args.confidence, args.nms_iou,
                             device, info, checkpoint_hash)
    except (ValueError, RuntimeError, cv2.error) as exc:
        p.exit(2, f"Video processing failed: {exc}\n")
    print(json.dumps({"output": str(out_dir), "processed_frames": meta["processed_frames"],
                      "detections": meta["detection_count"], "model_only_fps": meta["model_only_fps"],
                      "end_to_end_fps": meta["end_to_end_fps"]}, indent=2))


if __name__ == "__main__":
    main()
