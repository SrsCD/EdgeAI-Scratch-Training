"""Live six-class webcam inference with the fixed scratch-trained YOLO11n checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from hashlib import sha256
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from stage5h_events import DEFAULT_DB, EventStore, NAMES as EVENT_NAMES


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt"
EXPECTED_SHA256 = {
    "1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24",  # local original
    "ef582bf60d27ba640ff9a914de2b7e01fa203fb403e66476048a135e95e58276",  # public copy
}
NAMES = ("person", "helmet", "safety_vest", "backpack", "knife", "pistol")
COLORS = ((40, 200, 80), (40, 190, 240), (20, 130, 255),
          (230, 160, 30), (170, 50, 225), (210, 70, 110))  # BGR
TITLE = "Model B — Scratch-Trained YOLO11n"


def verify_checkpoint() -> str:
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Required scratch checkpoint is missing: {CHECKPOINT}")
    digest = sha256()
    with CHECKPOINT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() not in EXPECTED_SHA256:
        raise RuntimeError("Scratch checkpoint hash changed; refusing to use different weights")
    return digest.hexdigest()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow is usually the most reliable Windows webcam backend. Try the
    # OpenCV default backend too, since some cameras only support Media Foundation.
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                return camera
        camera.release()
    raise RuntimeError(
        f"Cannot read webcam index {index}. Check its index, Windows Camera privacy permission, "
        "and whether another app is using the camera."
    )


class SyntheticCapture:
    """One synthetic frame through the normal inference loop; no camera claim."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.used = False

    def read(self):
        if self.used:
            return False, None
        self.used = True
        # A plain gradient tests frame conversion and rendering without
        # pretending that a target object was detected.
        ramp = np.linspace(0, 80, self.width, dtype=np.uint8)
        frame = np.broadcast_to(ramp, (self.height, self.width)).copy()
        return True, cv2.merge((frame, frame, frame))

    def release(self):
        pass


def detections_from_result(result, frame_shape):
    height, width = frame_shape[:2]
    if tuple(result.orig_shape) != (height, width):
        raise RuntimeError("Model coordinates do not match the captured frame dimensions")
    found = []
    if result.boxes is None:
        return found
    for box in result.boxes:
        class_id = int(box.cls.item())
        if not 0 <= class_id < len(NAMES):
            raise RuntimeError(f"Unexpected class ID {class_id} from checkpoint")
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        x1, x2 = sorted((np.clip(x1, 0, width - 1), np.clip(x2, 0, width - 1)))
        y1, y2 = sorted((np.clip(y1, 0, height - 1), np.clip(y2, 0, height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue
        found.append((class_id, float(box.conf.item()),
                      (round(x1), round(y1), round(x2), round(y2))))
    return found


@lru_cache(maxsize=1)
def title_patch():
    """Pillow renders the em dash that OpenCV's Hershey fonts cannot render."""
    patch = Image.new("RGB", (470, 36), (20, 20, 20))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
    except OSError:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    ImageDraw.Draw(patch).text((10, 4), TITLE, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)


def annotate(frame, found, fps: float, confidence: float):
    # Preserve every captured pixel below a separate 70-pixel status band.
    # Ultralytics' xyxy values are in original-frame coordinates, so only the
    # band's vertical offset is added when drawing into the display image.
    height, width = frame.shape[:2]
    band = 70
    output = np.full((height + band, width, 3), (20, 20, 20), dtype=np.uint8)
    output[band:, :] = frame
    patch = title_patch()
    output[:patch.shape[0], :min(width, patch.shape[1])] = patch[:, :min(width, patch.shape[1])]
    cv2.putText(output, f"Live FPS {fps:.1f}  |  Confidence {confidence:.2f}  |  +/- adjust  |  Q quit",
                (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    for class_id, score, (x1, y1, x2, y2) in found:
        color = COLORS[class_id]
        y1 += band
        y2 += band
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{NAMES[class_id]} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(band, y1 - text_height - baseline - 7)
        label_right = min(width - 1, x1 + text_width + 8)
        cv2.rectangle(output, (x1, label_top),
                      (label_right, min(output.shape[0] - 1, label_top + text_height + baseline + 7)), color, -1)
        cv2.putText(output, label, (x1 + 4, min(output.shape[0] - 1, label_top + text_height + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def open_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    if path.suffix.lower() != ".mp4":
        raise ValueError("Recording path must end in .mp4")
    if path.exists():
        raise FileExistsError(f"Recording already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width + width % 2, height + height % 2))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("MP4 codec could not start. Check OpenCV codec support or omit --record.")
    return writer


def run(args) -> None:
    checkpoint_sha256 = verify_checkpoint()
    source = SyntheticCapture(args.width, args.height) if args.self_test else open_camera(
        args.camera_index, args.width, args.height)
    writer = None
    events = None
    window_open = False
    frames = 0
    confidence = args.confidence
    times = deque(maxlen=31)
    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("CUDA GPU unavailable; using CPU. Live FPS may be lower.")
        else:
            print(f"Inference device: {torch.cuda.get_device_name(0)}")
        model = YOLO(str(CHECKPOINT))
        model_names = tuple(model.names[i] for i in range(len(NAMES)))
        if model_names != NAMES or len(model.names) != len(NAMES):
            raise RuntimeError(f"Checkpoint class mapping differs from required six classes: {model.names}")
        if NAMES != EVENT_NAMES:
            raise RuntimeError("Event logger class mapping differs from model")
        if not args.self_test:
            events = EventStore(args.database, checkpoint_id=f"scratch_yolo11n_stage3c_experimental_b16/best.pt:{checkpoint_sha256[:16]}")
            print(f"Local detection history: {events.path}")
        if not args.self_test:
            cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE)
            window_open = True
            print("Camera opened. Press + or - to adjust confidence; Q to quit.")
        while True:
            ok, frame = source.read()
            if not ok or frame is None or frame.size == 0:
                if args.self_test:
                    break
                raise RuntimeError("Webcam stopped returning frames. Reconnect it and restart.")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError("Webcam returned a frame that is not BGR color")
            if frames == 0:
                print(f"Captured frame size: {frame.shape[1]}x{frame.shape[0]}")
            result = model.predict(source=frame, imgsz=640, conf=confidence,
                                   iou=0.70, device=device, verbose=False)[0]
            if device == 0:
                torch.cuda.synchronize(0)
            found = detections_from_result(result, frame.shape)
            times.append(perf_counter())
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            display = annotate(frame, found, fps, confidence)
            if events is not None:
                # Keep the highest-confidence model box for each class on this
                # frame. Cooldown suppresses repeat rows, never display boxes.
                best = {}
                for class_id, score, xyxy in found:
                    if class_id not in best or score > best[class_id][0]:
                        best[class_id] = (score, xyxy)
                for class_id, (score, xyxy) in best.items():
                    name = NAMES[class_id]
                    notify = name in args.notify_class
                    event_id = events.add(class_id=class_id, confidence=score, xyxy=xyxy,
                                          source_id=f"webcam:{args.camera_index}",
                                          cooldown=args.cooldown_by_class.get(name, args.cooldown_seconds),
                                          notification_status="console_sent" if notify else "disabled")
                    if event_id is not None:
                        if args.save_evidence:
                            evidence = events.path.parent / "evidence" / f"{event_id}.jpg"
                            evidence.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(evidence), display):
                                raise RuntimeError(f"Could not save evidence image: {evidence}")
                            events.set_evidence(event_id, evidence)
                        if notify:
                            print(f"\nUNVERIFIED DETECTION NOTICE: {name} {score:.2f}; event {event_id}")
            counts = Counter(NAMES[class_id] for class_id, _, _ in found)
            summary = "  ".join(f"{name}:{counts[name]}" for name in NAMES)
            print(f"\rFrame {frames + 1}  FPS {fps:.1f}  conf {confidence:.2f}  {summary}   ",
                  end="", flush=True)
            if writer is None and args.record is not None:
                writer = open_writer(args.record, display.shape[1], display.shape[0], args.record_fps)
                print(f"\nRecording live annotated frames to {args.record}")
            if writer is not None:
                # Most MP4 codecs require even dimensions; padding preserves all
                # displayed pixels and leaves box coordinates unchanged.
                writer.write(cv2.copyMakeBorder(display, 0, display.shape[0] % 2,
                                                0, display.shape[1] % 2,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0)))
            frames += 1
            if args.self_test:
                break
            cv2.imshow(TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("+"), ord("="), ord("]")):
                confidence = min(0.95, round(confidence + 0.05, 2))
            elif key in (ord("-"), ord("_"), ord("[")):
                confidence = max(0.05, round(confidence - 0.05, 2))
        if not frames:
            raise RuntimeError("No frame was processed")
        print(f"\nStopped after {frames} frame(s).")
        if args.self_test:
            print("Synthetic startup/inference/shutdown passed; physical webcam was not tested by this mode.")
    finally:
        if writer is not None:
            writer.release()
        if events is not None:
            events.close()
        source.release()
        if window_open:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height (default: 720)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Initial confidence (default: 0.25)")
    parser.add_argument("--record", type=Path, help="Optional output MP4 path for actual live annotated frames")
    parser.add_argument("--record-fps", type=float, default=20.0,
                        help="Constant playback FPS for MP4 recording (default: 20)")
    parser.add_argument("--self-test", action="store_true",
                        help="Process one synthetic frame through this checkpoint without opening a camera/window")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB,
                        help="Persistent local SQLite path (default: runs/stage5h/events.sqlite3)")
    parser.add_argument("--cooldown-seconds", type=float, default=10.0,
                        help="Seconds between logged events of each class/source (default: 10)")
    parser.add_argument("--cooldown-class", action="append", default=[], metavar="CLASS=SECONDS",
                        help="Override cooldown for one class; repeat as needed")
    parser.add_argument("--save-evidence", action="store_true", help="Save annotated JPEG for logged events")
    parser.add_argument("--notify-class", action="append", default=[], choices=NAMES,
                        help="Print an unverified detection notice for this class; repeat as needed")
    args = parser.parse_args()
    if args.camera_index < 0 or args.width < 64 or args.height < 64:
        parser.error("camera-index must be nonnegative and width/height must be at least 64")
    if not (0.0 < args.confidence < 1.0) or not (0.0 < args.record_fps <= 120.0):
        parser.error("confidence must be between 0 and 1; record-fps must be in (0, 120]")
    if not np.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("cooldown-seconds must be finite and nonnegative")
    args.cooldown_by_class = {}
    for item in args.cooldown_class:
        try:
            name, raw_seconds = item.split("=", 1)
            seconds = float(raw_seconds)
            if name not in NAMES or not np.isfinite(seconds) or seconds < 0:
                raise ValueError
        except ValueError:
            parser.error(f"Invalid --cooldown-class {item!r}; use CLASS=nonnegative seconds")
        args.cooldown_by_class[name] = seconds
    try:
        run(args)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; camera and recording closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
