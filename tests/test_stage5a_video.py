"""Synthetic fixture checks video plumbing; fake detections are not model claims."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage5a_detect_video import process_video, frame_source, pad_for_mp4


class FakeBox:
    def __init__(self, cid, confidence, xyxy):
        self.cls = np.array([cid]); self.conf = np.array([confidence])
        self.xyxy = np.array([xyxy], dtype=float)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes; self.speed = {"inference": 5.0}


class FakeModel:
    def __init__(self): self.calls = 0
    def predict(self, **_):
        i = self.calls; self.calls += 1
        boxes = [] if i == 0 else ([FakeBox(0, .81, (20, 20, 90, 110)),
                                     FakeBox(5, .63, (110, 40, 170, 100))] if i == 1 else [])
        return [FakeResult(boxes)]


class VideoFixtureTests(unittest.TestCase):
    def test_avi_frame_order_fps_empty_and_multiple_detections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "synthetic.avi"
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (192, 128))
            self.assertTrue(writer.isOpened())
            for value in (0, 50, 100, 150):
                writer.write(np.full((128, 192, 3), value, dtype=np.uint8))
            writer.release()
            meta = process_video(src, root / "out", FakeModel(), .25, .7, "cpu",
                                 {"initialization": "architecture YAML random weights"}, "fixture")
            self.assertEqual(meta["processed_frames"], 4)
            self.assertAlmostEqual(meta["output_fps"], 10.0, places=2)
            self.assertEqual(meta["detections_per_class"], {"person": 1, "helmet": 0,
                "safety_vest": 0, "backpack": 0, "knife": 0, "pistol": 1})
            with (root / "out/detections.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["frame_number"] for r in rows}, {"1"})
            self.assertEqual({r["class_name"] for r in rows}, {"person", "pistol"})
            self.assertEqual({r["confidence"] for r in rows}, {"0.81000000", "0.63000000"})
            with (root / "out/inference_timing.csv").open(encoding="utf-8-sig", newline="") as f:
                timing = list(csv.DictReader(f))
            self.assertEqual([int(r["detections"]) for r in timing], [0, 2, 0, 0])
            cap = cv2.VideoCapture(str(root / "out/annotated.mp4"))
            self.assertTrue(cap.isOpened())
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 10.0, places=2)
            frames = 0
            while True:
                ok, _ = cap.read()
                if not ok: break
                frames += 1
            cap.release()
            self.assertEqual(frames, 4)

    def test_gif_timestamps_and_decode(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "variable.gif"
            images = [Image.new("RGB", (48, 32), (v, 0, 0)) for v in (0, 80, 160)]
            images[0].save(src, save_all=True, append_images=images[1:], duration=[100, 200, 100], loop=0)
            fps, hint, frames, _ = frame_source(src)
            records = list(frames)
            self.assertEqual(hint, 3)
            self.assertAlmostEqual(fps, 10.0)
            self.assertEqual([round(t, 3) for _, t, _ in records], [0.0, 0.1, 0.3])
            self.assertEqual([i for i, _, _ in records], [0, 1, 2])

    def test_odd_gif_padding_preserves_source_pixels_and_detection_coordinates(self):
        raw = np.arange(239 * 251 * 3, dtype=np.uint8).reshape(239, 251, 3)
        padded = pad_for_mp4(raw)
        self.assertEqual(padded.shape, (240, 252, 3))
        np.testing.assert_array_equal(padded[:239, :251], raw)
        self.assertTrue(np.all(padded[239, :] == 0))
        self.assertTrue(np.all(padded[:, 251] == 0))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "odd.gif"
            images = [Image.new("RGB", (251, 239), (40, 60, 80)),
                      Image.new("RGB", (251, 239), (80, 100, 120))]
            images[0].save(src, save_all=True, append_images=images[1:], duration=100, loop=0)
            meta = process_video(src, root / "out", FakeModel(), .25, .7, "cpu",
                                 {"initialization": "architecture YAML random weights"}, "fixture")
            self.assertEqual((meta["width"], meta["height"]), (251, 239))
            self.assertEqual((meta["encoded_width"], meta["encoded_height"]), (252, 240))
            self.assertEqual((meta["padding_right"], meta["padding_bottom"]), (1, 1))
            cap = cv2.VideoCapture(str(root / "out/annotated.mp4"))
            ok, decoded = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(decoded.shape[:2], (240, 252))
            with (root / "out/detections.csv").open(encoding="utf-8-sig", newline="") as f:
                detections = list(csv.DictReader(f))
            self.assertEqual([(r["x1"], r["y1"], r["x2"], r["y2"]) for r in detections],
                             [("20.0000", "20.0000", "90.0000", "110.0000"),
                              ("110.0000", "40.0000", "170.0000", "100.0000")])

    def test_unreadable_video_has_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            invalid = Path(td) / "broken.mp4"
            invalid.write_bytes(b"not a video")
            with self.assertRaisesRegex(ValueError, "cannot open"):
                frame_source(invalid)


if __name__ == "__main__": unittest.main()
