# Stage 5A — offline CCTV video detection

**Complete, 23 November 2026.** [The Stage 5A script](../scripts/stage5a_detect_video.py) performs local video inference with only this project's 12-epoch scratch-trained [best.pt](../runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt). Its required SHA-256 is `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`; startup refuses a missing or changed checkpoint and checks the [Stage 4A provenance](../runs/scratch_yolo11n_stage3c_experimental_b16/stage4a_provenance.json) for architecture-YAML random initialization, `pretrained=false` and `resume=false`. No training, Stage 4B test evaluation, dataset download or Model A change was performed.

## Run a user-provided video on Windows

Open PowerShell in the cloned project directory. Choose a local MP4, AVI or MOV file (or GIF) and run:

```powershell
& .venv\Scripts\python.exe scripts\stage5a_detect_video.py "C:\path\to\your_cctv_video.mp4" --confidence 0.25 --nms-iou 0.70 --name mentor_demo
```

Use a new `--name` for each run; existing output directories are never overwritten. Omit `--name` to derive a name from the file stem and input hash. The script uses CUDA device 0 when the RTX 3050 is available, otherwise CPU. It accepts only the fixed classes **0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol**. A score such as `0.58` is the model's confidence score, **not a calibrated probability**. A detected person is never automatically labeled an intruder.

Outputs appear in `runs\stage5a\mentor_demo\`:

| File | Contents |
|---|---|
| `annotated.mp4` | Frames in source order, at source video FPS, with boxes, fixed-class names, confidence scores, model-only FPS and end-to-end FPS |
| `detections.csv` | Zero-based frame number, video timestamp in seconds, class ID/name, raw model confidence, pixel `x1,y1,x2,y2` |
| `inference_timing.csv` | Per-frame model inference and total frame processing milliseconds, detection count |
| `run_metadata.json` | Input/checkpoint hashes, device, thresholds, frame/FPS checks, duration, speeds, class counts and warnings |
| `class_count_summary.csv` | Detection totals for all six classes, including zeroes |

Open the result with `Invoke-Item "runs\stage5a\mentor_demo\annotated.mp4"`. Inspect detections with `Import-Csv "runs\stage5a\mentor_demo\detections.csv" | Select-Object -First 20`. For a mentor demonstration, show the input alongside the annotated MP4, then the detection CSV and metadata. Explain that the overlay's **model-only FPS** comes from summed Ultralytics inference timings; **end-to-end FPS** includes decoding, preprocessing, inference, box drawing and MP4 writing. Playback FPS is the source video's rate and is a separate quantity.

## Tested formats and fixture results

The deterministic [synthetic fixture generator](../scripts/stage5a_make_fixture.py) made 12-frame, 320×240, 10-FPS geometric videos. OpenCV decoded the generated **MP4, AVI and MOV** files as 12 frames at 10 FPS; Pillow decoded **GIF** as 12 frames. The actual checkpoint completed the pipeline for all four formats. Each produced an `annotated.mp4` that decoded as 12 frames at 10 FPS, with 14 CSV detections (3 `helmet`, 11 `safety_vest`) on the drawn shapes. The first blank frame had zero detections at confidence 0.25 **in this fixture**. A real empty room may produce false positives. The generated shapes are not ground-truth helmets or vests and their detections are not an accuracy measure.

The final [verified MP4 fixture output](../runs/stage5a/synthetic_cctv_mp4_verified) recorded:

| Measure | Value |
|---|---:|
| Processed input / decoded output frames | 12 / 12 |
| Input / output FPS metadata | 10 / 10 |
| Total processing duration | 1.130 s |
| Mean model-only inference latency | 19.708 ms/frame |
| Model-only FPS | 50.74 |
| End-to-end processing FPS | 10.62 |
| Recorded detections | 14 |

The [metadata](../runs/stage5a/synthetic_cctv_mp4_verified/run_metadata.json), [detection CSV](../runs/stage5a/synthetic_cctv_mp4_verified/detections.csv), [timing CSV](../runs/stage5a/synthetic_cctv_mp4_verified/inference_timing.csv), [class summary](../runs/stage5a/synthetic_cctv_mp4_verified/class_count_summary.csv) and [annotated MP4](../runs/stage5a/synthetic_cctv_mp4_verified/annotated.mp4) preserve those measurements. Startup and tiny-video overhead make this 12-frame speed unsuitable as a sustained CCTV throughput claim. An additional [offline fixture run](../runs/stage5a/synthetic_cctv_offline_check/run_metadata.json) succeeded with Python socket connection methods blocked; no internet access or weight download was needed.

The [synthetic tests](../tests/test_stage5a_video.py) passed **3/3**: one blank frame emitted no CSV detection when a fake model returned none; the next frame emitted two boxes with exact fixed-class names/confidences; a four-frame AVI retained frame count and 10-FPS output; a GIF with variable delays retained frame order and timestamps; and an unreadable MP4 produced a clear error. Fake-model test detections are software fixtures, not measured model results.

## Input and model limits

MP4, AVI and MOV support depends on the installed OpenCV decoder/encoder codecs. A corrupt or unreadable input gets an explicit failure; an early decoder stop relative to the reported frame-count hint is recorded as a warning. The MP4 writer requires the installed `mp4v` codec. GIFs can have variable per-frame delays: the CSV keeps source timestamps, while the output MP4 uses a constant nominal FPS derived from the median GIF delay, so exact variable-delay playback cannot be preserved.

The model was trained from scratch for 12 epochs on **provisional, not fully human-verified** labels. The [Stage 4B report](stage4b_experimental_evaluation_report.md) documents the measured provisional-test detection metrics and their limitations. CCTV footage can differ greatly from the training sources in camera angle, resolution, lighting and motion. Boxes and confidence scores are predictions only; no person identity, intent, intruder status, alert policy or independently verified real-world accuracy is provided.
