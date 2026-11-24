# Training and evaluation workflow (future stage; not implemented/run in Stage 1)

1. Approve provenance, annotations, duplicate audit and frozen split manifests.
2. Construct YOLO from configs/yolo11n.yaml (six-class YOLO11n derived from the
   installed upstream YAML). Train with pretrained=False, resume=False and
   amp=False. Never construct the initial trainable model from a .pt file.
   Require a future explicit training entry point with Windows __main__ guard,
   manifest validation and a fresh run directory. Do not launch training now.
3. Select checkpoints and tune thresholds using validation only. Retain exact
   package lock, source revision, architecture hash, seed, configs and logs.
   GPU reproducibility can vary across hardware/drivers despite fixed seeds.
4. Export validation confusion matrices (raw and normalized, including
   background), per-class precision/recall/F1/AP50/AP50:95 and support. Record
   IoU matching, confidence and NMS thresholds; detection TN/accuracy is undefined.
   Keep results.csv, loss/metric curves and PR/F1/precision/recall curves.
5. Export detections with image ID, class ID/name, confidence, box coordinates,
   image dimensions, split, checkpoint hash and inference settings. YOLO text
   exports must use save_txt=True and save_conf=True; additionally retain a
   tabular prediction export and COCO detection JSON for scoring.
6. Freeze checkpoint and all settings, then use the final test set for the final
   report. Evaluate a checkpoint produced by THIS scratch run only. Trained .pt
   files are allowed for later evaluation, never for scratch initialization.

## Mentor-required twelve object-detection parameters

The final evaluation report must explicitly give TP, FP, FN, precision,
recall, F1-score, per-class AP@50, per-class AP@75, mAP@50, mAP@50:95,
mean matched-box IoU, and inference latency plus FPS. Count these as the
mentor's twelve requested parameter categories; per-class AP is a vector,
and latency/FPS is a paired category. State matching IoU, confidence and NMS
thresholds, class support, formulas and hardware/timing protocol. Detection
true negatives are undefined. Any accuracy or specificity result requires a
separately defined, reviewed image-level presence/absence test.

## Supplemental standard COCO evaluation summaries

Use pycocotools COCOeval with bbox evaluation and the standard 12 summaries:

| # | Measurement | IoU | Area | maxDets |
|---|---|---|---|---|
| 1 | AP | 0.50:0.05:0.95 | all | 100 |
| 2 | AP50 | 0.50 | all | 100 |
| 3 | AP75 | 0.75 | all | 100 |
| 4 | AP small | 0.50:0.05:0.95 | small | 100 |
| 5 | AP medium | 0.50:0.05:0.95 | medium | 100 |
| 6 | AP large | 0.50:0.05:0.95 | large | 100 |
| 7 | AR@1 | 0.50:0.05:0.95 | all | 1 |
| 8 | AR@10 | 0.50:0.05:0.95 | all | 10 |
| 9 | AR@100 | 0.50:0.05:0.95 | all | 100 |
| 10 | AR small | 0.50:0.05:0.95 | small | 100 |
| 11 | AR medium | 0.50:0.05:0.95 | medium | 100 |
| 12 | AR large | 0.50:0.05:0.95 | large | 100 |

Use original-image pixel areas: small <32^2, medium 32^2 to 96^2, large >96^2
(COCO evaluator supplies precise boundary handling). Preserve a documented
mapping from fixed YOLO IDs 0..5 to COCO category IDs 1..6. Export low-confidence
detections for AP/AR (record cutoff), keeping maxDets and NMS settings consistent.
Report unsupported class/area slices as unavailable, never as invented zeros.
Supplement these 12 with class-wise results and threshold-specific P/R/F1.

No evaluation values exist at Stage 1. Synthetic verification tensors are only
CUDA/architecture health checks and must never be presented as model accuracy.
