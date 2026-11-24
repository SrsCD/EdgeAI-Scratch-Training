# Stage 4A — experimental YOLO11n scratch training

**INTERRUPTED AFTER 12 EPOCHS. EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED.** Training labels and validation labels are provisional. No final test evaluation was run.

## Preflight and split

The Stage 3C preflight passed before both smoke and full training. The split stayed 5,417 train / 1,473 test / 790 validation, with six fixed classes: 0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol. The 10 manual-review decisions were all **Hold**, unresolved, and none were approvals. The approved-dataset preflight remains separate and blocked. Training and checkpoint selection used train and validation only; the test split was not loaded for Stage 4A.

Dataset ledger SHA-256: `aa8ea6a9888101587554a96ead9270dcdfa160d27e074c55a6076cfdd369d0bb`. Split statistics SHA-256: `3f2566702166694444c218198eee66826a1198a1d644c211cb6acb3435d9b71b`. Architecture YAML SHA-256: `cfef81e558b212e6f4061f20419fab431066707720bf1e14286180f38c696c61`. Data YAML SHA-256: `a5ae3f01ebf7eb3ae35f3914f4cd3082748cf700a8c52b87974e66b514068bd0`. Initial train config SHA-256: `b7b271ba9d857eeef1683aaeafa40fa33a9ac5fd0102f0561ab00d1860beec64`.

## Scratch initialization and run settings

YOLO11n was constructed from `configs/yolo11n.yaml` with an empty Ultralytics checkpoint dictionary and no checkpoint path or model `.pt` path. `pretrained=False`, `resume=False`, `device=0`, `seed=42`, `imgsz=640`, `workers=0`, `amp=False`. The full run used **batch 16**, maximum **150 epochs**, and validation early stopping patience **30**. The initial batch-2 run completed one epoch and was interrupted for measured throughput tuning; its logs and checkpoint remain under `runs/scratch_yolo11n_stage3c_experimental`. A separate batch-16 smoke test passed with finite loss and 3.9 GB peak allocated VRAM. The full batch-16 run started afresh from YAML, not from either earlier checkpoint.

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6441926656 bytes). PyTorch 2.9.1+cu128; Python 3.13.9; Ultralytics 8.3.233. Started UTC 2026-11-23T14:35:49.640531+00:00. Outcome: interrupted_by_user_after_12_completed_epochs. Last logged cumulative training time: 2078.32 seconds. Peak allocated VRAM during the batch-16 smoke test: 3,896,231,936 bytes; the full run displayed about 4.5 GB in its training log.

## Actual training and validation results

Completed epoch rows: **12**. Latest epoch: **12**. Latest train losses (box / class / DFL): **1.44239 / 1.98858 / 1.75248**. Latest validation losses: **1.50549 / 1.98212 / 1.93131**.

Latest validation precision **0.54031**, recall **0.39859**, mAP@50 **0.45081**, mAP@50–95 **0.29567**. Highest CSV mAP@50–95 was **0.29567** at epoch **12**. Ultralytics writes `best.pt` using its own validation fitness rule; the maximum CSV mAP epoch is reported as a separate observation.

Epoch metrics CSV: [results.csv](../runs/scratch_yolo11n_stage3c_experimental_b16/results.csv). Training arguments: [args.yaml](../runs/scratch_yolo11n_stage3c_experimental_b16/args.yaml). Console log: [stage4a_full_b16_console.log](../reports/stage4a_full_b16_console.log). Interrupted-run training and validation loss curves: [training_validation_losses.png](../reports/evaluation/stage3c_experimental_test/training_validation_losses.png).

## Checkpoints and next stage

Best checkpoint: `runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt` (SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`).
Last checkpoint: `runs/scratch_yolo11n_stage3c_experimental_b16/weights/last.pt` (local only; SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`).

Stage 4B should use the completed best checkpoint once, on the untouched provisional test split. It must compute TP, FP, FN, precision, recall, F1, AP@50, AP@75, mAP@50, mAP@50–95, matched-box IoU, and inference latency/FPS, plus raw/normalized confusion matrices, per-class tables, confidence CSV and PR curves. If image-level accuracy or specificity is needed, define a separate reviewed positive/negative image-level protocol first.

## Limitations

The 7,680-image experimental dataset is mechanically checked but not fully human verified. Missing co-occurring objects, unsuitable pistol depictions, source rights, and provisional group independence remain unresolved. Validation metrics measure agreement with provisional annotations. Future test metrics also cannot establish independently verified real-world accuracy. No datasets were downloaded, Model A was not changed, and CCTV video detection was not implemented.
