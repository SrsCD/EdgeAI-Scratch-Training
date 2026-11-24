# Stage 4B — experimental six-class test evaluation

**Measured against provisional annotations; not independently verified real-world accuracy.** The 12-epoch scratch run was interrupted at the user’s four-hour limit. No training was resumed or restarted for this evaluation.

## Checkpoint and test protocol

The locally saved `best.pt` and `last.pt` both loaded and have SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`. Checkpoint metadata records zero-based epoch 11, corresponding to **epoch 12** in the preserved [training CSV](../runs/scratch_yolo11n_stage3c_experimental_b16/results.csv). That epoch recorded validation precision **0.54031**, recall **0.39859**, mAP@50 **0.45081**, and mAP@50–95 **0.29567**. The best checkpoint was selected with validation, not test, data. [Training report](stage4a_scratch_training_report.md) records the interruption, config, GPU and hashes.

The [Stage 3C ledger](../data/experimental/stage3c_v1/ledger.csv) SHA-256 is `aa8ea6a9888101587554a96ead9270dcdfa160d27e074c55a6076cfdd369d0bb`. The existing provisional test split contains **1,473 images**; its group assignment was unchanged. The 10 manual-review records are all **Hold**, never approvals. Inference used image size 640, batch one, a 0.001 detection export floor and NMS IoU 0.7; the operating point below is confidence **0.25** with classwise one-to-one matching IoU **0.50**. The exporter saved **198,805** post-NMS detections. The test was evaluated once from those saved predictions. Test scores were not used to train, tune thresholds or choose the checkpoint.

## Actual class distribution

| Class | Train images | Train boxes | Test images | Test boxes | Validation images | Validation boxes |
|---|---:|---:|---:|---:|---:|---:|
| person | 318 | 649 | 80 | 156 | 44 | 87 |
| helmet | 979 | 1555 | 278 | 448 | 129 | 231 |
| safety_vest | 988 | 1571 | 302 | 483 | 174 | 300 |
| backpack | 673 | 702 | 198 | 215 | 91 | 93 |
| knife | 749 | 749 | 149 | 149 | 82 | 82 |
| pistol | 2078 | 2432 | 557 | 623 | 324 | 364 |

## Test metrics at the stated operating point

| Class | TP | FP | FN | Precision | Recall | F1 | AP@50 | AP@75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 6 | 11 | 150 | 0.3529 | 0.0385 | 0.0694 | 0.0625 | 0.0178 |
| helmet | 153 | 85 | 295 | 0.6429 | 0.3415 | 0.4461 | 0.4260 | 0.2188 |
| safety_vest | 344 | 237 | 139 | 0.5921 | 0.7122 | 0.6466 | 0.7042 | 0.5449 |
| backpack | 186 | 127 | 29 | 0.5942 | 0.8651 | 0.7045 | 0.8321 | 0.7996 |
| knife | 11 | 10 | 138 | 0.5238 | 0.0738 | 0.1294 | 0.1081 | 0.0247 |
| pistol | 319 | 230 | 304 | 0.5811 | 0.5120 | 0.5444 | 0.5235 | 0.3541 |

**Overall:** TP **1019**, FP **700**, FN **1055**; precision **0.5928**, recall **0.4913**, F1 **0.5373**. mAP@50 **0.4427**, mAP@50–95 **0.2903**; mean matched-box IoU **0.8275**. Mean model-only inference latency **9.156 ms/image**, median **7.997 ms**, throughput **109.21 FPS** over 1,473 images. Timing excludes image loading, preprocessing and NMS, so it is not end-to-end CCTV FPS.

TP is a confidence-filtered prediction matched once to a same-class provisional box at IoU ≥0.50; unmatched predictions are FP and unmatched boxes are FN. Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is 2PR/(P+R). AP@50 and AP@75 are 101-point interpolated precision–recall areas at IoU 0.50 and 0.75 using the exported detections. mAP@50 averages six class AP@50 values; mAP@50–95 averages class AP over IoU 0.50:0.05:0.95. Mean matched IoU averages the one-to-one true-positive overlaps at confidence 0.25. FPS is 1,000 times image count divided by summed synchronized model-reported inference milliseconds.

## Artifacts

- [Raw confusion matrix](../reports/evaluation/stage3c_experimental_test/confusion_raw.csv) and [normalized confusion matrix](../reports/evaluation/stage3c_experimental_test/confusion_normalized.csv) (rows are provisional ground truth, columns predictions; background row/column represent FP/FN).
- [Per-class metrics](../reports/evaluation/stage3c_experimental_test/per_class_metrics.csv), [per-detection confidence scores](../reports/evaluation/stage3c_experimental_test/per_prediction_confidence.csv), and [full metrics JSON](../reports/evaluation/stage3c_experimental_test/metrics.json).
- [Precision–recall curve](../reports/evaluation/stage3c_experimental_test/precision_recall.png), [F1/confidence](../reports/evaluation/stage3c_experimental_test/f1_confidence.png), [precision/confidence](../reports/evaluation/stage3c_experimental_test/precision_confidence.png), and [recall/confidence](../reports/evaluation/stage3c_experimental_test/recall_confidence.png).
- [Training and validation losses](../reports/evaluation/stage3c_experimental_test/training_validation_losses.png), [validation metric curves](../reports/evaluation/stage3c_experimental_test/validation_metric_curves.png), and [annotated predictions](../reports/evaluation/stage3c_experimental_test/prediction_examples).
- [Inference inputs and checkpoint provenance](../reports/evaluation_inputs/stage3c) and [Stage 4A console log](../reports/stage4a_full_b16_console.log).

## Interpretation and limits

Person and knife recall are low at the fixed 0.25 confidence threshold; these values should be interpreted alongside their AP and PR curves. The image labels can omit co-occurring targets, and the source-group components are provisional, so some scored FP/FN may reflect annotation omissions or leakage. Pistol suitability and image rights are also unresolved. These test metrics measure agreement with provisional labels, not independently verified real-world accuracy. The 10 Hold samples do not validate the remaining images. Image-level accuracy and specificity are omitted because no separately reviewed positive/negative image-level protocol exists. No external dataset, pretrained checkpoint, new training run or CCTV video demo was used.
