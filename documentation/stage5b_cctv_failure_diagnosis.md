# Stage 5B — CCTV detection failure diagnosis

**24 November 2026. Diagnosis only.** No training, dataset download, full test evaluation or checkpoint change was performed. This audit used the existing [22-frame CCTV GIF](../data/demo_videos/cctv_test.gif), the [Stage 5A run at confidence 0.10](../runs/stage5a/mentor_demo_conf010/run_metadata.json), and the exact 12-epoch scratch [best.pt](../runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt) (SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`). Results remain **experimental measurements against provisional labels**, not independently verified real-world accuracy.

## Observed failure

The input is a 250×239 GIF at nominal 2 FPS. The existing Stage 5A run processed all 22 frames at confidence 0.10 and NMS IoU 0.70, reporting **0 person, 28 pistol and 2 safety_vest** detections. At confidence 0.25, the prior run reported no detections. The [frame comparison sheet](../reports/stage5b/cctv_comparison.jpg) shows visible people and a handgun-shaped object; the reported pistol boxes lie on the right wall rather than on that object. For example, frame 5's saved pistol box is `(212.3924, 5.0058)–(250.0000, 124.5847)` at confidence `0.10327137`. The right boundary `x2=250` is the legal edge coordinate for a 250-pixel-wide image, not evidence of an axis swap.

## Pipeline and coordinate audit

The [Stage 5A code](../scripts/stage5a_detect_video.py) decodes GIF frames with Pillow, converts RGB to BGR, passes the unscaled 250×239 frame directly to `YOLO.predict(imgsz=640)`, and reads `result.boxes.xyxy` in original-frame **pixel** coordinates. It does not manually resize, letterbox, invert a letterbox transform, or mix normalized and pixel coordinates. Ultralytics' result for frame 5 reports `orig_shape=(239, 250)`. Its normalized pistol box multiplied by `(width,height,width,height)` gives `(212.39243, 5.00582, 250, 124.58475)`, agreeing with `xyxy` and the saved CSV. The checkpoint's class map is the fixed order: person, helmet, safety_vest, backpack, knife, pistol.

The read-only [direct comparison script](../scripts/stage5b_compare_frames.py) decoded source frames **5, 7, 11 and 19** and predicted with the same checkpoint, `imgsz=640`, confidence 0.10 and NMS IoU 0.70. Direct class IDs, scores and boxes match the existing CSV in every selected frame; the largest numeric difference is under **0.00005 pixel**, explained by the CSV's four-decimal box formatting. [Direct predictions](../reports/stage5b/direct_predictions_0_10.csv) and the [comparison summary](../reports/stage5b/direct_comparison.json) preserve the evidence. In the saved annotated frame 5, the rendered border near `(212,60)` has the pistol class color, confirming the video drawing follows that same pixel box. The box is wrong **before** CSV export or drawing.

One small output defect exists: the MP4 encoder wrote the odd-height 239-pixel GIF as a **250×238** video, cropping the bottom row. This can be fixed later by padding to even dimensions and recording original versus output dimensions. It cannot explain misplaced boxes in the upper half, the missed people, or the source-frame CSV coordinates. No Stage 5A artifact was overwritten during this diagnosis.

## Low-confidence person check

Direct prediction at confidence **0.001** found many low-score person proposals on the four frames, but none reached 0.10. The [person proposals CSV](../reports/stage5b/low_confidence_person_boxes.csv) is diagnostic only:

| Frame | Direct/CSV detections at 0.10 | Person proposals at 0.001 | Highest person score |
|---:|---:|---:|---:|
| 5 | 1 / 1 | 68 | 0.01713 |
| 7 | 2 / 2 | 73 | 0.01758 |
| 11 | 2 / 2 | 100 | 0.02426 |
| 19 | 5 / 5 | 74 | 0.02871 |

Several top low-score boxes cover wall trim, overlays, furniture or fragments of people rather than full people. Lowering a production threshold toward 0.02 would admit numerous weak proposals and does not repair the localization problem. These scores are uncalibrated model confidences, not probabilities.

## Likely causes, in priority order

1. **Weak person model and class imbalance.** In the unchanged [Stage 4B per-class results](../reports/evaluation/stage3c_experimental_test/per_class_metrics.csv), person had **6 TP, 150 FN**, recall **0.0385** at confidence 0.25, and AP@50 **0.0625**. The Stage 3C training split contained only **318 person-positive images / 649 person boxes**, versus **2,078 pistol-positive images / 2,432 pistol boxes**. Missing people in this CCTV clip are consistent with the already measured weakness.
2. **Incomplete six-class annotations.** The [Stage 3C ledger](../data/experimental/stage3c_v1/ledger.csv) flags **7,236 of 7,680 included images** as coming from sources whose taxonomy does not label `person`. This is a missing-label *risk*, not proof that every such image contains an unlabeled person. When visible people in those images were left unlabeled, training could teach the detector to suppress person predictions. Other co-occurring targets may likewise be omitted.
3. **Domain mismatch and pistol false positives.** This is low-resolution, compressed, indoor surveillance-style footage with dark, partly occluded people and an unusually small handheld object. Much of the experimental source data consists of construction, product or weapon images. The Stage 4B pistol result (recall **0.5120**, AP@50 **0.5235** against provisional test labels) does not guarantee localization on this footage. On the selected CCTV frames, direct model boxes target the wall and door area. The two likely false vest boxes also reflect this transfer problem.
4. **Training stopped early.** The scratch run ended after **12 completed epochs**. Validation mAP@50–95 rose to **0.29567** at epoch 12. More optimization may improve the original data distribution, but this trend does not show that extra epochs alone will solve omitted labels or CCTV domain shift.

**Conclusion:** the main failure is a combination of weak scratch-trained person detection, incomplete provisional labels and CCTV domain mismatch. The Stage 5A coordinate/class pipeline is not the cause of the observed false pistol locations or missed people. The one-pixel MP4 crop is a separate output issue.

## Prioritized next action and RTX 3050 plan

1. **First, create a small reviewed diagnostic set without training.** Fully annotate target objects on representative frames 5, 7, 11 and 19: every visible person, the handgun-shaped object only if it meets the fixed `pistol` definition, and any other fixed-class object. Record the right-wall false-positive regions as reviewed background and preserve the original GIF/session group. The four frames are related views and **cannot** be split across train, validation and test or used alone to claim independent accuracy. This is the minimum correction needed to measure these specific errors; no box coordinates should be invented from this report.
2. **Before any checkpoint continuation, fix the labels used for that continuation.** Choose a focused subset from the existing archives and fully verify all six target classes in every retained image, especially visible people in PPE, knife, backpack and pistol-source images. Remove or quarantine images with unresolved omissions; include real handgun examples and reviewed hard negatives resembling the wall/door false positives. Keep source/video groups together and separate a reviewed validation scene. Correcting only the four GIF frames while continuing on all 5,417 provisional training images would leave the dominant missing-label risk in place. The exact number of source images needed depends on their actual content; a defensible numeric minimum cannot be inferred from metadata alone.
3. **Then consider a bounded continuation from this scratch checkpoint, not pretrained weights.** If a corrected focused set and separate reviewed validation data exist, test **10–20 additional epochs** from `best.pt`, `resume=False`, batch **8** as a conservative 6 GB starting point, image size 640, workers 0, `amp=False`, seed 42 and validation-based stopping. The batch-16 run previously occupied about 5.3 GB total GPU memory; batch 8 leaves more headroom. From the recorded batch-16 pace (~2.9 minutes/epoch), a batch-8 continuation might take roughly **40–100 minutes** for 10–20 epochs; this is an estimate, not a measured batch-8 benchmark. Keep the frozen provisional test results untouched while selecting changes on validation data. Extra training on the unchanged, incomplete dataset is low priority and may reinforce the failure.
4. **For the video pipeline, pad odd dimensions before MP4 encoding** and record both original and encoded dimensions. Recheck coordinates on a few frames after that change. This improves output fidelity but should not be presented as a fix for model detection quality.

No training or full test rerun was performed in Stage 5B. The **next recommended action** is the four-frame, full-six-class annotation audit plus a focused source-label completeness audit; only then decide whether a short checkpoint continuation is justified.
