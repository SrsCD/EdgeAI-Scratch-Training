# Stage 2D — manual dataset audit

**Date:** 23 November 2026. **Scope:** four ZIPs in `data/raw/manual_imports/`; originals read only. This is a mechanical and representative visual audit, not human approval of all images or a training run. The fixed project IDs remain 0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol.

## Inventory and measured counts

The requested `EL(2).zip` is absent. The folder contains `EL.zip`, whose YAML names the six robbery headwear classes and whose README and image count match the described Dataset D. It was audited as D. Archive metadata, including SHA-256 and verbatim README/YAML content, is saved in [archive_metadata.json](../reports/stage2d_manual/archive_metadata.json). The duplicate `data.yaml` member name in `EL.zip` appears three times with identical text; this is archive packaging noise and should be resolved before using generic ZIP extractors.

| Archive/source | Source splits (train/test/valid) | Images/paired labels | Valid box rows | Valid polygon rows | Empty labels | Corrupt images / invalid label images |
|---|---:|---:|---:|---:|---:|---:|
| `images for EL MAIN.zip` / knife | 802 / 0 / 201 | 1,003 / 1,003 | 1,003 | 0 | 0 | 0 / 0 |
| `Motorcycle Helmet.v1i.yolov8.zip` | 439 / 69 / 121 | 629 / 629 | 1,180 | 0 | 8 | 0 / 0 |
| `guns.zip` / People with | 309 / 0 / 0 | 309 / 309 | 395 | 253 | 1 | 0 / 0 |
| `EL.zip` / robbery headwear | 405 / 10 / 79 | 494 / 494 | 779 | 0 | 0 | 0 / 0 |
| **Total** | **1,955 / 79 / 401** | **2,435 / 2,435** | **3,357** | **253** | **9** | **0 / 0** |

All 2,435 image files decoded, had positive dimensions, and had a paired TXT file. **2,426 labels were nonempty**; no orphan labels were found. Class IDs, finite normalized coordinates, positive widths/heights, in-frame box extents and polygon point extents passed the parser for all nonempty rows. The 9 empty files are **unverified**, not confirmed negative images. ZIP reads performed CRC checking. The detailed split and source-class counts are in [source_statistics.csv](../reports/stage2d_manual/source_statistics.csv); every image's paths, dimensions, source counts, issues, SHA-256, pHash and provisional group are in [per_image_audit_ledger.csv](../reports/stage2d_manual/per_image_audit_ledger.csv).

| Source class ID/name | Source-positive images | Source objects | Taxonomy finding |
|---|---:|---:|---|
| Knife 0 / `2` | 1,003 | 1,003 boxes | Sampled boxes surround real knives; name `2` is a string, while YOLO row ID is 0. Conditional target **4 knife**, never target 2. |
| Motorcycle 0 / With Helmet | 445 | 785 boxes | Boxes focus on helmet or head; sampled helmets are bicycle/motorcycle types, outside industrial hard-hat scope. |
| Motorcycle 1 / Without Helmet | 216 | 395 boxes | Boxes focus on heads; this absence category is neither person nor helmet. |
| People-with 0 / Turkish name | 306 | 312 boxes | Sampled boxes look like game HUD/name-tag areas; no real pistol evidence. |
| People-with 1 / people | 303 | 83 boxes + 253 polygons | Sampled objects are game avatars. Polygons are segmentation rows, not YOLO detection boxes. |
| Robbery 0 / Balaclava | 32 | 32 boxes | No target mapping. |
| Robbery 1 / Hat | 212 | 244 boxes | No target mapping. |
| Robbery 2 / Helmet | 66 | 99 boxes | Broad name; sampled imagery does not establish industrial helmets. Individual review needed. |
| Robbery 3 / Hood | 231 | 319 boxes | No target mapping. |
| Robbery 4 / Mask | 62 | 64 boxes | No target mapping. |
| Robbery 5 / Sunglasses | 21 | 21 boxes | No target mapping. |

Counts are source labels, **not accepted target-class ground truth**. Source-positive images can overlap classes. The complete mapping decision for all 11 source classes is [class_mapping_proposal.csv](../reports/stage2d_manual/class_mapping_proposal.csv).

## Visual findings and usability

The [knife review sheets](../reports/stage2d_manual/knife_source_review_01.jpg) show knives including utility and kitchen-style blades, often held by people. Person boxes are absent because the source has one knife class. Several tiles share very similar presenters/scenes, and some have visible third-party watermarks. The source is a **candidate for manual review and correction**, not ready for merging. Review every image for all six classes; add visible persons and other target boxes, check knife scope and provenance, and exclude unresolved cases. Only then convert source ID 0 to target ID 4.

The [motorcycle review sheets](../reports/stage2d_manual/motorcycle_helmet_review_01.jpg) show helmet/head-sized boxes across cyclists, motorcycle riders and bare-headed people, not whole-person boxes. Bicycle/motorcycle helmets must not become target `helmet`. Many visible people, and some backpacks, lack target-class annotations. Retain only images that can be independently and fully reannotated for qualifying target objects; the source labels provide no direct bulk mapping. The eight empty labels need inspection.

The [People-with review sheets](../reports/stage2d_manual/people_with_review_01.jpg) show Call of Duty first-person shooter screenshots in the deterministic sample; all 309 names start with `Call-of-Duty-`. The box for `Turkish name` appears to target a game HUD/name area, while `people` traces avatars. No real pistol was confirmed. **Reject this source for real-world pistol and person training.** The 253 polygons could mechanically become enclosing rectangles after quality review, but conversion would not solve domain or taxonomy mismatch, so no conversion was performed. The single empty label is not an approved negative.

The [robbery headwear review sheets](../reports/stage2d_manual/robbery_headwear_review_01.jpg) show CCTV-like robbery footage. Boxes surround hats, hoods, masks and some broad `Helmet` examples; visible people are generally unboxed as persons. This source is a reference or full-reannotation candidate, with no automatic target mapping. The `Helmet` label must be assessed per object for industrial hard-hat identity; source name alone is insufficient. Frame series and third-party CCTV/video overlays raise independence and underlying-image-rights questions.

All contact sheets display **unverified source annotations**, with rectangles for box rows and outlines for polygon rows. They are review aids, not proof that every annotation is correct. There are two deterministic sheets per archive; see all `*_review_*.jpg` files in [the audit folder](../reports/stage2d_manual/).

## Duplication and source-group leakage

Across all four archives, SHA-256 found **zero exact duplicate image pairs**. A 64-bit pHash Hamming distance ≤4 produced **1,444 near-duplicate candidate pairs**, touching 573 images: 1,241 knife, 19 motorcycle, 1 game and 183 headwear pairs. **475 candidate pairs cross source-provided split boundaries** (403 knife, 8 motorcycle, 64 headwear). These are similarity signals, not visually adjudicated duplicate counts; the [candidate list](../reports/stage2d_manual/near_duplicate_candidates.csv) needs review. It can include genuinely different but visually similar images and may miss crops/transformations.

Filename grouping finds 64 groups with more than one image; 43 span source splits. In robbery footage, 42 of 62 multi-image filename groups span splits, including `Robbery008`, `Robbery016` and `Robbery021`. The 10 `knifeDefenseTechnique` images also span train/valid. `BikesHelmets###` and `knife_###` are broad numbered collections, **not** assumed to be one recording; their individual source-session identity remains unresolved. All 309 Call of Duty screenshots are flagged as one game-title collection, not proof of one capture session. See [related_source_groups.csv](../reports/stage2d_manual/related_source_groups.csv). Existing source split assignments cannot be reused as our independent test split.

Before any accepted split: visually adjudicate near pairs, identify original URLs/videos/cameras/recording sessions, join duplicate and related-frame components, then assign whole groups to train/test/validation. Recheck all three split pairs for exact, near and source-group overlap after assignment. Group integrity outranks exact percentages.

## Provenance and rights status

All four ZIP YAMLs and Roboflow README files **state CC BY 4.0** and give version-1 Roboflow URLs. No separate license text was found in the archives. These are publisher/license claims, not independent verification of third-party pixels or the right to redistribute every image. `images for EL MAIN.zip` identifies Tuan Tran/knifes; motorcycle identifies Dendel Pino/Motorcycle Helmet; `guns.zip` identifies Patrick's workspace/People with; `EL.zip` identifies Valentins Workspace/Robbery Headwear Detection. The [Motorcycle Helmet project page](https://universe.roboflow.com/dendel-pino-27gsx/motorcycle-helmet-fhsbn) and [Robbery Headwear version page](https://universe.roboflow.com/valentins-workspace-ya5k3/robbery-headwear-detection/dataset/1) displayed the same CC BY 4.0 claim during this audit; the other two live version pages could not be verified through the available web fetch. Original image creators, source-video rights, attribution and derivative/redistribution status remain open. Watermarks in sampled knife/headwear images and game screenshots in `guns.zip` make source-level claims particularly insufficient.

No original archive was altered. No image or label was copied into `data/processed`. **Approved images = 0; six-class verified images = 0; rights-cleared images = 0.** The [human-review queue](../reports/stage2d_manual/human_review_queue.csv) holds one pending row per image with Approve, Reject and Correct decision fields, six-class completeness, taxonomy, rights and reviewer/date fields. No approvals were fabricated.

## Remaining work and later experimental protocol

The [remaining requirements report](../reports/stage2d_manual/remaining_dataset_requirements.md) gives class-wise gaps and arithmetic bounds. Even if all 2,435 manual candidates passed, at least 7,565 more unique accepted images would be needed for 10,000. Presently none passed, so the measured accepted-image gap is 10,000. The knife source has at most 1,003 provisional positive images, below the 1,500 independent-positive aspiration by at least 497 even under ideal acceptance. Safety vest, backpack and real pistol have no verified supply here. Stage 2B/2C's 187 candidates are still quarantined and must be reconciled for overlap.

After rights and full six-class image review, build a source-group manifest and aim for **70% train / 20% test / 10% validation** (7,000 / 2,000 / 1,000 at 10,000), recording actual proportions when group integrity forces deviations. Generate a class-wise table of positive images and boxes in each split and totals; require meaningful representation of all six classes in all splits. Freeze the independent test membership and its manifest hash before model development. Use validation for model/threshold selection and test once after the protocol is frozen. No final split is built from these unverified labels.

Future training must instantiate `YOLO("yolo11n.yaml")` (or the six-class project architecture YAML derived from it) from random initialization, pass `pretrained=False`, and record architecture/config hashes, no-checkpoint evidence, seed, environment and GPU. Never initialize from `yolo11n.pt`; do not train in this stage. The existing RTX 3050 6 GB plan in Stage 2A calls for a measured small-batch memory pilot when training begins.

On the locked test set, calculate the mentor's **12 required parameters**: TP, FP, FN, precision, recall, F1, AP@50 per class, AP@75 per class, mAP@50, mAP@50–95, mean matched-box IoU, and latency/FPS. Define confidence, NMS and IoU matching thresholds; count duplicate detections as FP; report class-wise support and confidence scores. Use AP area under interpolated precision–recall curves and mean over the six fixed classes with support, with unsupported slices marked unavailable. Measure latency with warmup, fixed batch/input size and CUDA synchronization; FPS = images / measured inference seconds. Save raw and row-normalized confusion matrices including background, PR/F1/precision/recall versus confidence curves, training/validation losses, per-detection confidence CSV, prediction examples, class-distribution plots, failure cases and a reproducibility report. Standard COCO AP/AR summaries in [evaluation_plan.md](../docs/evaluation_plan.md) are supplemental. Detection true negatives are undefined; if accuracy or specificity is requested, define and score a **separate image-level presence/absence task** with reviewed positive and negative images. No evaluation values exist yet.

At the declared matching threshold, precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2TP/(2TP+FP+FN), with undefined denominators reported as unavailable. AP@50 and AP@75 integrate each class's precision–recall curve at IoU 0.50 and 0.75; mAP@50 averages class AP@50, and mAP@50–95 averages AP over IoU thresholds 0.50, 0.55, …, 0.95 and classes. Mean matched-box IoU averages intersection/union over matched prediction–ground-truth pairs under the stated matching rule. The confusion matrix counts matched classes and background FP/FN; normalized rows divide by the corresponding ground-truth support. Record the exact AP interpolation implementation and handling of classes with zero support in the eventual report.

## Reproduction and validation

Run `.\.venv\Scripts\python.exe scripts\stage2d_audit_manual.py` from this project root. The script reads ZIP members directly, checks decoding and normalized labels, hashes images, emits ledgers and review sheets, and never extracts or edits originals. The audit directory also contains [summary.json](../reports/stage2d_manual/summary.json). Generated CSV row counts and the source totals were checked against the ZIP inventory. No datasets were downloaded, no model or pretrained checkpoint was loaded, and no training occurred.
