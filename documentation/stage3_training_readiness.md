# Stage 3 — scratch-training readiness

**23 November 2026. Status: data acceptance blocked; training not started.** This work uses only the eight ZIP archives and Stage 2D/2E audit artifacts in `EdgeAI-Scratch-Training`. No dataset was downloaded or searched for, no original ZIP or `EdgeAI-Surveillance` file was modified, and no pretrained `.pt` weight was loaded. The fixed class order remains **0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol** with the Stage 2D definitions.

## Selection and provisional annotation result

The eight archives hold **9,441** image candidates. The [Stage 3 selection manifest](../data/interim/stage3_candidates_v0/selection_and_conversion_manifest.csv) records a disposition and reason for every image:

| Disposition | Images | Meaning |
|---|---:|---|
| Mapped candidate, quarantined | **7,806** | Provisional six-class source labels written under `data/interim`, not accepted ground truth |
| Priority-source candidate with no mapped target label | **201** | 200 Construction Site Safety and 1 Backpack; target absence is unverified |
| Excluded source | **1,432** | 629 motorcycle-helmet, 309 game/HUD, 494 unrelated headwear images |
| Excluded image | **2** | Visually confirmed pistol cartoons `armas (1814)` and `armas (2952)` in the Stage 2E review sheet |
| **Total** | **9,441** | **Accepted = 0** |

The five prioritized sources thus contain **8,007** images after two known cartoon exclusions. More unsuitable pistol depictions may exist; the other 2,969 pistol images remain quarantined pending individual review. The 7,806 provisional TXT files contain **11,387 mapped candidate boxes**. [Polygon-to-box log](../data/interim/stage3_candidates_v0/polygon_to_box_log.csv) records **1,317** construction-PPE `Helmet` segmentation enclosures, with original source class ID and label line. Each enclosure uses min/max polygon coordinates, remains provisional, and must be checked against the industrial hard-hat definition and object extent. No fabricated boxes were added. Source classes outside the six targets were omitted from provisional TXT files but remain recorded in the source manifest; this omission does **not** certify target completeness. No zero-target file was turned into a negative training label.

| Target class | Provisional source-positive images | Provisional mapped boxes | Accepted positive images / boxes |
|---|---:|---:|---:|
| person | 515 | 1,148 | 0 / 0 |
| helmet | 1,438 | 2,391 | 0 / 0 |
| safety_vest | 1,492 | 2,406 | 0 / 0 |
| backpack | 962 | 1,010 | 0 / 0 |
| knife | 1,003 | 1,003 | 0 / 0 |
| pistol | 2,969 | 3,429 | 0 / 0 |

Image counts overlap across classes and include unresolved related views. The [candidate class statistics CSV](../data/interim/stage3_candidates_v0/candidate_class_statistics.csv) gives per-source counts. Stage 2D/2E review sheets and ZIP member labels remain the evidence for source-to-target proposals; every retained image still needs full six-class review. Construction PPE has no person source class even where people are visible; knife, backpack and pistol sources likewise omit co-occurring target categories. Site Safety annotates person/hardhat/vest but not all other fixed classes. The selection manifest includes these source-taxonomy omission flags per image. Provenance and image-use rights are unverified for all 8,007 priority candidates. A Roboflow license badge is not an individual original-image rights check.

## Duplicate and split status

The [duplicate/source-group report](../reports/stage3/duplicate_source_group_report.md) documents **1,929 unresolved pHash review pairs** within the prioritized candidates, **575** crossing source-provided split labels. There are **828** multi-image filename/video group hints, **442** crossing source splits. A conservative union of hints and candidate pairs puts 3,360 images in multi-image provisional components; these are **not final independent groups or confirmed duplicates**. Exact SHA-256 duplicates across the eight ZIPs were zero in Stage 2D/2E. No visually similar but independent image was deleted. The previous Roboflow splits were not adopted.

**No accepted train/test/validation split exists.** The [accepted split statistics](../reports/stage3/accepted_split_statistics.csv) show zero accepted images and objects for every class, marked `not_created_no_accepted_images`; these are not source-positive or final evaluation counts. The required target remains **70% training / 20% independent testing / 10% validation**, with class-wise positive images and object counts in every split wherever reviewed data permits. Final group IDs must keep originals, variants, scenes and video sessions together. Only after review should the [group-aware split script](../scripts/stage2e_split_approved.py) assign groups, record actual percentages, check each class in all splits, and freeze a hashed test membership. If verified test labels are unavailable, subsequent test scores cannot be presented as reliable real-world performance.

## Minimum actions to unblock training

The [experimental dataset proposal](../reports/stage3/experimental_dataset_proposal.md) gives the review order and gate detail. In practical terms: (1) review image pixels and correct all six target labels, including person omissions and the 1,317 polygon enclosures; (2) reject remaining cartoons/toys/ambiguous items and inspect the 201 no-mapped-box cases; (3) record underlying image/annotation rights and attribution for each retained image or source group; (4) adjudicate related pairs and capture sessions and assign final group IDs; (5) approve a subset with real positive support for all six classes, produce reviewed label files and an independent group-separated 70/20/10 split; (6) copy only approved image/label pairs to `data/processed`, validate class-wise split counts and hash/freeze the independent test set. **The accepted subset can be smaller than 10,000** if the evidence supports it; 10,000 is a target, not permission to bypass a gate.

The candidate pool is 1,993 images short of 10,000 even before further rejections (**10,000 − 8,007**), but the user has chosen not to acquire more data now. The current accepted-image gap is 10,000 because acceptance has not begun. The project's separate Stage 2B/2C 187-image pilot remains quarantined and is outside this eight-ZIP preparation; reconcile overlap before using it.

## Scratch-training configuration prepared, not executed

The unified six-class [data YAML](../configs/stage3_unified_data.yaml) points to `data/processed/images/{train,val,test}` and corresponding labels. These directories contain no accepted images. The [training YAML](../configs/train.yaml) selects the project's six-class `configs/yolo11n.yaml` architecture (constructed as `YOLO("configs/yolo11n.yaml")` when training later), **`pretrained: false`**, **`resume: false`**, CUDA device 0, seed 42, deterministic mode, image size 640, batch 2, workers 0, 150-epoch maximum, no RAM/disk cache, and `amp: false` to avoid the pretrained reference-model AMP check. Batch 2 is a conservative starting point for the RTX 3050 6 GB; probe memory and throughput when training is authorized, and adjust only with a recorded config revision. No pretrained checkpoint is used for initialization.

| Prepared artifact | SHA-256 |
|---|---|
| `configs/yolo11n.yaml` | `cfef81e558b212e6f4061f20419fab431066707720bf1e14286180f38c696c61` |
| `configs/stage3_unified_data.yaml` | `5db1295c82772a5b76defc124344667a518aa811f1c7c2bc815a1b33a0b404db` |
| `configs/train.yaml` | `39e828d6f0fc4d5ce2ddf49dc46388ee922358e70ff23f0f0c9792183f6c1a11` |

The read-only [training preflight](../scripts/stage3_training_preflight.py) must pass after approval and materialization: it checks scratch flags, fixed classes, frozen split manifest, class support, group/hash separation, processed image/label pairing and label counts. It presently stops because no approved split manifest exists. Do not invoke the trainer until this gate passes and the test set is frozen.

## Mentor evaluation contract

After scratch training, evaluate a frozen checkpoint on the independent reviewed test split. Calculate **TP, FP, FN, precision, recall, F1, per-class AP@50, per-class AP@75, mAP@50, mAP@50–95, mean matched-box IoU, and inference latency/FPS** with documented confidence, NMS and IoU matching thresholds. Export raw and normalized confusion matrices, per-class support/results, confidence-score CSV, PR and confidence curves, train/validation losses, prediction examples and failure cases. [The evaluation plan](../docs/evaluation_plan.md) retains formulas and the supplemental COCO AP/AR summaries. Do not treat mAP as classification accuracy. Accuracy/specificity require a separately defined image-level presence/absence protocol and reviewed negative images. No training or evaluation results are claimed here.

## Reproducibility and validation

Run `scripts/stage3_prepare_candidates.py` to regenerate provisional mapped labels and selection reasons from the immutable Stage 2E manifest, then `scripts/stage3_group_candidates.py` to regenerate provisional leakage components. These scripts write only `data/interim` and `reports/stage3`. Stage 3 validation checked candidate counts, mapped label coordinates, fixed class IDs, polygon conversion totals, source ZIP hashes, empty `data/processed` directories and the preflight's expected refusal. No final dataset was released.
