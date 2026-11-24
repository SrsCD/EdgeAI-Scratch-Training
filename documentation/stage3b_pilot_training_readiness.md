# Stage 3B — pilot dataset and scratch-training readiness

**23 November 2026. Status: review queue ready; dataset acceptance and training blocked.** This pilot uses only the existing EdgeAI-Scratch-Training archives. No additional dataset was acquired, no model was trained, and no pretrained `.pt` checkpoint was used to initialize a model. The fixed classes and Stage 2D definitions remain **0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol**. Model A and CCTV video work were not touched.

## What is ready to review

The deterministic, seed-42 [pilot queue](../data/interim/stage3b_review_v1/queue.csv) has **900 unapproved images**, copied byte for byte from the existing source ZIPs into [offline review assets](../data/interim/stage3b_review_v1/assets). The [queue JSON](../data/interim/stage3b_review_v1/queue.json) preserves original archive/member path, image hash, source split and group hint, pHash, source URL, review flags and provisional mapped boxes. [Thirty-one contact sheets](../data/interim/stage3b_review_v1/contact_sheets) show the provisional boxes with an explicit **UNREVIEWED SOURCE LABELS** caption. Selection required a singleton in the Stage 3 provisional component graph and pHash distance greater than 10 within the pilot. It does not assert that selected views are independent.

| Source | Queued images | Provisional positive images: person / helmet / vest / backpack / knife / pistol |
|---|---:|---|
| Construction Site Safety | 240 | 239 / 184 / 145 / 0 / 0 / 0 |
| Construction PPE | 180 | 0 / 122 / 122 / 0 / 0 / 0 |
| Backpack Detection | 160 | 0 / 0 / 0 / 160 / 0 / 0 |
| Knife dataset | 150 | 0 / 0 / 0 / 0 / 150 / 0 |
| Pistols Detection | 170 | 0 / 0 / 0 / 0 / 0 / 170 |
| **Total** | **900** | **239 / 306 / 267 / 160 / 150 / 170** |

The corresponding provisional box counts are **712 / 753 / 569 / 160 / 150 / 190**. Positive-image counts overlap among classes. These are source annotation proposals, **not reviewed class support**: PPE, backpack, knife and pistol source taxonomies omit possible co-occurring target classes, especially people; Site Safety also omits other target classes. The 57 queued Construction PPE polygon enclosures require checking against the hard-hat definition and object extent. A reviewer must inspect every visible instance of all six classes before approval. No absent source label is treated as a verified negative and no missing box was invented.

Nine unsuitable or ambiguous pistol depictions were screened out of this pilot; one pHash-distance-10 cross-source candidate was deferred. Their paths and reasons are in [pre-screen exclusions](../data/interim/stage3b_review_v1/pre_screen_exclusions.csv). They remain in the source archive and were not declared human-reviewed. The [source-group/leakage report](../reports/stage3b/pilot_source_group_leakage.md) records the screen and remaining group decisions. All 900 queue records currently say **Pending**; `reviews/` contains **zero** decision files.

## Offline correction and approval workflow

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts\stage3b_review_server.py --port 8765
```

Open `http://127.0.0.1:8765/` on this computer. The server binds to loopback only. Filter by source/status or ID, inspect the contact sheets in batches, then inspect each image at full resolution in the browser. Existing boxes are editable proposals. Use **Draw new box**, select/move/resize, the numeric normalized coordinates, and Delete to correct labels. Choose among **Correct**, **Hold**, **Reject** and **Approve**. `Correct` saves progress but does not export. `Reject` requires a reason. `Approve` requires reviewer name, completed taxonomy and full six-class checks, image/annotation rights evidence, a resolved final source-group ID, and explicit negative confirmation if no boxes remain. Decisions are saved per image in `data/interim/stage3b_review_v1/reviews/`; `history.jsonl` appends every decision. The queue preparation script refuses to regenerate IDs once review files exist.

For each image, determine whether every provisional box matches the fixed definition and whether any target object is unlabeled. Add only boxes supported by pixels. Reject cartoons, game content, toys/replicas or other unsuitable examples; put uncertain imagery on Hold. A source group must encompass related frames, crops, variants and shared underlying originals, even when supplied source split labels differ. Assign the same final group ID to related approved images. Use Hold when the relationship or rights cannot be resolved. A checkbox records the reviewer's assertion; software cannot prove its truth.

## Rights and attribution gate

The [source rights register](../data/interim/stage3b_review_v1/source_rights_register.csv) separates publisher-level claims from individual-image evidence. Live project pages show CC BY 4.0 claims for [Construction Site Safety v30](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30), [Construction PPE v1](https://universe.roboflow.com/gaos-workspace/construction-ppe-qofi4/dataset/1) and [Backpack Detection v1](https://universe.roboflow.com/vrk2/backpack-y9y87/dataset/1), and a Public Domain claim for [Pistols Detection v1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1). The knife ZIP README/YAML asserts CC BY 4.0, but its live page was not verified. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits sharing and adaptation with attribution, license link and change indication; it does not itself establish that the publisher owns each underlying photograph. Source-level review supports preserving source attribution and local review. It does **not** clear stock watermarks, retail/product photographs, borrowed web photos, people/privacy or annotation provenance. Images needing such clarification stay on Hold or are rejected. For each approval, the reviewer must enter concrete image-specific or defensible source-group evidence and preserve original creator/project, URL, license terms and annotation changes as applicable. A dataset badge alone does not satisfy the approval gate.

## Acceptance, split and training gate

There are currently **0 approved images, 0 accepted boxes and no train/test/validation split**. Consequently no honest per-class split distribution or real model metric can be reported. The [approved export script](../scripts/stage3b_export_approved.py) presently stops with `No human-approved images`. After genuine approvals, it validates box coordinates and asset hashes, checks very close approved duplicate candidates, assigns **whole final groups** toward 70% train / 20% test / 10% validation, and requires at least **10 / 5 / 5 positive images per class** across train/test/validation and at least three independent positive groups per class. If those conditions cannot be met, it refuses export. It writes approved image/label pairs under `data/processed`, an attribution/provenance [split manifest](../data/manifests), [per-class split statistics](../reports/stage3b/pilot_split_stats.csv) with positive-image and box counts, and a hashed independent test freeze. The source splits are not reused. Actual ratios may deviate to keep groups intact.

```powershell
.venv\Scripts\python.exe scripts\stage3b_export_approved.py
.venv\Scripts\python.exe scripts\stage3_training_preflight.py
```

The [preflight](../scripts/stage3_training_preflight.py) currently stops with `No frozen approved split manifest`. After export, it checks fixed classes, scratch flags, test freeze/manifest consistency, group and exact-hash separation, image/label pairs, content hashes and per-class box counts. The [data YAML](../configs/stage3_unified_data.yaml) already specifies the unified six-class processed paths. The [training config](../configs/train.yaml) uses the six-class [YOLO11n architecture YAML](../configs/yolo11n.yaml), `pretrained: false`, `resume: false`, CUDA device 0, seed 42, image size 640, batch 2, workers 0 and `amp: false`. The future [training launcher](../scripts/stage3b_train_scratch.py) calls preflight before constructing `YOLO(configs/yolo11n.yaml)` with random weights and checks the RTX 3050; **it was not run**. Batch size 2 is a conservative 6 GB starting point, to be revised only if the actual run requires it.

## Evaluation pipeline

The [evaluation core](../src/edgeai_scratch/detection_eval.py) computes classwise one-to-one confidence-ordered TP, FP and FN, precision, recall, F1, AP@50, AP@75, mAP@50, mAP@50–95, mean matched-box IoU and synchronized batch-one model-only latency/FPS. AP uses 101 recall-point interpolation; the 50–95 summary uses ten IoU thresholds from 0.50 to 0.95. Unsupported class denominators are marked unavailable. The [inference exporter](../scripts/stage3b_predict_test.py) is limited to a checkpoint under this project's scratch run directory with provenance; loading that checkpoint later for inference is distinct from initializing training. The [evaluation report script](../scripts/stage3b_evaluate.py) requires a frozen approved manifest, writes per-class tables, confidence-score CSV, raw and row-normalized confusion matrices, PR and F1/precision/recall confidence curves, prediction examples, optional training/validation loss plots, and a report. It refuses to overwrite an existing test result. Test metrics are credible only to the extent that held-out labels, groups and source rights have actually been reviewed; a small curated pilot does not establish real-world surveillance accuracy.

The [evaluation synthetic fixtures](../tests/test_stage3b_eval.py) passed **3/3** and the [group-split fixture](../tests/test_stage3b_export.py) passed **1/1** on 23 November 2026. They check IoU, known **TP=2, FP=2, FN=1**, AP, timing, a wrong-class confusion case and feasible independent-group class support. These are software tests only. No model predictions or performance results were created.

Detection accuracy and specificity are omitted because box detection has no natural true-negative box count. If a mentor requires them, define a **separate image-level, per-class presence/absence protocol** before collecting results: use reviewed positive and reviewed negative images for each class; declare a class present when at least one prediction at a fixed confidence threshold exists in an image; count image-level TP/TN/FP/FN; report accuracy `(TP+TN)/(TP+TN+FP+FN)` and specificity `TN/(TN+FP)` per class with denominators and threshold. Keep the independent test split untouched while choosing that threshold on validation data. The current queue has no reviewed negatives.

## Exact remaining actions

1. Review and correct a sufficient subset of the 900 queue images in the offline tool. Record each image's taxonomy completeness, source/group relationship, rights evidence, reviewer and decision. Keep uncertain cases on Hold or Reject.
2. Obtain enough truly accepted independent positives for **all six classes**, including enough per class for the exporter’s 10/5/5 split minimum. If a class or rights source cannot clear, the exporter will remain blocked; do not lower a gate silently.
3. Run the export command, inspect `reports/stage3b/pilot_split_stats.csv` and the frozen test membership, then run preflight. Only a passing preflight makes the first scratch experiment eligible. Training and evaluation commands should be run after that separate decision, never as part of this Stage 3B preparation.

Verification this stage: the local HTTP server returned 900 Pending queue records and an individual item, rejected a deliberately incomplete approval with HTTP 400, and saved **no** review record; export and preflight refused as expected. No accepted split, checkpoint or real test output exists.
