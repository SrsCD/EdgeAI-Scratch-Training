# Stage 3C — experimental six-class dataset

**23 November 2026. EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED. No training has started.** The approved dataset and `stage3_training_preflight.py` remain separate and unchanged in their approval requirements. Stage 3C uses only the existing eight source archives and fixed classes 0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol.

## Materialized data and candidate ledger

The [experimental dataset](../data/experimental/stage3c_v1) contains **7,680 image/label pairs**. The [ledger](../data/experimental/stage3c_v1/ledger.csv) records all **9,441** original candidates with source archive/member, hash, original split, provisional component ID, inclusion disposition, reason, annotation warning, unresolved rights status, and explicit `human_review=No`. Original ZIPs were not edited. Stage 3 source-to-target mappings and polygon enclosures were reused from the provisional labels; no missing box was generated. All included labels are nonempty, use the six fixed class IDs, and passed normalized box validation. Image bytes were checked against the Stage 3 hashes. The dataset [status marker](../data/experimental/stage3c_v1/STATUS.txt) states its provisional nature.

| Disposition | Images |
|---|---:|
| Included, mechanically valid provisional labels | 7,680 |
| Excluded incompatible sources (motorcycle helmet, game/HUD and unrelated headwear) | 1,432 |
| No mapped target label, target absence unverified | 201 |
| Missing, empty or mechanically invalid provisional label | 116 |
| Previously excluded cartoons | 2 |
| Stage 3B known unsuitable/ambiguous/deferred pistol examples | 10 |
| **Total** | **9,441** |

The included source counts are Pistols 2,959, Construction PPE 2,335, knife 980, backpack 962 and Construction Site Safety 444. The 116 invalid/empty labels are quarantined, not repaired by guesswork. The ledger's `annotation_warning` flags source taxonomies that can omit people and other co-occurring target objects, polygon enclosure checks and real-pistol ambiguity. **Mechanical validity does not establish annotation completeness, individual image rights or real-world class suitability.** These limitations affect both training and test labels. Source-level license claims and unresolved underlying-image rights remain in the [Stage 3B rights register](../data/interim/stage3b_review_v1/source_rights_register.csv).

## Exactly ten manual-review examples

The separate [ten-image review queue](../data/experimental/stage3c_v1/review_sample/queue.csv) has two examples from each of the five contributing sources, including person/helmet/vest multiclass images, two PPE polygon conversions, possible omitted people with backpacks or knives, and real-pistol-versus-depiction checks. Its [contact sheet](../data/experimental/stage3c_v1/review_sample/contact_sheet.jpg) says **UNREVIEWED**. The ten IDs are `P0001`, `P0002`, `P0305`, `P0308`, `P0421`, `P0422`, `P0581`, `P0582`, `P0731`, `P0732`. All start Pending; none of the other 7,670 included images is represented as human reviewed.

Run the local review tool from the project root, then open `http://127.0.0.1:8765/`:

```powershell
.venv\Scripts\python.exe scripts\stage3c_review_server.py --port 8765
```

The interface permits box drawing, moving, resizing, class correction and explicit Hold/Correct/Reject/Approve decisions. It stores individual decisions and an append-only history under `review_sample/reviews`. This small sample diagnoses dataset quality; it cannot certify the other images. A rejected or corrected sample means the experimental source needs reconsideration before interpreting model metrics. The experimental ledger deliberately stays `human_review=No` until a future explicit reconciliation process updates it.

## Provisional group-aware split

The Stage 3 provisional component graph supplies **5,674** included groups. A seeded hash of each component ID assigns every member to the same split. Source-provided split names were ignored. This keeps known filename/video/near-duplicate candidate components together, but these are *provisional* groups; unseen shared scenes, originals or related views can still cross splits. Therefore **the test split is held out by this procedure but is not proven independent**. The measured image distribution is train **5,417 (70.5%)**, test **1,473 (19.2%)**, validation **790 (10.3%)**. The [actual per-class table](../data/experimental/stage3c_v1/split_stats.csv) is:

| Class | Train positive images | Train boxes | Test positive images | Test boxes | Val positive images | Val boxes |
|---|---:|---:|---:|---:|---:|---:|
| person | 318 | 649 | 80 | 156 | 44 | 87 |
| helmet | 979 | 1,555 | 278 | 448 | 129 | 231 |
| safety_vest | 988 | 1,571 | 302 | 483 | 174 | 300 |
| backpack | 673 | 702 | 198 | 215 | 91 | 93 |
| knife | 749 | 749 | 149 | 149 | 82 | 82 |
| pistol | 2,078 | 2,432 | 557 | 623 | 324 | 364 |

Counts are based on provisional annotations. An absent class label in an included image is **not** a verified negative. The split is suitable for an experimental software/model run, not for a reliable real-world accuracy claim.

## Scratch configuration and evaluation

The separate [data YAML](../configs/stage3c_data.yaml) and [training YAML](../configs/stage3c_train.yaml) specify the fixed six classes and YOLO11n architecture YAML with random initialization, `pretrained: false`, `resume: false`, RTX 3050 CUDA device 0, seed 42, image size 640 and initial batch 2. CUDA reports **NVIDIA GeForce RTX 3050 6GB Laptop GPU**. The [Stage 3C preflight](../scripts/stage3c_preflight.py) **passed** on 23 November 2026: it checked class/config invariants, every included image hash and label, one-to-one pairing, nonempty labels, class presence in each split and group separation without asserting unperformed approval. The future [launcher](../scripts/stage3c_train_scratch.py) invokes that preflight before constructing `YOLO(configs/yolo11n.yaml)` and rejects any loaded initialization checkpoint. **It was not invoked.** The approved-data preflight still refuses training for lack of human-approved data; it was not bypassed or modified for this experimental path.

The experimental mode of [prediction export](../scripts/stage3b_predict_test.py) and [evaluation](../scripts/stage3b_evaluate.py) uses the Stage 3C ledger and labels and marks results **measured against provisional annotations**. It supports TP, FP, FN, precision, recall, F1, AP@50, AP@75, mAP@50, mAP@50–95, matched-box IoU, inference latency/FPS, raw and normalized confusion matrices, per-class tables, confidence CSV, PR/F1/confidence curves and optional training/validation loss plots. The evaluation core's synthetic fixtures passed; no actual checkpoint or metric exists. Image-level accuracy and specificity require separately reviewed positive and negative image labels and are not computed here.

## Remaining limitations

The ten-image review has not yet occurred. Provisional labels can omit people and co-occurring target objects; pistol suitability and individual-image rights remain unresolved. The component graph prevents known candidate groups from crossing splits but does not prove scene/source independence. Test metrics after a future run must be described as **experimental agreement with provisional annotations**, never reliable real-world accuracy. A model trained from these data may absorb missing-label and rights problems. No additional images, Model A changes or CCTV video detection were introduced.
