# Stage 2A — 10,000-image dataset feasibility

Research snapshot: 23 November 2026. Model B only. This is an acquisition plan,
not an approved dataset or an evaluation result. No image archives, pretrained
weights, or datasets were downloaded, and no training was run. The small official
Open Images class-name CSV was read in memory to confirm category names.

## Decision

**Conditional feasibility, not yet demonstrated.** Approximately 10,000 distinct,
reviewed images is physically and computationally plausible. Available source
counts cannot be added: several repositories repackage the same photographs,
PPE images can omit other target objects, weapon video frames are correlated,
and public access does not establish image reuse rights. The present evidence
does not support promising 1,500 *independent, rights-cleared positive images*
for each of `safety_vest`, `knife`, and especially `pistol`. The independent final
test set is a second bottleneck. Count images containing a class, unique source
groups, and boxes separately; 1,500 boxes can occur in far fewer images.

At 10,000 accepted images, target 7,000 train / 2,000 test / 1,000 validation.
The 1,500-positive-per-class goal is an acquisition target, not an entry criterion
that can be satisfied by repeated frames or augmentation. If the rights and
quality gates fail, use the audited number actually available (a plausible
**4,000–7,000-image contingency planning range**, not a measured yield), state
the class coverage, and collect or annotate additional permissioned source groups.
Do not increase counts with unrelated objects, invented boxes, or variants of
one image. A more defensible fallback may be a smaller six-class proof of concept
with limited pistol and vest claims and a genuinely independent test set.

## Ranked source candidates

Counts below are **published source totals or provisional acquisition ceilings**,
not usable six-class counts. `Unknown` means no reliable estimate was found
without a metadata/image audit. Download sizes are archive sizes where sourced;
otherwise they must be measured with HTTP metadata or source manifests before
acquisition. URLs point to owners' dataset documentation or repository.

| Rank / source | Owner; stated license and rights risk | Access; imagery; format; size | Classes, nominal supply, conflicts and leakage |
|---|---|---|---|
| **1. [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)** ([download](https://storage.googleapis.com/openimages/web/download_v7.html), [terms](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)) | Google LLC: annotations CC BY 4.0; images *listed* as CC BY 2.0, but Google expressly disclaims verification of every image's license. Preserve image-level URL, author, license and attribution; verify selected originals before inclusion or redistribution. | Public metadata and selective download, no account documented; real web/Flickr images; Open Images box CSV plus image metadata CSV. Full 1.9M boxed-image corpus is far too large for this project; selective 3,000–6,000-image candidate pool would be roughly **2–9 GB**, an unverified planning range based on 0.7–1.5 MB/image. Actual bytes are in metadata `OriginalSize`. | Official boxable names include Person, Backpack, Helmet, Knife, Kitchen knife and Handgun. **No exact safety-vest boxable class** found in the official class list. Use exact MID mapping; Handgun needs visual pistol/revolver policy. Source has 16M boxes over 600 classes, but per-target candidate image counts are **unmeasured**. Flickr reposts and overlap with COCO/other datasets require hashes and source URL checks. Bounding boxes are not automatically exhaustive for our six labels; review every accepted frame. |
| **2. [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)** | Ultralytics states images and annotations CC BY 4.0, with attribution and change notice. Verify archive license and any human-collected image terms on receipt; owner statement is not proof for every photograph. | Public archive, no account stated; **mixed real/AI-generated**, YOLO normalized boxes; archive size **not published in owner documentation** (do not budget a precise figure). | 1,000 images / 14,863 total boxes across 12 classes; `helmet`, `vest`, `Person` plus unrelated and absence classes. **0–1,000 usable images**, class-positive counts unknown until audit. Vest definition includes high-visibility shirt/jacket, broader than our fixed `safety_vest`; accept only qualifying vest instances or revise boxes manually. Some labels are AI assisted. Synthetic images must be tagged and capped/stratified; fixed 800/200 source split does not supply independent test. Potential overlap with repackaged PPE sets. |
| **3. [Atharion PPE v1.0-public](https://github.com/atharion-team/ppe-dataset-v1)** | Atharion team claims mixed **AGPL-3.0** (1,026 upstream images) and **CC BY 4.0** (104 images), with per-file attribution. Check upstream rights and whether AGPL terms are acceptable for intended redistribution; do not assume the maintainer's “licence-clean” claim verifies originals. | Public [release](https://github.com/atharion-team/ppe-dataset-v1/releases/tag/v1.0), no account stated; real/staged construction photos (confirm any synthetic subset); YOLO normalized boxes; **94 MB ZIP**. | 1,130 images / 5,355 boxes: Person 2,002; Hardhat 1,776; Safety Vest 1,577 **box counts**, not positive-image counts. Provisional ceiling 1,130 images, likely less after overlap audit. Scene-group splits claimed. It includes material from Ultralytics Construction-PPE and Roboflow Construction Site Safety, so **never add totals to those sources** without exact/near-duplicate and provenance reconciliation. Prior class-ID mismatch was reported by this source; visually verify mapping. |
| **4. [COCO 2017 detection](https://cocodataset.org/#download)** ([terms](https://cocodataset.org/dataset/termsofuse.htm), [detection task](https://cocodataset.org/dataset/detection-2017.htm)) | COCO Consortium annotations CC BY 4.0; consortium **does not own image copyright**. Images are governed by Flickr terms. Per-image reuse rights and attribution require review. | Public access, no account stated for train/val; real photos; COCO JSON boxes. Complete train2017 archive is roughly **18 GB** and val2017 roughly **1 GB** (common release figures; recheck official archive headers before downloading). Selective download size unknown. | 80 object categories; relevant **person** and **backpack** only. 118k train + 5k val labeled images, but relevant positive-image count **unmeasured here**. Provisional selection ceiling 2,000–4,000 after rights/completeness review, not a supply claim. Helmets, vests, knives and pistols can be visible but unannotated; full six-class relabeling or exclusion required. Flickr overlap with Open Images and reused challenge images is possible. Do not use unlabeled COCO test2017 for scored test. |
| **5. [OD-WeaponDetection](https://github.com/ari-dasci/OD-WeaponDetection)** | Ari-DASCI repository says **CC BY-SA 4.0**; GitHub sidebar also shows “CC-BY-4.0 license,” an apparent inconsistency. Read exact dataset/license files and clarify coverage of underlying image/video rights with maintainers. Repository license alone does not settle third-party pixels. | Public GitHub/linked Google Drive; Drive may require Google sign-in, quota access, or manual approval. Real web/video imagery; pistol and knife **detection** sets, with separate classification sets. Detection annotation format appears Pascal VOC XML in downstream description but must be inspected at source. **Image count, box count, archive bytes unverified**; record metadata before use. | Strong candidate for pistol/knife diversity, but exclude classification-only images unless newly annotated. Check `pistol` really means handgun; exclude rifles/replicas if policy demands. Video/web duplicates and overlap with other weapon collections are likely. Provisional usable contribution **unknown**, not counted toward 10,000 yet. |
| **6. [Safety Helmet Wearing Dataset (SHWD)](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)** | njvisionpower repository provides no clearly located image license in its README; positives are reported from Google/Baidu and negatives partly from SCUT-HEAD. **Hold for rights clarification**; public code is not enough. | Public repository/link, external host access may vary; real web/head images; Pascal VOC XML expected, inspect actual files; download bytes unknown. | Published 7,581 images, 9,044 positive helmet objects and many normal heads. Helmet/head focus, **not** comprehensive person or vest boxes. Maximum 7,581 *candidate* images, usable rights-cleared six-class count unknown. Web duplicates and other SHWD repackages require grouping. Do not turn unlabeled bodies into negative person examples. |
| **7. [Real-Time Weapon Detection custom dataset](https://github.com/CostiCatargiu/Yolov12_WeaponDetection)** | Authors state **research-only, no commercial use**; frames come from 1,200+ YouTube videos, films and web search. Third-party video/image rights are not shown verified. **Quarantine pending explicit acceptable-use and image-rights review.** | Public Roboflow companion projects linked from repository; downloads may need Roboflow account/API key. Real video frames and web images; YOLO boxes, corrected labels separately. Full download bytes unknown. | Published 26,528 images / 38,067 instances across `knife`, `pistol`, `long_gun`, `no_weapon`; per-class positive-image totals need inspection. `no_weapon` is not one of our six classes, and long guns must never become pistol. Highly correlated frames; retain video IDs and cluster near duplicates. Do **not** treat the 26,528 as eligible supply. |
| **8. [Armed CCTV Footage (ACF)](https://www.mdpi.com/1424-8220/22/19/7158)** | Academic paper reports participant consent and a repository, but the linked GitHub URL currently returned 404 in this research pass. Dataset-specific license/redistribution terms **unconfirmed**. | Source access currently unresolved; real staged CCTV; paper describes detection annotations; exact downloadable format/size unknown. | Paper reports 8,319 CCTV frames combined, including 3,559 knife frames; pistol counts should be checked against source tables. **0 currently acquirable/approved** until host and terms are resolved. Strong video correlation means frame counts overstate independent evidence. |

**Do not use Open Images Extended as a ready detection source:** its owner says the
382k-image expansion currently has image-level labels but no bounding boxes.
Likewise, a Roboflow Universe page's availability or a Kaggle mirror's license
badge cannot establish rights to all underlying photos. Dataset mirrors must be
traced to their original owner and deduplicated before ranking as new supply.

## Coverage budget and gates

These are **planning targets and unverified ceilings**, not observations. Images
may carry several classes; columns must not be summed as if mutually exclusive.

| Class | Target positive images | Likely sources | Current evidence / gap |
|---|---:|---|---|
| person | ≥1,500 | COCO, Open Images, PPE | Likely abundant; six-class annotation completeness and rights limit acceptance. |
| helmet | ≥1,500 | PPE, Open Images; SHWD conditional | PPE published helmet *boxes* do not prove 1,500 distinct images. SHWD rights unresolved. |
| safety_vest | ≥1,500 | PPE, permissioned new capture/annotation | **Major limit.** No exact Open Images boxable class; Atharion has 1,577 vest boxes across ≤1,130 images and overlaps upstream. Need new independent groups. |
| backpack | ≥1,500 | COCO, Open Images | Likely attainable from large pools; rights and six-class review still needed. |
| knife | ≥1,500 | Open Images, OD-WeaponDetection conditional, permissioned capture | Need class-disambiguated, rights-cleared positives; web/video duplicates and mixed kitchen/weapon contexts. |
| pistol | ≥1,500 | Open Images Handgun, OD-WeaponDetection conditional, permissioned capture | **Primary limit.** Handgun/pistol semantics, image rights, diversity and independent test groups require review. Research-only large dataset cannot be counted as broadly reusable. |

Do a metadata-only feasibility tally first: candidate image IDs, original URLs,
source licenses, class-box counts, and estimated `OriginalSize`. Then acquire a
small pilot and measure usable yield by source/class, unique group, and rights
status. If two successive pilots yield <60% accepted images or the pistol/vest
positive trajectory falls below 1,500 at the projected 10k budget, reduce the
claim or obtain permissioned capture/annotation; do not quietly pad the pool.

## Annotation contract

Fixed IDs are `0 person`, `1 helmet`, `2 safety_vest`, `3 backpack`, `4 knife`,
`5 pistol`, matching `configs/dataset.yaml`. Canonical boxes cover visible object
extent, clipped to the image, with positive normalized YOLO `cx cy w h`. Keep
original labels, transforms and reviewer changes in a versioned manifest.

- **Person:** visible human body or sufficiently visible partial body; exclude
  mannequins, posters and reflections unless a written policy explicitly includes
  them. Mark crowd/uncertain cases for review. Helmet-only head datasets cannot
  imply a complete person box.
- **Helmet:** wearable protective safety helmet/hard hat; exclude bicycle, sports,
  costume and motorcycle helmets for this surveillance/PPE scope. Open Images'
  broad `Helmet` must be visually filtered. Draw the helmet object, not the head.
- **Safety vest:** visible high-visibility protective vest with distinct vest
  structure; exclude plain bright shirts, jackets, ordinary clothing and partial
  colored fabric when identity is uncertain. Source `vest` may include shirts and
  jackets; relabel qualifying instances and reject/flag the rest.
- **Backpack:** bag worn or designed to be worn on the back; exclude handbags,
  briefcases, purses and generic luggage.
- **Knife:** visible bladed knife, including kitchen/utility knives if within the
  intended broad class; exclude scissors, swords, machetes, toy knives and
  knife-like tools unless a later written taxonomy decision explicitly includes
  them. Keep use-context tags so kitchen examples do not dominate threat claims.
- **Pistol:** visible handheld handgun, including revolvers under the operational
  handgun definition; exclude rifles, shotguns, toy/replica guns when discernible,
  and ambiguous side views. If genuine/replica status cannot be judged visually,
  quarantine rather than guessing.

For **every accepted image**, audit the pixels for all six classes, including
small/occluded objects and co-occurring people, helmets and vests. A target
object missing from a source's taxonomy is **unknown**, not background. Add a
reviewed box or exclude that image; do not merely drop source classes. Source
`no_helmet`, `no_vest`, `long_gun`, `no_weapon` and other non-target boxes are
not mapped to a target. A non-target object may remain visible, but any target
class it contains must be boxed. Explicit empty label files are allowed only
after review confirms no visible target object. Keep `ignore/uncertain` records
outside YOLO labels and exclude unresolved frames from train/val/test.

Two reviewers independently inspect a stratified sample of at least 10% per
source and all ambiguous pistol/knife/PPE instances; one reviewer boxes and one
adjudicates the complete six-class image. Escalate disagreement to a third
reviewer or exclude. Track confusion reasons, corrected/missing boxes, category
mapping and source-specific acceptance rates. Before finalizing, audit **all**
test images and a random 10% of train/val, increasing review if systematic
errors appear. `1,000 images × 2–5 minutes` is roughly 33–83 reviewer-hours
for one pass; two-pass review of 10,000 images is roughly **670–1,670
reviewer-hours**, before difficult relabeling. This is a planning rate, not a
measured throughput. Human effort is likely the dominant cost.

## Split and leakage policy

Build a master manifest before assigning splits: source ID/version, original
image ID and URL, license/attribution, capture/video/session/site/scene IDs,
photographer where available, class-presence vector, dimensions, hashes, and
review status. Preserve source-provided split only as metadata; reassign at
**source-group** level to meet this project's 70/20/10 design. Never allow one
video, burst, camera session, staged scene, original image or its derivatives
across splits. Where group metadata is missing, derive conservative groups from
URL, filename, visual similarity and manual review.

Compute SHA-256 for exact duplicates; canonical pixel hashes for recompressed
copies; perceptual hash and embedding similarity as **candidate** near-duplicate
signals. Review neighbors manually, including crops, rotations, watermarks and
Roboflow augmentations. Union connected duplicates and related frames into one
group before stratifying. Audit train↔val, train↔test, and val↔test with the same
checks after splitting. Reject any cross-split duplicate or uncertain source
family. Hold out **independent capture/source groups** for test, with positive
coverage for every class if available. If source groups are too few for a class,
report that limitation instead of distributing correlated frames to satisfy 20%.

At 10k accepted images aim for 7k/2k/1k, with group integrity taking priority
over exact counts. Freeze test manifest and SHA-256 hash before development;
restrict access to its labels, use validation for early stopping, augmentation,
threshold and checkpoint choice, and run final test once after the protocol is
locked. Document any later test amendment as a new version and evaluation.

## Training and evaluation compatibility

Convert reviewed annotations to `data/processed/images/{train,val,test}` and
matching `labels/`, with exactly the fixed ID order. Stage 1's YAML-derived
architecture is already six class; for future training instantiate
`YOLO("configs/yolo11n.yaml")`, set `pretrained=False`, `resume=False`, and do
not initialize from `yolo11n.pt`. Keep `amp=False` unless a future verified
offline check can avoid fetching pretrained reference weights. No training in
this stage.

For validation and locked final test, define IoU matching and confidence/NMS
settings. Export standard and row-normalized confusion matrices with background;
TP, FP, FN; precision, recall, F1; AP@50, AP@75, mAP@50, mAP@50:95; mean
matched-box IoU; and per-class support/results. Export image-level detection
rows with class, confidence, pixel/normalized box and run settings, plus PR/F1
curves, training loss curves, and threshold policy. Measure inference latency
with warmup, synchronization, fixed batch/image size and hardware conditions;
report median and percentile latency and FPS = images/elapsed seconds. Avoid
confusing inference FPS with full pipeline FPS. COCO bbox evaluation's 12 standard
AP/AR summaries remain specified in `docs/evaluation_plan.md`. Conventional
accuracy/specificity require a separately defined **image-level** present/absent
task and reviewed negatives; they are not ordinary detection metrics.

## Storage, processing and RTX 3050 plan

Current project-drive free space measured during this stage: **108,396,916,736
bytes (~100.95 GiB)** on C:. Stage 1's venv and CUDA packages are already on this
drive; the free-space figure is live and can change. For 10k images, plan roughly
**7–20 GB** raw candidate downloads (depending on selection and image size),
**7–20 GB** decoded/reviewed copies, **2–10 GB** intermediate/resized or audit
derivatives, and **5–20 GB** runs/checkpoints/plots. Set an operational reserve
of **≥30 GiB free** after acquisition and stop before crossing it. These are
engineering estimates, not dataset-published sizes. Large full COCO/Open Images
downloads would consume much more and are unnecessary. Measure source archive
headers and manifest `OriginalSize` before committing storage.

Processing: decode and repair orientation, record dimensions and image hash,
convert COCO/VOC/OID boxes to canonical pixel XYXY then YOLO normalized XYWH,
validate bounds/class IDs/paired files, annotate missing target objects,
deduplicate, group and split, then run visual spot checks. Keep raw originals
immutable and attribution alongside derived labels. Quarantine synthetic data
and annotate its provenance; do not let similar generated variants leak.

The RTX 3050 has 6 GiB VRAM. Stage 1 only verified a 320-pixel inference pass,
not 640-pixel training. Future run should probe batch 2–4 at 640, workers 0–2
on Windows, watch memory/thermals and reduce batch or image size if needed.
Scratch training may need more epochs than fine-tuning; training time and final
accuracy cannot be estimated credibly before a pilot. Use validation curves and
fixed seeds, but report hardware/software versions and any nondeterminism.

## Milestones and concrete first acquisition proposal

1. **Metadata and rights gate, no bulk images:** pin source versions, read the
   actual archive/dataset licenses, verify selected per-image rights and creator
   attribution. Query Open Images box CSV + image metadata for exact Person,
   Backpack, Helmet, Knife and Handgun MIDs; tally *distinct images and boxes*
   by class, source license, and `OriginalSize`. Ask OD-WeaponDetection maintainers
   to clarify the CC BY versus CC BY-SA discrepancy and upstream image rights.
   Keep research-only and unlicensed collections quarantined.
2. **First small acquisition after approval:** fetch **up to 500 candidate images**:
   200 Open Images (40 each for Person, Backpack, protective-helmet candidates,
   Knife, Handgun), 150 from the Ultralytics Construction-PPE archive (whole
   archive may be required; verify its bytes first), and 150 Atharion PPE images
   (94 MB archive), **subject to source rights acceptance**. If archive access
   is all-or-nothing, cap pixel review to 150 and record full downloaded bytes.
   The PPE sources overlap: select disjoint original IDs, or count duplicates
   once. Do not pull COCO or weapon Drive archives in this first pilot. Preserve
   source metadata and original labels; no model weights.
3. **Pilot audit:** verify image rights and six-class completeness, count unique
   positive images/boxes/groups, overlap, ambiguous taxonomy, reviewer time and
   actual storage. Review all 500 for target visibility; make an explicit
   accept/reject ledger with reasons. Do not expose pilot test labels to tuning.
4. **Scale decision:** if ≥60% of the pilot survives and pistol/vest groups are
   diverse, expand in 1,000–2,000 candidate batches, always re-auditing
   duplicates and rights. Reserve independent test groups early. If coverage
   stalls, source permissioned new scenes and manually annotate; revise the
   claimed dataset size and per-class target honestly.
5. **Freeze acquisition:** publish counts of accepted images per class/split,
   unique source groups, license buckets, synthetic fraction, rejected images,
   unresolved rights, duplicate clusters and manifest hashes. Only then schedule
   Stage 2B preparation and later training.

**Approval checkpoint:** Stage 2A stops here. The concrete proposal is the
rights/metadata gate followed by the capped 500-candidate pilot. No acquisition
or training is authorized by this report alone.
