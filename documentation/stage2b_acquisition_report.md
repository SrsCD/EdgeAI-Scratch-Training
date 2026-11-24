# Stage 2B — first acquisition pilot

Completed 23 November 2026 in **EdgeAI-Scratch-Training** only. No Model A files,
environment, data or checkpoints were touched. No YOLO model or pretrained
weights were loaded, and no training was run. This is an acquisition and
**mechanical audit**, not a training set. Every downloaded image remains
quarantined pending the human six-class audit described below. Source labels are
shown separately from verified project labels.

## Results at a glance

| Measure | Actual result |
|---|---:|
| Images selected for the pilot | 200 (150 Construction-PPE, 50 Open Images; 0 Atharion) |
| Candidate images downloaded | **187** (150 Construction-PPE, 37 Open Images) |
| Open Images selections not downloaded | 13 (7 exceeded 5 MB per-image cap; 6 live Flickr pages did not confirm recorded CC BY 2.0) |
| Accepted for six-class training/evaluation | **0** |
| Rejected after download by mechanical checks | **0** |
| Quarantined pending human review | **187** |
| Measured final acceptance rate | **0/187 = 0% currently accepted**; this is a workflow status, not an estimate of eventual usable yield |
| Source-positive images / boxes | See table below; provisional source annotations only |
| Verified independent capture/video/scene groups | **0**; source group IDs are unavailable in the acquired metadata |
| Exact duplicate images among 187 | **0** by SHA-256 |
| Near-duplicate candidates among 187 | **0** at 64-bit pHash Hamming distance ≤4 (also 0 at ≤10 in exploratory check); visual/source-group duplication remains unverified |
| Asset payload transferred | **731,357,592 bytes** (~697.48 MiB); excludes HTTP page checks, protocol overhead and possible cache traffic |
| Free C: after pilot | **107,462,221,824 bytes** (~100.08 GiB), above 30 GiB reserve |

The 647,666,037-byte Construction-PPE ZIP was downloaded whole, although only
150 of its 1,000 images were extracted for this pilot. Original ZIP and metadata
are unchanged in `data/raw/stage2b_v1`; extracted byte-identical image/label
copies are in `construction_ppe_subset`. Its owner manifest SHA-256 values
matched all 150 selected images. The live archive was much larger than the
Stage 2A estimate, so the measured value replaces that estimate.

| Network asset | Bytes transferred | SHA-256 |
|---|---:|---|
| Construction-PPE ZIP | 647,666,037 | `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232` |
| Open Images validation image metadata CSV | 15,245,485 | `ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17` |
| Open Images validation box CSV | 25,105,048 | `d8bbd59410af14835d7733165a7bb8a3f0213981b22dd5077b0b9f7878991ff2` |
| Open Images boxable class CSV | 12,064 | `1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9` |
| 37 individually checked Open Images originals | 43,328,958 | Individual hashes in manifest |

The extracted 150 Construction-PPE image/label pairs occupy **104,218,395
bytes** locally in addition to the ZIP; the 37 Open Images originals occupy
**43,328,958 bytes**. Check disk again before any later expansion.

## Access and rights gate

| Source | Access, formats, size checked before image download | License/attribution decision |
|---|---|---|
| [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe) | Direct public GitHub release, no authentication; HEAD returned 647,666,037 bytes. ZIP contains 1,000 JPEGs, matching YOLO TXT labels, `manifest.jsonl`, `README.md`, `data.yaml` and `LICENSE.txt`; uncompressed ZIP member total 647,834,864 bytes. | Publisher documentation and archive README explicitly state images **and** annotations CC BY 4.0. Preserve “Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026,” dataset/license links and change notice. This is the publisher's rights assertion, not independent verification of every human-collected image. Per-image real/generated provenance is absent from the archive manifest, so every pilot image remains `unknown_per_image`. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | Public validation metadata and box CSV; direct original image URLs; no authentication for these requests. Metadata lists `OriginalSize`, source URL, Flickr landing URL, author, title and license. Selected first 10 sorted image IDs for each exact boxable MID Person, Helmet, Backpack, Knife and Handgun; 50 distinct IDs. Used a 5 MB cap per original; 37 acquired. | Google licenses annotations CC BY 4.0 and says images are *listed* CC BY 2.0 while expressly disclaiming verification of every image's status. For each downloaded image, metadata reported CC BY 2.0 and the live original Flickr landing page contained the CC BY 2.0 license URI. This verifies a displayed claim at acquisition time, not ownership, consent, or a permanent license guarantee. Author, title, URL, license and page-check status are retained per image; attribution and changes are required. Six pages failed this check, so their images were not fetched. |
| [Atharion PPE v1.0-public](https://github.com/atharion-team/ppe-dataset-v1) | Public v1.0 release API listed a 98,135,012-byte ZIP; no account required in metadata check. YOLO format is advertised. **No Atharion archive or images downloaded.** | [Atharion NOTICE](https://github.com/atharion-team/ppe-dataset-v1/blob/main/NOTICE.md) calls 1,026 images from Ultralytics Construction-PPE 1.0.0 **AGPL-3.0** and 104 Roboflow images CC BY 4.0, while [Ultralytics now states the Construction-PPE dataset itself is CC BY 4.0](https://docs.ultralytics.com/datasets/detect/construction-ppe). Version/provenance and underlying photo rights need reconciliation. Its 1,026-image portion also overlaps our acquired source. It failed this pilot's clear-rights/independence gate. No download or acceptance was attempted. |

All three data sources offer publicly readable pages; none required bypassing
authentication. This pilot did not fetch from the Atharion archive, Drive,
Roboflow, COCO, or weapon collections. The Open Images publisher's own [license
warning](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
means metadata alone is insufficient for inclusion.

## Annotation and image audit

The table below counts mapped boxes **as the sources labeled them**. These are
not verified six-class ground truth and must not enter `data/processed`.

| Fixed class | Provisional positive images in 187 | Provisional source boxes | Accepted positive images | Accepted boxes |
|---|---:|---:|---:|---:|
| 0 person | 161 | 514 | 0 | 0 |
| 1 helmet | 120 | 302 | 0 | 0 |
| 2 safety_vest | 121 | 360 | 0 | 0 |
| 3 backpack | 8 | 11 | 0 | 0 |
| 4 knife | 8 | 13 | 0 | 0 |
| 5 pistol | 4 | 4 | 0 | 0 |

Construction-PPE source IDs were mapped provisionally as `Person` 5→0,
`helmet` 0→1, and `vest` 2→2. Its other nine classes were not relabeled as any
target. In its 150 pilot images, the source had 480 person boxes in 150 images,
282 helmet boxes in 111 images, and 360 vest boxes in 121 images. Open Images
MIDs mapped provisionally: Person `/m/01g317`, Helmet `/m/0zvk5`, Backpack
`/m/01940j`, Knife `/m/04ctx`, Handgun `/m/0gxl3`. Kitchen knife and other
related MIDs were not silently merged. `Handgun` still requires pistol semantic
review. The source validation split has 20 Handgun and 24 Backpack positive
image IDs, so this particular split cannot supply large independent test counts
for those classes; the training corpus must be evaluated separately.

All 150 Construction-PPE pilot images opened successfully; original SHA-256
matched publisher manifest; all 150 label files existed with publisher-reported
row counts; class IDs and YOLO normalized coordinates/box extents passed
mechanical checks. All 37 acquired Open Images files decoded, and their mapped
source boxes passed normalized extent checks. Their per-image original metadata,
all source box rows, live landing check, bytes and file hash are in the immutable
acquisition manifest. No label conversion was written into processed training
folders. The audit can be regenerated with the scripts listed below.

**Visual triage found source taxonomy conflicts**, so mechanical success is not
annotation acceptance. For example, Open Images image IDs
`020c7b8e6d605069` and `0684426d4c177038` show sports helmets in the
contact sheet, outside the fixed protective-helmet scope. Open Images `Person`
can include depictions, and its `Handgun` includes items needing pistol/replica
review. Construction-PPE `vest` includes high-visibility shirts/jackets as
described by the publisher, broader than this project's vest definition. Some
Construction-PPE images appear visually similar by scene even though the
hash-based check found no pairs; source group identity is not supplied. The
archive mixes human-collected and generated images without a per-image flag.
These observations are review flags; no full-image adjudication has occurred.

Every image is therefore quarantined with `six_class_complete=no` and
`split_eligible=no`. A source label for one class does not establish that all
people, helmets, vests, backpacks, knives and pistols in the image are boxed.
No unannotated visible target is treated as background. An accepted count of
zero is deliberate until review completes.

## Annotated contact sheets

The colored rectangles are **source boxes mapped to proposed class IDs**.
These sheets are for human triage and show `QUARANTINED` under each tile.
They are not verified annotations.

![Open Images pilot contact sheet](../reports/stage2b_v1/open_images_v7_contact_01.jpg)

![Construction-PPE pilot contact sheet](../reports/stage2b_v1/construction_ppe_contact_01.jpg)

Additional sheets: [Open Images 2](../reports/stage2b_v1/open_images_v7_contact_02.jpg),
[Construction-PPE 2](../reports/stage2b_v1/construction_ppe_contact_02.jpg),
[3](../reports/stage2b_v1/construction_ppe_contact_03.jpg),
[4](../reports/stage2b_v1/construction_ppe_contact_04.jpg),
[5](../reports/stage2b_v1/construction_ppe_contact_05.jpg).

## Review workflow to turn candidates into accepted images

1. Work from the [200-row review queue](../reports/stage2b_v1/review_queue.csv).
   Keep 13 Open Images selections marked `not_downloaded`; recheck rights or
   size before any future attempt. Do not substitute images without a new pilot
   version.
2. For each of the 187 downloaded images, inspect full resolution and its
   original labels with the fixed definitions in `docs/dataset_protocol.md`
   and Stage 2A. Identify every visible instance of all six classes. Add missing
   boxes in a *derived* annotation file or reject the image; never alter raw
   originals. Record `no target visible` only after explicit review.
3. Specifically adjudicate safety vest versus bright shirt/jacket, worksite
   helmet versus sport/costume helmet, pistol versus long gun/replica, knife
   versus non-knife tool, depiction versus real person, occlusion, small objects,
   and source image rights. Two reviewers should independently check disputed
   cases; a third adjudicates or quarantines unresolved images.
4. Seek upstream source/video/scene IDs for Construction-PPE. Mark each image
   `real`, `synthetic`, or `unknown` with evidence. Group images that depict the
   same underlying scene even when pHash distance is high. Use SHA-256 and pHash
   candidates plus manual visual inspection; rerun cross-source checks after
   Atharion or further sources are considered.
5. Only after rights, full six-class boxes, synthetic flag and source group are
   resolved, change a row to `accepted` or `rejected` with reviewer names and
   reason. A frozen reviewed manifest—not this quarantine queue—will be the
   input to a later group-aware 70/20/10 split. Keep test labels sealed during
   model development.

## Reproducibility and scale decision

Raw downloads and owner metadata: `data/raw/stage2b_v1/`. The
`transfer_receipt.json` records official URL, transferred asset bytes, fetch
time, archive/CSV hashes. `construction_pilot_manifest.json` contains the
seed-42 75+75 selection and per-image checks. `open_images_pilot_manifest.json`
contains the 50 selected IDs, each Flickr license page check, original metadata,
all source box rows, failures, bytes and hashes. `reports/stage2b_v1/audit.json`
contains cross-source hashes and provisional class counts. `checksums.sha256`
pins the derived manifests and queue. Scripts:
`scripts/stage2b_open_images_pilot.py`, `scripts/stage2b_construction_pilot.py`,
and `scripts/stage2b_audit_and_sheets.py`. Rerunning acquisition scripts may
attempt network access; existing immutable raw files should be retained and
checked before reruns. No data were inserted into the six-class train/val/test
directories.

This pilot **does not yet establish a usable acceptance rate or a path to
10,000 accepted images**. The 187 candidates contain only 4 provisional
Handgun-positive images and 8 each Knife/Backpack positive images, and the
Construction-PPE sample has no source group IDs or per-image synthetic status.
The Open Images selection used the first 10 IDs per class and is not a random
estimate of corpus yield. Scaling requires full human review of these 187,
metadata counts from Open Images *train*, new independent pistol/knife and vest
sources with verifiable rights, and group-level test coverage. Atharion cannot
be counted as an independent 150-image addition while its license/version and
overlap questions remain unresolved. If these gates do not clear, publish the
smaller audited six-class dataset with its actual class coverage rather than
claiming 10,000.

**Stage 2B stops here.** No dataset split, training, evaluation or further
acquisition is authorized by this report.
