# Stage 5D — six-image review reconciliation

**24 November 2026. Read-only reconciliation.** The six saved records in [Stage 5C reviews](../data/experimental/stage5c_v1/review_queue/reviews) were compared with the [review queue](../data/experimental/stage5c_v1/review_queue/queue.json) and its append-only `history.jsonl`. Each record has a matching history entry, decision `Hold`, reviewer `gg`, and an image SHA-256 matching both the queue and saved review asset. No image was approved or corrected for training.

| Saved item | Split / source | Decision | Person boxes (ID 0) | Pistol boxes (ID 5) | Other saved boxes |
|---|---|---|---:|---:|---|
| `P5C05` | CCTV frame 5, holdout | Hold | 2 | 1 | None |
| `P5C07` | CCTV frame 7, holdout | Hold | 2 | 1 | None |
| `P5C11` | CCTV frame 11, holdout | Hold | 3 | 0 | None |
| `P5C19` | CCTV frame 19, holdout | Hold | 3 | 0 | None |
| `P5CT1` | Training image, construction PPE | Hold | 1 | 0 | 6 helmet, 3 safety vest |
| `P5CT2` | Training image, knife source | Hold | 2 | 0 | 1 knife |
| **Total** | **Six images** | **Six Hold** | **13** | **2** | **10** |

The first two CCTV records each contain one saved `class_id: 5` box. These are **reviewer-added, unapproved pistol annotations**: the Stage 5C queue had no pistol boxes, and the review history records the added boxes with the Hold decisions. The exact saved count is **two**, based on the records rather than an assumption about boxes per frame. The ten CCTV person boxes and three training-image person boxes also remain on Hold. Saved box counts do not establish box accuracy, object identity, or complete six-class ground truth.

The [accepted-corrections ledger](../data/experimental/stage5c_v1/accepted_corrections.csv) still has **zero correction rows**. Thus **zero training annotations were explicitly accepted** and the actual accepted change in person annotations is **zero**. The Stage 5C train and validation copies remain provisional; the four CCTV frames remain a separate single-scene holdout. No Hold box was promoted into a training label or verified ground truth. Stage 3C data and all train/validation/test splits, Stage 4B confusion matrices and 12 evaluation metrics, model checkpoints, and earlier CCTV results were not modified. No training, download or full evaluation was run.

## Next recommended action

Do **not** continue training from the scratch checkpoint yet. With zero accepted training corrections, a continuation would use effectively unchanged provisional labels and would not test the proposed person-label fix. Preserve the six Hold records as review evidence. If the existing six-image review scope is revisited, resolve the Hold reasons and explicitly approve only complete, defensible training annotations; separately confirm the pistol identities before treating the CCTV boxes as ground truth. A distinct reviewed CCTV scene is still needed for independent assessment. Reassess continuation only after meaningful, traceable training corrections and that validation protocol are available.
