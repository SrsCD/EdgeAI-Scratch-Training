# Stage 2C — quarantine resolution and review handoff

23 November 2026. This review uses only the 187 images acquired in Stage 2B:
150 Ultralytics Construction-PPE and 37 Open Images V7. **No image has been
accepted or released to a training, validation, or test folder.** No further
images were downloaded, no pretrained weights were loaded, and no training or
final split was created. The six fixed classes remain `person`, `helmet`,
`safety_vest`, `backpack`, `knife`, `pistol` in IDs 0–5.

## Outcome

The original Stage 2B [acquisition report](stage2b_acquisition_report.md),
[review queue](../reports/stage2b_v1/review_queue.csv), raw manifests, original
source labels, and empty project attribution/image-manifest templates were
inspected. Stage 2C adds a structured [per-image JSON ledger](../reports/stage2c_v1/review_ledger.json),
[CSV ledger](../reports/stage2c_v1/review_ledger.csv),
[attribution rows](../reports/stage2c_v1/attribution.csv),
[proposed corrections](../reports/stage2c_v1/proposed_corrections.json), and an
[offline review interface](../reports/stage2c_v1/review.html). Raw image and
annotation files remain unchanged. The code that rebuilds these derived review
artifacts is `scripts/stage2c_build_review.py`.

The ledger records **independent gate states** for license, exact-image rights,
attribution, box geometry, class definition, all-six-class annotation
completeness, source group, exact/near duplicates, and real/generated status.
The route below is a *primary work queue*, not a claim that other gates passed.

| Primary route | Images | Meaning |
|---|---:|---|
| Eligible pending human review | **0** | None has a verified source group and a complete six-class annotation review yet. |
| Requires annotation correction | **9** | Full-image visual triage found a specific source-label conflict; proposed actions are unapproved. These nine also need rights and group review. |
| Requires rights clarification | **28** | Open Images entries without a specific proposed correction; the original Flickr photo's license applicability still needs human confirmation. These also need group and annotation review. |
| Requires source-group clarification | **150** | Construction-PPE entries without a specific correction proposal; scene/session provenance is missing. These also need six-class review. |
| Rejected with documented reason | **0** | Stage 2C proposes potential rejections but makes none on behalf of the human reviewer. |

### Remaining blockers, counted independently

| Gate / flag | Images | Evidence and next action |
|---|---:|---|
| Exact-image rights clarification | **37** | Open Images metadata and the Stage 2B live Flickr page check both showed a CC BY 2.0 URI, but the check did not preserve a page snapshot or prove that the URI applied to the exact photo. The [Open Images license page](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) expressly disclaims image-level license verification. Recheck the exact landing photo, creator and current terms; retain evidence. |
| Publisher license statement, individual creator not verified | **150** | [Ultralytics Construction-PPE documentation](https://docs.ultralytics.com/datasets/detect/construction-ppe) and the archived README say images and annotations are CC BY 4.0. This is a publisher assertion adequate to document the source license; it does **not** prove every individual human-collected photo's chain of title. Record this limitation in reuse/redistribution decisions. |
| Attribution metadata missing | **0** | Source/license and change-notice requirements are prepared for all 187 in the attribution CSV. Open Images rows include original author, title and landing URL; the final publication notice remains a later task. |
| Invalid box geometry or unreadable file | **0** | Stage 2B checks passed; Stage 2C rehashed all 187 images and the four raw archive/metadata files against their recorded SHA-256 values. |
| Specific annotation correction proposed | **9** | See table below; no correction was applied to raw or processed labels. |
| Targeted class-definition review | **130** | Includes 121 Construction-PPE images with source `vest` boxes and Open Images broad Helmet/Handgun/depiction or object-scope cases without a confirmed correction. These categories overlap within an image. |
| All-six-class completeness unverified | **187** | Contact sheets and selected full-resolution images were inspected, but no exhaustive per-image audit of every target object has been signed off. Missing labels must not become background. |
| Source group unverified | **187** | No verified video, camera-session or scene IDs in the acquired records. Provisional grouping is described below; no image is split-eligible. |
| Real/generated status unknown per image | **150** | Construction-PPE source says real and generated images are mixed but supplies no per-image flag in the acquired manifest. |
| Exact or pHash near-duplicate match within this 187-image pilot | **0 found** | No SHA-256 match and no 64-bit pHash Hamming distance ≤10 pair. This is a limited image-similarity check, not proof of independent scenes. |

These counts overlap. For example, an Open Images image with a wrong helmet box
has both a correction proposal and a rights clarification gate.

## Proposed annotation corrections

The following are **visual review proposals**, recorded in the structured
correction file with `applied=false` and `approved=false`. Full-resolution
inspection was performed for the illustration, cake decorations, football
helmet, long gun, antique handgun and modern pistol examples; other flags were
identified from contact sheets and should be checked at full resolution.
Dropping a wrong box does not automatically make an image usable: any remaining
target instances need complete labels, or the whole image must be rejected.

| Open Images ID | Observed issue | Proposed human action |
|---|---|---|
| `001997021f01f208` | Manga characters labeled Person | Remove two Person boxes; reject if no real target remains. |
| `004e21eb2e686f40` | Cake figures labeled Person | Remove Person box; reject if no real target remains. |
| `00a7655d4eabf186` | Baseball helmets labeled Helmet | Remove two Helmet boxes; add/review visible people if retaining. |
| `020c7b8e6d605069` | Football helmet labeled Helmet | Remove Helmet box; assess partial person. |
| `0543f1b76e391db9` | Hockey helmet labeled Helmet | Remove Helmet box; assess person. |
| `0684426d4c177038` | Roller-derby helmet labeled Helmet | Remove Helmet box; review all people for missing boxes. |
| `081e72dabebe235a` | Motorsport helmet labeled Helmet | Remove Helmet box; verify existing Person box. |
| `0c4b629e801e8181` | Break-action long gun labeled Handgun | Remove proposed Pistol box; reject if no target remains. |
| `45708b9bca9d7b8b` | Shotgun labeled Handgun | Remove proposed Pistol box; reject if no target remains. |

Additional targeted questions are recorded without assuming a correction:
`208de556fa323b60` is an antique short-barrel handgun candidate;
`0c87545ba250f18c` visibly contains a modern pistol; both need box and
replica/status checks. `15a16396c6dd9b84` mixes a physical backpack with
printed packaging depictions. `2ab108bbaa221cd7` needs backpack-versus-case
review. `19aaf40e64cdfb7b` needs knife-versus-utensil review. The 121
Construction-PPE vest-labeled images need the fixed `safety_vest` distinction:
a bright shirt or jacket alone is not a safety vest. These are not automatic
relabels. See [Open Images contact sheet](../reports/stage2b_v1/open_images_v7_contact_01.jpg)
and [Construction-PPE contact sheet](../reports/stage2b_v1/construction_ppe_contact_05.jpg).

## Source-group and duplicate handling

The Construction-PPE owner manifest supplies filename, original train/val
assignment, image hash and annotation count, but no source video, camera,
photographer or scene identifier. `source-*` filenames are opaque hashes and
`ultralytics-ppe-nov2026-*` names are serial-looking IDs, **not evidence of
independent sessions**. Until provenance is obtained or a scene-level visual
group audit is completed, the ledger proposes **one conservative group for all
150 Construction-PPE candidates**. This is intentionally restrictive for any
future split. It is not a claim they actually all come from one session.

Each Open Images record retains the original Flickr photo ID, landing URL and
photographer. The ledger provisionally groups by photographer account (37
distinct accounts in this pilot), not by Open Images ID alone. Same-scene images
from different accounts or reposts remain possible. These yield **38 provisional
group buckets** including the one Construction-PPE bucket, but **zero verified
independent capture groups**. Before splitting, a reviewer must inspect wider
similarity candidates, source URLs, background/people/setting and any original
sequence metadata; merge related images into one group. The absence of hash or
pHash matches here does not clear the group gate.

## Human review workflow

Open [review.html](../reports/stage2c_v1/review.html) locally in a browser.
It contains all 187 candidates in numbered order, shows the original full-size
image and draws the proposed source boxes in the six fixed colors. Each image
shows its separate gate evidence and proposed correction. A per-box control
lets the reviewer mark every source box keep, drop or resize and enter revised
normalized coordinates; a separate field records new boxes. The controls let the
reviewer choose pending, approve, correct or reject; record exact-image rights
evidence, attribution, box/class/completeness checks, a source group ID,
synthetic status, correction details, and notes. **All decisions start pending.**
The browser draft can be exported as JSON or CSV and imported later. Its
“approve” choice records a human review decision only; it does not copy any
image into `data/processed` or alter the immutable source labels. The exported
decision file must be reconciled with the ledger by `review_id` and checked
against all gates before a future release script is written.

Suggested review order: first the nine specific corrections; then the other
Open Images rights and taxonomy questions; then Construction-PPE vest cases;
finally the rest. Review each at full resolution for all six classes. A reviewer
should explicitly mark every source box keep/delete/resize, add missing target
boxes in a separate derived annotation file, and document an explicit true
negative if no target remains. A second reviewer should adjudicate ambiguous
helmets, vests, knives and firearms. Record rights evidence for the exact
Flickr photo and a justified source group; unresolved cases stay quarantined.

## Recoverability and limits

**0 are currently ready for acceptance.** The optimistic ceiling is **187
potentially recoverable candidates**, conditional on rights, full annotation
review, provenance grouping and human decisions. Nine already have concrete
correction proposals; several may become unusable if removing an invalid box
leaves no relevant target and the image is not intentionally selected as a
reviewed negative. The current evidence does not support a narrower numerical
yield estimate. If the Construction-PPE group cannot be separated into
independent scenes, many images may still be usable for training, but this
source cannot by itself supply independent validation and final test groups.

No new images, final split, training or Model A changes were made. Stage 2C
ends with the review ledger and human decision interface; acceptance remains
pending.
