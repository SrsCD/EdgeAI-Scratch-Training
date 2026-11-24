# Dataset preparation contract (future stage)

Target approximately 10,000 reviewed images with YOLO detection labels. Class IDs
are fixed: 0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol.
No sources have been acquired or approved in Stage 1.

For every source, record provenance and image/annotation licenses separately in
data/attribution/sources.csv. Retain license texts and attribution requirements;
review permitted academic use, derivatives and redistribution before inclusion.
Keep originals in data/raw, conversions in data/interim, and reviewed labels in
data/processed. Do not import data or weights from EdgeAI-Surveillance.

Validate image decoding, dimensions, class IDs, finite normalized xywh values,
positive box sizes, box bounds, and image/label pairing. Review completeness,
occlusion, small objects, class ambiguity and class balance manually. Record
intentional negative images explicitly; missing annotations are not negatives.

Compute SHA-256 for exact duplicates, perceptual hashes for candidate near
duplicates, and manually review near matches. Cluster related images BEFORE
splitting. Keep frames from one video, capture session, scene, or duplicate
cluster in the same split; account for identities/cameras where relevant.
Balance multi-label class coverage at group level with seed 42. Aim for
70% train / 20% test / 10% val (7,000 / 2,000 / 1,000 at 10,000 images);
group integrity takes priority over exact counts. Document deviations.

Audit all split pairs for path, content-hash and near-duplicate overlap; record
results and source/class counts. Version manifests and their SHA-256 hashes.
Freeze test membership before training, restrict test access, and use only train
and val for model selection, thresholds, augmentations and error analysis.
Test labels/images must not enter training caches or development reports.
Run the final test once after checkpoint and evaluation protocol are frozen.
