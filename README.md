# EdgeAI Scratch Training

An experimental six-class YOLO11n object detector trained from random initialization for surveillance-style demonstrations. This repository contains the source, reproducibility notes, test code, aggregate evaluation artifacts, and one scratch-trained checkpoint distributed through Git LFS. It is independent of EdgeAI-Surveillance (Model A).

**The results measure agreement with provisional annotations, not independently verified real-world accuracy.** Dataset image rights, annotation completeness, and some scene-group independence remain unresolved. Do not use predictions as evidence that an object or threat is present.

## Classes and experiment

| ID | Class |
| ---: | --- |
| 0 | person |
| 1 | helmet |
| 2 | safety_vest |
| 3 | backpack |
| 4 | knife |
| 5 | pistol |

The experimental dataset has **7,680 image/label pairs**: **5,417 train (70.5%)**, **1,473 test (19.2%)**, and **790 validation (10.3%)**. The split uses provisional source groups with seed 42; it is not proven independent. The original candidate ledger covers 9,441 items. Details and class counts are in [the dataset report](documentation/stage3c_experimental_dataset_report.md).

The model was constructed from [`configs/yolo11n.yaml`](configs/yolo11n.yaml), with `pretrained=False`, `resume=False`, and no `.pt` initialization. The final batch-16 run used 640-pixel images, seed 42, AMP off, and validation for checkpoint selection. It stopped after **12 completed epochs** of a planned 150. See [the training report](documentation/stage4a_scratch_training_report.md) and [epoch results](runs/scratch_yolo11n_stage3c_experimental_b16/results.csv). The root `yolo11n.pt` is a pretrained file and is explicitly excluded; it is **not** this model.

## Setup

Use 64-bit Python 3.13, Git LFS, a webcam for live capture, and enough disk space for your own licensed dataset. The recorded training setup was Windows 11, an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM, PyTorch 2.9.1+cu128, and Ultralytics 8.3.233. Stage 5G inference also supports CPU. Training and the saved test exporter were originally written for CUDA; training on other hardware needs the device and batch settings adjusted and must create a new run directory.

From PowerShell in the cloned repository:

```powershell
git lfs install
git lfs pull
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CUDA 12.8 example; choose the supported PyTorch wheel for your driver/platform.
.\.venv\Scripts\python.exe -m pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

For CPU inference, install a matching CPU PyTorch/torchvision pair from the [official PyTorch installer](https://pytorch.org/get-started/locally/) before `requirements.txt`. On Linux or macOS, use `python3 -m venv .venv` and `.venv/bin/python` in the commands below. The recorded resolved Windows CUDA environment is in `requirements-lock.txt`; it is a reference, not a cross-platform lockfile.

### Checkpoint

The **only distributed model** is [`models/distribution/best.pt`](models/distribution/best.pt) (**10,701,058 bytes**, SHA-256 `ef582bf60d27ba640ff9a914de2b7e01fa203fb403e66476048a135e95e58276`). It is a privacy-redacted copy of the exact 12-epoch scratch checkpoint: all 499 saved EMA model tensors were checked for equality with the local original. Only embedded local-path metadata changed. The original training artifact (**10,761,935 bytes**, SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`) remains on the training machine and is excluded from Git. After `git lfs pull`, install the public copy to the application path:

```powershell
.\.venv\Scripts\python.exe scripts\install_checkpoint.py
(Get-FileHash runs\scratch_yolo11n_stage3c_experimental_b16\weights\best.pt -Algorithm SHA256).Hash
```

The printed hash must match the public digest above (or the original digest on the training machine). If the distribution file is a small LFS pointer, run `git lfs pull`. The installer never replaces an existing valid local original. Stage 5G checks the digest before loading and refuses other weights. Do not put pretrained `yolo11n.pt` at this path.

## Run the applications

Live webcam detection, with local SQLite event logging:

```powershell
.\.venv\Scripts\python.exe scripts\stage5g_live_webcam.py --camera-index 0 --width 1280 --height 720
```

In another terminal, open the local history dashboard at `http://127.0.0.1:8765/`:

```powershell
.\.venv\Scripts\python.exe scripts\stage5h_history.py --port 8765
```

The database defaults to `runs/stage5h/events.sqlite3` and remains local. The dashboard supports class/date filters and CSV export. Optional `--save-evidence` saves annotated event JPEGs; handle them as private recordings. Optional local console notices use `--notify-class backpack` (repeat per class). They do not send network alerts. See [Stage 5H details](docs/stage5h_logging.md).

To record annotated webcam video, add `--record runs\stage5g\my_demo.mp4` to the live command. Use a fresh filename; existing recordings are not overwritten. Press `Q` to quit; `+`/`-` adjusts confidence. The synthetic `--self-test` checks inference without opening a camera or writing events. Webcam, evidence, databases, and recordings are ignored by Git.

## Dataset and reproducibility

The image and label files are **not distributed**. The five contributing Stage 3C sources are listed below. These are provenance links, **not** download instructions or redistribution permission. The local detailed rights register is `data/interim/stage3b_review_v1/source_rights_register.csv`. Dataset-level claims do not settle rights to individual underlying photos.

| Source | Recorded publisher-level claim | Unresolved issue |
| --- | --- | --- |
| [Construction Site Safety v30](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30) | CC BY 4.0 | Individual photo rights |
| [Construction PPE v1](https://universe.roboflow.com/gaos-workspace/construction-ppe-qofi4/dataset/1) | CC BY 4.0 | Product and web photo rights |
| [Backpack v1](https://universe.roboflow.com/vrk2/backpack-y9y87) | CC BY 4.0 | Retail product photo rights |
| [Knifes v1](https://universe.roboflow.com/tuan-tran/knifes-lfu4v/dataset/1) | CC BY 4.0 claim in archive | Live page and individual photo rights |
| [Pistols v1](https://universe.roboflow.com/joseph-nelson/pistols/dataset/1) | Public-domain claim | Third-party imagery and watermarks |

[The acquisition report](documentation/stage2b_acquisition_report.md) additionally discusses a separate pilot using Ultralytics Construction-PPE and Open Images V7. Source claims vary, underlying image rights remain unresolved, and these data must not be assumed freely redistributable or available for download. Obtain access and permissions independently before preparing local data. The [dataset protocol](docs/dataset_protocol.md) explains review, attribution, leakage checks, and labeling.

To reproduce this *specific* Stage 3C experiment, prepare `data/experimental/stage3c_v1/` locally with `STATUS.txt`, `ledger.csv`, `split_stats.csv`, `images/{train,val,test}/`, and `labels/{train,val,test}/`. Images and YOLO TXT labels must pair by stem. The ledger must retain all 9,441 candidate records and the hashes/group assignments expected by `scripts/stage3c_preflight.py`; a different dataset is a **new experiment** and needs its own config, ledger, and validation protocol. `configs/stage3c_data.yaml` points to this relative directory. Dataset preparation helpers are in `scripts/stage2*` and `scripts/stage3*`; do not run acquisition or training until source terms and local paths are reviewed.

With the original local dataset and compatible CUDA setup, run the mechanical preflight:

```powershell
.\.venv\Scripts\python.exe scripts\stage3c_preflight.py
```

The historical training command was `scripts/stage3c_train_scratch.py --batch 16`; it refuses an existing run directory, including the distributed final run. To retrain, use a fresh checkout/run name and review the hardware assumptions in `configs/stage3c_train.yaml` and the launcher. The saved test prediction exporter and evaluator are `scripts/stage3b_predict_test.py --experimental-stage3c` and `scripts/stage3b_evaluate.py --experimental-stage3c`; both require the local dataset, and the evaluator refuses to overwrite existing output. Do not rerun the held-out test to tune the model.

## Evaluation

At confidence 0.25 and same-class matching IoU 0.50, the 1,473-image provisional test split produced **TP 1,019, FP 700, FN 1,055; precision 0.5928, recall 0.4913, F1 0.5373, mAP@50 0.4427, mAP@50–95 0.2903**. Mean matched-box IoU was 0.8275. Model-only mean latency was 9.156 ms/image; it excludes loading, preprocessing, and NMS. Person and knife recall were especially low. See [the evaluation report](documentation/stage4b_experimental_evaluation_report.md).

The 12 reported measures are **TP, FP, FN, precision, recall, F1, AP@50, AP@75, mAP@50, mAP@50–95, matched-box IoU, and inference latency/FPS**. Per-class values and operating curves are in [`reports/evaluation/stage3c_experimental_test/`](reports/evaluation/stage3c_experimental_test/). [Raw confusion matrix](reports/evaluation/stage3c_experimental_test/confusion_raw.png), [normalized confusion matrix](reports/evaluation/stage3c_experimental_test/confusion_normalized.png), [PR curve](reports/evaluation/stage3c_experimental_test/precision_recall.png), and [loss curves](reports/evaluation/stage3c_experimental_test/training_validation_losses.png) are included. Image-level accuracy and specificity were not calculated because verified image-level negatives were unavailable.

Genuine model prediction examples exist locally in `reports/evaluation/stage3c_experimental_test/prediction_examples/`, and real webcam recordings exist locally in `runs/stage5g/`. They are excluded pending image rights and participant/privacy review. We will add screenshots only when publication rights are documented; synthetic self-test frames are not detection examples.

## Tests and troubleshooting

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\stage5g_live_webcam.py --self-test
```

If the camera will not open, close other camera apps, check OS camera permissions, and try `--camera-index 1`. For slow inference, use `--width 640 --height 480` or the CPU fallback when CUDA is unavailable. For codec errors, omit `--record` or use a writable MP4 destination. If predictions look poor, check lighting and distance; adjusting confidence changes the display and does not establish accuracy. If a checkpoint hash mismatch appears, restore the exact LFS artifact and verify the digest above.

## Collaboration and licenses

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [the teammate setup guide](docs/teammate-setup.md) before opening a pull request. Keep datasets, personal capture evidence, databases, and other weights out of Git. State whether a proposed result used provisional or reviewed labels. [The publishing checklist](docs/github-publishing-checklist.md) records the review gates.

**License recommendation:** review Ultralytics' AGPL-3.0 obligations for the derived architecture YAML and runtime before choosing a repository code license; an AGPL-3.0-compatible choice is prudent. No project `LICENSE` is asserted yet, and no code license would grant rights to third-party datasets, photos, or videos. See [license notes](docs/license-recommendation.md).
