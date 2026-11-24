# Teammate setup

1. Clone the published GitHub repository with Git LFS installed, then run `git lfs pull`.
2. Create a Python 3.13 virtual environment. Install a compatible PyTorch/torchvision pair for your CPU or CUDA platform, then `pip install -r requirements.txt` and `pip check`. Exact PowerShell commands are in the [README](../README.md#setup).
3. Verify the Git LFS file `models/distribution/best.pt` against SHA-256 `ef582bf60d27ba640ff9a914de2b7e01fa203fb403e66476048a135e95e58276`. Run `python scripts/install_checkpoint.py` to copy it to `runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt`. It contains the exact scratch-trained model tensors with local path metadata redacted; root `yolo11n.pt` is pretrained and not distributed.
4. Run `python scripts/stage5g_live_webcam.py --self-test`, then the webcam command in the README. Run `python scripts/stage5h_history.py --port 8765` to view local event history.
5. If working on training or evaluation, obtain a permitted dataset independently. Prepare the local structure and ledger described in the README, run `scripts/stage3c_preflight.py`, and use a new experiment name. The original source images and annotations are not in GitHub.
6. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before a pull request.

The Stage 3C preflight expects the original 9,441-row candidate ledger and exact image hashes; a new dataset will need a new config and preflight contract. The included aggregate results can be inspected without dataset access.
