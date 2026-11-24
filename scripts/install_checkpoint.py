"""Install the privacy-redacted scratch checkpoint at the app's expected path."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "models/distribution/best.pt"
TARGET = ROOT / "runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt"
PUBLIC_PROVENANCE = ROOT / "models/distribution/stage4a_provenance.json"
TARGET_PROVENANCE = TARGET.parent.parent / "stage4a_provenance.json"
PUBLIC_HASH = "ef582bf60d27ba640ff9a914de2b7e01fa203fb403e66476048a135e95e58276"
ORIGINAL_HASH = "1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if not PUBLIC.is_file() or digest(PUBLIC) != PUBLIC_HASH:
        raise SystemExit("Public checkpoint missing or invalid; run git lfs pull")
    if TARGET.exists():
        if digest(TARGET) not in {PUBLIC_HASH, ORIGINAL_HASH}:
            raise SystemExit(f"Refusing to replace existing checkpoint: {TARGET}")
        print("Verified checkpoint already installed; leaving it unchanged.")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PUBLIC, TARGET)
        if digest(TARGET) != PUBLIC_HASH:
            raise SystemExit("Copied checkpoint failed SHA-256 verification")
        print(f"Installed public scratch checkpoint at {TARGET}")
    if not TARGET_PROVENANCE.exists():
        shutil.copyfile(PUBLIC_PROVENANCE, TARGET_PROVENANCE)
        print("Installed checkpoint provenance")


if __name__ == "__main__":
    main()
