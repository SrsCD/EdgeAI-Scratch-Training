"""Future guarded YOLO11n scratch run. Do not call before human review."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch
import yaml

from stage3_training_preflight import main as preflight

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as f:return sha256(f.read()).hexdigest()


def main():
    preflight()  # Raises before any model construction when data are unapproved.
    config_path=ROOT/"configs/train.yaml"
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["pretrained"] is not False or config["resume"] is not False or config["amp"] is not False:
        raise SystemExit("Scratch flags changed")
    if config["model"]!="configs/yolo11n.yaml" or not torch.cuda.is_available():
        raise SystemExit("Six-class architecture YAML or CUDA RTX 3050 is unavailable")
    gpu=torch.cuda.get_device_properties(0)
    if "3050" not in gpu.name:
        raise SystemExit(f"Unexpected CUDA device: {gpu.name}")
    from ultralytics import YOLO
    # Only architecture YAML is passed to YOLO; no pretrained .pt is loaded.
    model=YOLO(str(ROOT/config["model"]))
    if getattr(model,"ckpt",None) is not None:
        raise SystemExit("Unexpected checkpoint loaded before scratch training")
    args={k:v for k,v in config.items() if k!="model"}
    args["data"]=str(ROOT/config["data"])
    args["project"]=str(ROOT/config["project"])
    result=model.train(**args)
    save_dir=Path(model.trainer.save_dir)
    manifest=ROOT/"data/manifests/stage3b_pilot_v1_splits.csv"
    provenance={"initialization":"six_class_architecture_yaml_random_weights",
                "pretrained":False,"resume":False,"amp":False,"cuda_device":gpu.name,
                "cuda_total_memory_bytes":gpu.total_memory,"seed":config["seed"],
                "architecture_sha256":digest(ROOT/config["model"]),
                "train_config_sha256":digest(config_path),
                "data_yaml_sha256":digest(ROOT/config["data"]),
                "split_manifest_sha256":digest(manifest),
                "run_directory":str(save_dir)}
    (save_dir/"scratch_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":main()
