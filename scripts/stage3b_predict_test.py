"""Future frozen-test prediction/timing export from this project's scratch run."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import torch

from stage3_training_preflight import main as preflight

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"data/manifests/stage3b_pilot_v1_splits.csv"
OUT=ROOT/"reports/evaluation_inputs"


def digest(path):
    with path.open("rb") as f:return sha256(f.read()).hexdigest()


def write_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--experimental-stage3c",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True,help="best.pt from this scratch run")
    p.add_argument("--split",choices=["test","val"],default="test")
    args=p.parse_args()
    if args.experimental_stage3c:
        from stage3c_preflight import main as experimental_preflight
        experimental_preflight()
        manifest=ROOT/"data/experimental/stage3c_v1/ledger.csv"
        image_root=ROOT/"data/experimental/stage3c_v1/images"
        output=ROOT/"reports/evaluation_inputs/stage3c"
    else:
        preflight()
        manifest=MANIFEST
        image_root=ROOT/"data/processed/images"
        output=OUT
    checkpoint=args.checkpoint.resolve()
    if not checkpoint.is_file() or ROOT/"runs" not in checkpoint.parents:
        raise SystemExit("Checkpoint must be an existing file inside this project's runs directory")
    provenance=checkpoint.parent.parent/("stage4a_provenance.json" if args.experimental_stage3c else "scratch_provenance.json")
    if not provenance.is_file():
        raise SystemExit("Scratch-run provenance record missing")
    info=json.loads(provenance.read_text(encoding="utf-8"))
    expected_init="architecture YAML random weights" if args.experimental_stage3c else "six_class_architecture_yaml_random_weights"
    if info.get("initialization")!=expected_init or info.get("pretrained") is not False or info.get("resume") is not False:
        raise SystemExit("Checkpoint is not evidenced as this project's scratch initialization")
    if args.experimental_stage3c and (info.get("mode")!="full" or info.get("source")!="data/experimental/stage3c_v1"
                                     or info.get("test_used_for_training_or_selection") is not False):
        raise SystemExit("Stage 4A full scratch provenance is incomplete")
    with manifest.open(newline="",encoding="utf-8-sig") as f:
        rows=[r for r in csv.DictReader(f) if r["assigned_split" if args.experimental_stage3c else "split"]==args.split
              and (not args.experimental_stage3c or r["status"]=="included")]
    ids=[r["id"] for r in rows]
    image_dir=image_root/args.split
    images={p.stem:p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}}
    if set(ids)!=set(images):raise SystemExit("Processed image IDs differ from frozen manifest")
    output.mkdir(parents=True,exist_ok=True)
    prefix="stage3c_experimental" if args.experimental_stage3c else "stage3b_pilot_v1"
    pred_path=output/f"{prefix}_{args.split}_predictions.csv"
    timing_path=output/f"{prefix}_{args.split}_timings.csv"
    if pred_path.exists() or timing_path.exists():
        raise SystemExit("Frozen split predictions already exported; do not silently rerun")
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))  # Evaluation checkpoint, never scratch initialization.
    if tuple(model.names[i] for i in range(6))!=("person","helmet","safety_vest","backpack","knife","pistol"):
        raise SystemExit("Checkpoint class order differs from fixed six-class contract")
    # Warm up CUDA without writing evaluation rows.
    warm=images[ids[0]]
    for _ in range(5):model.predict(source=str(warm),imgsz=640,device=0,conf=.001,iou=.7,verbose=False)
    torch.cuda.synchronize(0)
    preds=[];times=[]
    for image_id in ids:
        result=model.predict(source=str(images[image_id]),imgsz=640,device=0,conf=.001,iou=.7,verbose=False)[0]
        torch.cuda.synchronize(0)
        latency=float(result.speed["inference"])
        times.append({"image_id":image_id,"latency_ms":f"{latency:.6f}"})
        if result.boxes is None:continue
        for b in result.boxes:
            x,y,w,h=(float(v) for v in b.xywhn[0].tolist())
            preds.append({"image_id":image_id,"class_id":int(b.cls[0]),
                          "confidence":f"{float(b.conf[0]):.8f}","cx":f"{x:.10f}",
                          "cy":f"{y:.10f}","w":f"{w:.10f}","h":f"{h:.10f}"})
    write_csv(pred_path,preds,["image_id","class_id","confidence","cx","cy","w","h"])
    write_csv(timing_path,times,["image_id","latency_ms"])
    (output/f"{prefix}_{args.split}_provenance.json").write_text(json.dumps({
        "checkpoint_sha256":digest(checkpoint),"scratch_provenance_path":str(provenance),
        "evaluation_label_status":"EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED" if args.experimental_stage3c else "approved",
        "split":args.split,"image_count":len(ids),"detection_count":len(preds),
        "imgsz":640,"confidence_export_threshold":.001,"nms_iou":.7,
        "latency":"Ultralytics result.speed inference milliseconds, batch one after five warmups, CUDA synchronized"},indent=2)+"\n",encoding="utf-8")
    print(f"Exported {len(preds)} detections for {len(ids)} {args.split} images")


if __name__=="__main__":main()
