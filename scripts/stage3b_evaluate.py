"""Score frozen reviewed detections; never run on synthetic fixtures as results."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MPL_CACHE=ROOT/".cache/matplotlib"
MPL_CACHE.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(MPL_CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0,str(ROOT/"src"))
from edgeai_scratch.detection_eval import NAMES,evaluate  # noqa: E402

MANIFEST=ROOT/"data/manifests/stage3b_pilot_v1_splits.csv"
FREEZE=ROOT/"data/manifests/stage3b_pilot_v1_test_freeze.json"
PROCESSED=ROOT/"data/processed"


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def save_matrix(path,matrix,status):
    rows=[]
    for i,row in enumerate(matrix):
        rows.append({"annotation_status":status,"ground_truth":NAMES[i] if i<6 else "background",
                     **{(NAMES[j] if j<6 else "background"):v for j,v in enumerate(row)}})
    write_csv(path,rows,list(rows[0]))


def plot_matrix(matrix,path,title):
    fig,ax=plt.subplots(figsize=(8,7))
    vals=[[0 if v is None else v for v in row] for row in matrix]
    im=ax.imshow(vals,cmap="Blues")
    ax.set_xticks(range(7),list(NAMES)+["background"],rotation=45,ha="right")
    ax.set_yticks(range(7),list(NAMES)+["background"])
    ax.set_xlabel("Predicted");ax.set_ylabel("Ground truth");ax.set_title(title)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_curves(result,out):
    fig,ax=plt.subplots(figsize=(7,5))
    for r in result["per_class"]:
        if r["pr_recall"]:ax.plot(r["pr_recall"],r["pr_precision"],label=f"{r['class_name']} AP50={r['ap50']:.3f}")
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1),title="Precision–recall at IoU 0.50")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"precision_recall.png",dpi=160);plt.close(fig)
    curves=result["confidence_curves"]
    for key,title in [("f1","F1"),("precision","Precision"),("recall","Recall")]:
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot([r["confidence_threshold"] for r in curves],[r[key] if r[key] is not None else float("nan") for r in curves])
        ax.set(xlabel="Confidence threshold",ylabel=title,xlim=(0,1),ylim=(0,1),title=f"{title}–confidence")
        fig.tight_layout();fig.savefig(out/f"{key}_confidence.png",dpi=160);plt.close(fig)


def plot_losses(path,out):
    rows=read_csv(path)
    if not rows:return False
    columns=[k for k in rows[0] if "loss" in k.lower()]
    if not columns:return False
    fig,ax=plt.subplots(figsize=(9,5))
    for key in columns:
        try:vals=[float(r[key]) for r in rows]
        except (ValueError,KeyError):continue
        ax.plot(range(1,len(vals)+1),vals,label=key.strip())
    ax.set(xlabel="Epoch",ylabel="Loss",title="Training and validation losses")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"training_validation_losses.png",dpi=160);plt.close(fig)
    return True


def examples(ids,ground_truth,predictions,split,out):
    ex=out/"prediction_examples";ex.mkdir()
    images={p.stem:p for p in (PROCESSED/"images"/split).iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}}
    for image_id in ids[:12]:
        if image_id not in images:continue
        im=Image.open(images[image_id]).convert("RGB")
        draw=ImageDraw.Draw(im)
        for g in ground_truth[image_id]:
            x,y,w,h=g["box"];rect=((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height)
            draw.rectangle(rect,outline="#00df76",width=3)
            draw.text((rect[0],rect[1]),"GT "+NAMES[g["class_id"]],fill="#00df76")
        for p in predictions:
            if p["image_id"]!=image_id or p["confidence"]<.25:continue
            x,y,w,h=p["box"];rect=((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height)
            draw.rectangle(rect,outline="#f52944",width=2)
            draw.text((rect[0],rect[1]+14),f"P {NAMES[p['class_id']]} {p['confidence']:.2f}",fill="#f52944")
        im.save(ex/f"{image_id}.jpg",quality=90)


def main():
    global MANIFEST,PROCESSED
    p=argparse.ArgumentParser()
    p.add_argument("--experimental-stage3c",action="store_true",help="Evaluate provisional Stage 3C labels, never claim verified test accuracy")
    p.add_argument("--predictions",type=Path,required=True,help="CSV: image_id,class_id,confidence,cx,cy,w,h")
    p.add_argument("--timings",type=Path,required=True,help="CSV: image_id,latency_ms; synchronized model-only batch-one timing")
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--split",choices=["test","val"],default="test")
    p.add_argument("--training-results",type=Path)
    p.add_argument("--confidence",type=float,default=.25)
    p.add_argument("--iou",type=float,default=.5)
    args=p.parse_args()
    if args.experimental_stage3c:
        MANIFEST=ROOT/"data/experimental/stage3c_v1/ledger.csv"
        PROCESSED=ROOT/"data/experimental/stage3c_v1"
    if not MANIFEST.is_file() or (not args.experimental_stage3c and not FREEZE.is_file()):
        raise SystemExit("Frozen approved pilot manifest is required before evaluation")
    if len(args.checkpoint_sha256)!=64 or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256.lower()):
        raise SystemExit("Provide the actual scratch checkpoint SHA-256")
    freeze=json.loads(FREEZE.read_text(encoding="utf-8")) if not args.experimental_stage3c else None
    with MANIFEST.open("rb") as f: manifest_hash=sha256(f.read()).hexdigest()
    if freeze and manifest_hash!=freeze["split_manifest_sha256"]:
        raise SystemExit("Frozen split manifest hash changed")
    rows=[r for r in read_csv(MANIFEST) if r["assigned_split" if args.experimental_stage3c else "split"]==args.split
          and (not args.experimental_stage3c or r["status"]=="included")]
    if args.split=="test" and not args.experimental_stage3c:
        test_rows=sorted((r["id"],r["sha256"],r["final_group_id"]) for r in rows)
        test_digest=sha256(json.dumps(test_rows,sort_keys=True).encode()).hexdigest()
        if freeze.get("test_rows_sha256")!=test_digest or freeze.get("test_rows")!=[list(r) for r in test_rows]:
            raise SystemExit("Frozen test membership changed")
    ids=[r["id"] for r in rows]
    gt={}
    for image_id in ids:
        label=PROCESSED/"labels"/args.split/f"{image_id}.txt"
        boxes=[]
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            parts=line.split()
            if not parts:continue
            if len(parts)!=5:raise ValueError(f"invalid label in {label}")
            boxes.append({"class_id":int(parts[0]),"box":tuple(map(float,parts[1:]))})
        gt[image_id]=boxes
    raw_preds=read_csv(args.predictions)
    preds=[{"image_id":r["image_id"],"class_id":int(r["class_id"]),
            "confidence":float(r["confidence"]),"box":tuple(float(r[k]) for k in ("cx","cy","w","h"))}
           for r in raw_preds]
    timings={r["image_id"]:float(r["latency_ms"]) for r in read_csv(args.timings)}
    if len(timings)!=len(read_csv(args.timings)):
        raise ValueError("duplicate timing image IDs")
    result=evaluate(gt,preds,timings,args.confidence,args.iou)
    status="EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED; provisional annotations" if args.experimental_stage3c else "approved reviewed annotations"
    result["annotation_status"]=status
    out=ROOT/"reports/evaluation"/(f"stage3c_experimental_{args.split}" if args.experimental_stage3c else f"stage3b_pilot_v1_{args.split}")
    if out.exists():raise SystemExit(f"Evaluation output already exists; frozen {args.split} must not be silently rerun")
    out.mkdir(parents=True)
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    simple=[{"annotation_status":status,**{k:v for k,v in r.items() if not k.startswith("pr_")}} for r in result["per_class"]]
    write_csv(out/"per_class_metrics.csv",simple,list(simple[0]))
    write_csv(out/"per_prediction_confidence.csv",[
        {**r,"checkpoint_sha256":args.checkpoint_sha256,"split":args.split,"annotation_status":status} for r in raw_preds],
        (list(raw_preds[0]) if raw_preds else ["image_id","class_id","confidence","cx","cy","w","h"])+["checkpoint_sha256","split","annotation_status"])
    curves=[{"annotation_status":status,**r} for r in result["confidence_curves"]]
    write_csv(out/"confidence_curves.csv",curves,list(curves[0]))
    save_matrix(out/"confusion_raw.csv",result["confusion_raw"],status)
    save_matrix(out/"confusion_normalized.csv",result["confusion_normalized"],status)
    plot_matrix(result["confusion_raw"],out/"confusion_raw.png","Confusion matrix: "+status)
    plot_matrix(result["confusion_normalized"],out/"confusion_normalized.png","Normalized confusion matrix: "+status)
    plot_curves(result,out)
    has_losses=plot_losses(args.training_results,out) if args.training_results else False
    examples(ids,gt,preds,args.split,out)
    overall=result["overall"]
    report=f"""# {status}: {args.split} evaluation\n\nSplit ledger SHA-256: `{manifest_hash}`. Scratch checkpoint SHA-256: `{args.checkpoint_sha256}`. Evaluated {len(ids)} images with confidence ≥{args.confidence} and IoU matching ≥{args.iou}.\n\nTP={overall['tp']}; FP={overall['fp']}; FN={overall['fn']}; precision={overall['precision']}; recall={overall['recall']}; F1={overall['f1']}; mAP@50={overall['map50']}; mAP@50–95={overall['map50_95']}; mean matched-box IoU={overall['mean_matched_iou']}. Per-class AP@50/AP@75 and support are in `per_class_metrics.csv`. Mean latency/FPS: {overall['timing']}.\n\nPrecision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). AP uses 101-point interpolated precision at each recall level; mAP averages supported classes and IoU thresholds 0.50:0.05:0.95. Mean matched IoU averages one-to-one same-class matched boxes at the stated operating threshold. Timing CSV must be synchronized batch-one model-only latency; FPS=1000×images/sum(latency_ms). Confusion matrix rows are ground truth and columns are predictions, including background. Unsupported denominators are unavailable, not zero.\n\nTraining/validation loss plot generated: {has_losses}. Accuracy/specificity are omitted because detection true negatives are undefined; a separate verified image-level presence/absence task would be required. Test outputs must not be used to tune thresholds or checkpoints.\n"""
    (out/"evaluation_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"output":str(out),"overall":overall},indent=2))


if __name__=="__main__":main()
