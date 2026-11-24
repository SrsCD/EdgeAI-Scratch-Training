"""Pure object-detection scoring for the fixed six-class contract."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean, median

NAMES = ("person", "helmet", "safety_vest", "backpack", "knife", "pistol")


def iou(a, b):
    """IoU of normalized (cx,cy,w,h) boxes."""
    ax1,ay1,ax2,ay2=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx1,by1,bx2,by2=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.0


def check_box(box):
    if len(box)!=4 or not all(isfinite(v) for v in box):
        raise ValueError("box must have four finite normalized values")
    x,y,w,h=box
    # Ultralytics float32 xywhn can round a clipped edge past 0/1 by ~1e-7.
    if not (-1e-6<=x-w/2<x+w/2<=1+1e-6 and -1e-6<=y-h/2<y+h/2<=1+1e-6):
        raise ValueError("box exceeds image")


def class_match(ground_truth, predictions, class_id, iou_threshold=.5, confidence_threshold=0.0):
    """Confidence-ordered one-to-one matching within a class.

    Returns events for predictions above confidence threshold and false-negative
    count. A repeated detection of one ground-truth object is a false positive.
    """
    truth={image_id:[g for g in boxes if g["class_id"]==class_id]
           for image_id,boxes in ground_truth.items()}
    used=defaultdict(set)
    preds=sorted((p for p in predictions if p["class_id"]==class_id and
                  p["confidence"]>=confidence_threshold),key=lambda p:-p["confidence"])
    events=[]
    for p in preds:
        pool=truth.get(p["image_id"],[])
        candidates=[(iou(p["box"],g["box"]),i) for i,g in enumerate(pool)
                    if i not in used[p["image_id"]]]
        best=max(candidates,default=(0.0,-1))
        matched=best[1]>=0 and best[0]>=iou_threshold
        if matched:used[p["image_id"]].add(best[1])
        events.append({"image_id":p["image_id"],"class_id":class_id,
                       "confidence":p["confidence"],"tp":int(matched),
                       "fp":int(not matched),"matched_iou":best[0] if matched else None})
    fn=sum(len(v)-len(used[k]) for k,v in truth.items())
    return events,fn


def ap_for_class(ground_truth,predictions,class_id,iou_threshold):
    support=sum(g["class_id"]==class_id for boxes in ground_truth.values() for g in boxes)
    if support==0:return None,[],[]
    events,_=class_match(ground_truth,predictions,class_id,iou_threshold,0.0)
    tp=fp=0
    recalls=[];precisions=[]
    for event in events:
        tp+=event["tp"];fp+=event["fp"]
        recalls.append(tp/support)
        precisions.append(tp/(tp+fp))
    # COCO-style 101 recall-point interpolation at a single IoU threshold.
    area=sum(max((p for r,p in zip(recalls,precisions) if r>=k/100),default=0.0)
             for k in range(101))/101
    return area,recalls,precisions


def confusion_matrix(ground_truth,predictions,confidence_threshold=.25,iou_threshold=.5):
    """Rows: ground truth 0..5/background; columns: predicted 0..5/background."""
    matrix=[[0 for _ in range(7)] for _ in range(7)]
    by_image=defaultdict(list)
    for p in predictions:
        if p["confidence"]>=confidence_threshold:by_image[p["image_id"]].append(p)
    for image_id,gt in ground_truth.items():
        used=set()
        for p in sorted(by_image[image_id],key=lambda x:-x["confidence"]):
            best=max(((iou(p["box"],g["box"]),j) for j,g in enumerate(gt) if j not in used),default=(0,-1))
            if best[1]>=0 and best[0]>=iou_threshold:
                used.add(best[1]);matrix[gt[best[1]]["class_id"]][p["class_id"]]+=1
            else:matrix[6][p["class_id"]]+=1
        for j,g in enumerate(gt):
            if j not in used:matrix[g["class_id"]][6]+=1
    return matrix


def evaluate(ground_truth,predictions,timings_ms=None,confidence_threshold=.25,iou_threshold=.5):
    for image_id,gt in ground_truth.items():
        for g in gt:
            if g["class_id"] not in range(6):raise ValueError(f"invalid GT class in {image_id}")
            check_box(g["box"])
    for p in predictions:
        if p["image_id"] not in ground_truth:raise ValueError("prediction outside evaluated split")
        if p["class_id"] not in range(6) or not (0<=p["confidence"]<=1):raise ValueError("invalid prediction class/confidence")
        check_box(p["box"])
    per_class=[]; all_events=[]; matched=[]
    for cid,name in enumerate(NAMES):
        events,fn=class_match(ground_truth,predictions,cid,iou_threshold,confidence_threshold)
        tp=sum(e["tp"] for e in events);fp=sum(e["fp"] for e in events)
        support=tp+fn
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        ap50,pr_recall,pr_precision=ap_for_class(ground_truth,predictions,cid,.5)
        ap75,_,_=ap_for_class(ground_truth,predictions,cid,.75)
        aps=[ap_for_class(ground_truth,predictions,cid,round(.5+.05*k,2))[0] for k in range(10)]
        per_class.append({"class_id":cid,"class_name":name,"support_objects":support,
                          "tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":f1,
                          "ap50":ap50,"ap75":ap75,"ap50_95":mean(aps) if ap50 is not None else None,
                          "pr_recall":pr_recall,"pr_precision":pr_precision})
        all_events.extend(events)
        matched.extend(e["matched_iou"] for e in events if e["matched_iou"] is not None)
    tp=sum(r["tp"] for r in per_class);fp=sum(r["fp"] for r in per_class);fn=sum(r["fn"] for r in per_class)
    timing=None
    if timings_ms is not None:
        if set(timings_ms)!=set(ground_truth) or any(v<=0 or not isfinite(v) for v in timings_ms.values()):
            raise ValueError("timings must contain one positive finite latency per evaluated image")
        vals=list(timings_ms.values())
        timing={"mean_latency_ms":mean(vals),"median_latency_ms":median(vals),
                "fps_model_only":1000*len(vals)/sum(vals),"image_count":len(vals)}
    matrix=confusion_matrix(ground_truth,predictions,confidence_threshold,iou_threshold)
    normalized=[[v/sum(row) if sum(row) else None for v in row] for row in matrix]
    ap50=[r["ap50"] for r in per_class if r["ap50"] is not None]
    ap5095=[r["ap50_95"] for r in per_class if r["ap50_95"] is not None]
    curves=[]
    for k in range(101):
        threshold=k/100
        events=[];fns=[]
        for cid in range(6):
            ev,miss=class_match(ground_truth,predictions,cid,iou_threshold,threshold)
            events.extend(ev);fns.append(miss)
        t=sum(e["tp"] for e in events);f=sum(e["fp"] for e in events);n=sum(fns)
        curves.append({"confidence_threshold":threshold,"precision":t/(t+f) if t+f else None,
                       "recall":t/(t+n) if t+n else None,"f1":2*t/(2*t+f+n) if 2*t+f+n else None})
    return {"thresholds":{"confidence":confidence_threshold,"iou_match":iou_threshold},
            "overall":{"tp":tp,"fp":fp,"fn":fn,
                       "precision":tp/(tp+fp) if tp+fp else None,
                       "recall":tp/(tp+fn) if tp+fn else None,
                       "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                       "map50":mean(ap50) if ap50 else None,
                       "map50_95":mean(ap5095) if ap5095 else None,
                       "mean_matched_iou":mean(matched) if matched else None,
                       "timing":timing},"per_class":per_class,"prediction_events":all_events,
            "confusion_raw":matrix,"confusion_normalized":normalized,"confidence_curves":curves}
