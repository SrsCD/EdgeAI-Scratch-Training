"""Local-only six-class image/box review server. No model or network dependency."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from threading import Lock
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("STAGE3_REVIEW_BASE",str(ROOT / "data/interim/stage3b_review_v1")))
QUEUE = json.loads((BASE / "queue.json").read_text(encoding="utf-8"))["items"]
BY_ID = {r["id"]:r for r in QUEUE}
REVIEWS = BASE / "reviews"
REVIEWS.mkdir(exist_ok=True)
UI = Path(__file__).with_name("stage3b_review_ui.html")
LOCK = Lock()
CLASSES = ("person","helmet","safety_vest","backpack","knife","pistol")


def review_of(image_id):
    p=REVIEWS/f"{image_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def valid_boxes(boxes):
    if not isinstance(boxes,list) or len(boxes)>500:
        raise ValueError("boxes must be a list of at most 500")
    cleaned=[]
    for b in boxes:
        cid=b.get("class_id")
        if type(cid) is not int or cid not in range(6):
            raise ValueError("box class_id must be 0..5")
        vals=[float(b[k]) for k in ("cx","cy","w","h")]
        x,y,w,h=vals
        if not all(map(math.isfinite,vals)) or not (
            -1e-9<=x-w/2<x+w/2<=1+1e-9 and -1e-9<=y-h/2<y+h/2<=1+1e-9):
            raise ValueError("box coordinates must be finite normalized xywh inside image")
        cleaned.append({"class_id":cid,"cx":x,"cy":y,"w":w,"h":h})
    return cleaned


class Handler(BaseHTTPRequestHandler):
    def send(self,code,body,content_type="application/json"):
        data=body if isinstance(body,bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u=urlparse(self.path)
        if u.path=="/":
            self.send(200,UI.read_bytes(),"text/html; charset=utf-8")
        elif u.path=="/api/queue":
            saved={p.stem:json.loads(p.read_text(encoding="utf-8")) for p in REVIEWS.glob("P*.json")}
            out=[]
            for r in QUEUE:
                review=saved.get(r["id"])
                out.append({"id":r["id"],"source":r["source"],
                            "status":review["decision"] if review else "Pending",
                            "box_count":len(review["boxes"] if review else r["provisional_boxes"])})
            self.send(200,json.dumps(out).encode("utf-8"))
        elif u.path=="/api/item":
            image_id=parse_qs(u.query).get("id",[""])[0]
            if image_id not in BY_ID:
                self.send(404,b'{"error":"unknown image id"}')
                return
            self.send(200,json.dumps({"item":BY_ID[image_id],"review":review_of(image_id)}).encode("utf-8"))
        elif u.path.startswith("/image/"):
            image_id=u.path.rsplit("/",1)[-1]
            if image_id not in BY_ID:
                self.send(404,b'{"error":"unknown image id"}')
                return
            p=BASE/BY_ID[image_id]["asset_path"]
            self.send(200,p.read_bytes(),"image/jpeg")
        else:
            self.send(404,b'{"error":"unknown route"}')

    def do_POST(self):
        if urlparse(self.path).path!="/api/save":
            self.send(404,b'{"error":"unknown route"}')
            return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if length<=0 or length>1_000_000:
                raise ValueError("invalid request length")
            data=json.loads(self.rfile.read(length))
            image_id=data.get("id")
            if image_id not in BY_ID:
                raise ValueError("unknown image id")
            decision=data.get("decision")
            if decision not in {"Approve","Correct","Reject","Hold"}:
                raise ValueError("decision must be Approve, Correct, Reject or Hold")
            reviewer=str(data.get("reviewer","")).strip()
            if not reviewer:
                raise ValueError("reviewer name is required")
            boxes=valid_boxes(data.get("boxes"))
            notes=str(data.get("notes","")).strip()
            rights_evidence=str(data.get("rights_evidence","")).strip()
            group_id=str(data.get("final_group_id","")).strip()
            if len(notes)>4000 or len(rights_evidence)>4000 or len(group_id)>200:
                raise ValueError("text field too long")
            gates={k:data.get(k) is True for k in (
                "taxonomy_checked","six_class_checked","rights_verified","group_resolved","verified_negative")}
            if decision=="Approve":
                if not all(gates[k] for k in ("taxonomy_checked","six_class_checked","rights_verified","group_resolved")):
                    raise ValueError("Approve requires taxonomy, six-class, rights and group checks")
                if not rights_evidence or not group_id:
                    raise ValueError("Approve requires rights evidence and a final source-group ID")
                if not boxes and not gates["verified_negative"]:
                    raise ValueError("Empty labels require verified-negative confirmation")
            if decision=="Reject" and not notes:
                raise ValueError("Reject requires a reason in notes")
            record={"id":image_id,"source":BY_ID[image_id]["source"],"image_sha256":BY_ID[image_id]["image_sha256"],
                    "decision":decision,"reviewer":reviewer,"reviewed_at_utc":datetime.now(timezone.utc).isoformat(),
                    "boxes":boxes,"notes":notes,"rights_evidence":rights_evidence,"final_group_id":group_id,**gates}
            with LOCK:
                target=REVIEWS/f"{image_id}.json"
                tmp=REVIEWS/f"{image_id}.tmp"
                tmp.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
                tmp.replace(target)
                with (REVIEWS/"history.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record,ensure_ascii=False)+"\n")
            self.send(200,json.dumps({"ok":True,"id":image_id,"decision":decision}).encode("utf-8"))
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
            self.send(400,json.dumps({"error":str(e)}).encode("utf-8"))


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=8765)
    args=p.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Offline review: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)",flush=True)
    server.serve_forever()


if __name__=="__main__":main()
