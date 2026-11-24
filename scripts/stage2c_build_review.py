"""Build Stage 2C evidence ledger and offline human review UI.

Reads Stage 2B records and raw files. Never edits raw assets or model artifacts.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/stage2c_v1"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = json.loads((ROOT / "reports/stage2b_v1/audit.json").read_text(encoding="utf-8"))
OI = json.loads((ROOT / "data/raw/stage2b_v1/open_images_pilot_manifest.json").read_text(encoding="utf-8"))
CONST = json.loads((ROOT / "data/raw/stage2b_v1/construction_pilot_manifest.json").read_text(encoding="utf-8"))

# These are proposals grounded in full-image visual inspection, never edits or
# approvals. Target-class removals are still conditional on human adjudication.
TRIAGE = {
    "001997021f01f208": ("illustrated_people", "Remove the two Person boxes: the image is a manga drawing. Reject if no real target remains."),
    "004e21eb2e686f40": ("cake_figures", "Remove the Person box: the visible figures are cake decorations. Reject if no real target remains."),
    "00a7655d4eabf186": ("baseball_helmets", "Remove two Helmet boxes; inspect and add all visible Person boxes if retaining the image."),
    "020c7b8e6d605069": ("football_helmet", "Remove the Helmet box; inspect whether a Person box is required for the partial player."),
    "0543f1b76e391db9": ("hockey_helmet", "Remove the Helmet box; add/verify the visible Person box if retaining the image."),
    "0684426d4c177038": ("roller_derby", "Remove the sports Helmet box; review all visible people for missing Person boxes."),
    "081e72dabebe235a": ("racing_helmet", "Remove the motorsport Helmet box; verify the existing Person box."),
    "0c4b629e801e8181": ("long_gun", "Remove Pistol box: pictured break-action long gun is outside handgun definition. Reject if no target remains."),
    "45708b9bca9d7b8b": ("shotgun", "Remove Pistol box: pictured shotgun is outside handgun definition. Reject if no target remains."),
    "208de556fa323b60": ("antique_handgun", "Human check: antique short-barrel handgun appears within pistol scope; verify functional/replica status and box."),
    "0c87545ba250f18c": ("pistol_with_magazines", "Human check: pistol is visible; verify source box and any other target objects."),
    "15a16396c6dd9b84": ("packaging_depiction", "Separate physical backpack from printed depiction; remove depiction boxes if mapped to targets."),
    "2ab108bbaa221cd7": ("bag_scope", "Verify that the black bag is a backpack, not luggage or a carrying case."),
    "19aaf40e64cdfb7b": ("knife_scope", "Verify that crossed light-colored objects are knives rather than utensils or letter openers."),
}


def source_facts(record, oi_by_id, const_by_id):
    if record["source"] == "construction_ppe":
        r = const_by_id[record["id"]]
        return {
            "publisher": "Ultralytics", "source_version": "Construction-PPE, November 2026",
            "license": "CC BY 4.0 (publisher claims images and annotations)",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://docs.ultralytics.com/datasets/detect/construction-ppe",
            "rights_status": "publisher_dataset_license_stated; original photographer/generator rights unverified",
            "rights_blocker": False,
            "rights_note": "Publisher README and archive license assert CC BY 4.0; do not claim individual creator rights independently verified.",
            "attribution_status": "citation prepared; include source/license link and derived-label change notice",
            "attribution_blocker": False,
            "original_id": r["source_image_id"], "original_url": "",
            "original_split": r["source_split"],
            "group_proposal": "construction_ppe_all_unresolved",
            "group_evidence": "No original video/camera/scene ID. One conservative cohort for all 150 until provenance or visual grouping resolves it.",
            "synthetic_status": "unknown_per_image",
        }
    r = oi_by_id[record["id"]]
    meta = r["metadata"]
    parsed = urlparse(meta["OriginalLandingURL"])
    parts = [p for p in parsed.path.split("/") if p]
    photographer = parts[1] if len(parts) > 1 and parts[0] == "photos" else "unknown"
    return {
        "publisher": "Google Open Images / original Flickr photographer",
        "source_version": "Open Images V7 validation",
        "license": meta["License"], "license_url": meta["License"],
        "source_url": "https://storage.googleapis.com/openimages/web/download_v7.html",
        "rights_status": "Open Images CC listing and live Flickr page URI matched; exact-photo rights still need human confirmation",
        "rights_blocker": True,
        "rights_note": "Stage 2B checked that the landing page contained the license URI, but did not preserve a page snapshot or prove that URI applied to this exact photo. Google disclaims image-level license warranty.",
        "attribution_status": "image author/title/landing URL/license retained; final notice pending",
        "attribution_blocker": False,
        "original_id": meta["ImageID"], "original_url": meta["OriginalURL"],
        "original_split": meta["Subset"],
        "group_proposal": f"flickr_photographer:{photographer}",
        "group_evidence": f"Original landing photo ID {parts[-1] if parts else 'unknown'}; group all selected photos from photographer account {photographer}. Same scene across accounts is not ruled out.",
        "synthetic_status": "real_photo_claim; visually verify depictions/composites",
        "author": meta["Author"], "title": meta["Title"],
        "landing_url": meta["OriginalLandingURL"],
        "landing_http_status_stage2b": r.get("landing_http_status"),
    }


def main():
    oi_by_id = {r["image_id"]: r for r in OI["records"] if r.get("sha256")}
    const_by_id = {r["source_image_id"]: r for r in CONST["records"]}
    ledger, corrections, attribution = [], [], []
    for r in AUDIT["records"]:
        facts = source_facts(r, oi_by_id, const_by_id)
        source_counts = r["source_box_counts"]
        issue, proposal = TRIAGE.get(r["id"], (None, None))
        vest = source_counts.get("safety_vest", 0) > 0 and r["source"] == "construction_ppe"
        broad_helmet = source_counts.get("helmet", 0) > 0 and r["source"] == "open_images_v7"
        broad_handgun = source_counts.get("pistol", 0) > 0 and r["source"] == "open_images_v7"
        depiction = r.get("depiction_source_boxes", 0) > 0
        flags = [x for x, present in (
            ("vest_scope", vest), ("broad_helmet_scope", broad_helmet),
            ("handgun_scope", broad_handgun), ("depiction_source_flag", depiction),
            ("full_image_visual_triage", bool(issue))) if present]
        correction = bool(issue in {"illustrated_people", "cake_figures", "baseball_helmets",
                                    "football_helmet", "hockey_helmet", "roller_derby",
                                    "racing_helmet", "long_gun", "shotgun"})
        if correction:
            route = "requires_annotation_correction"
        elif facts["rights_blocker"]:
            route = "requires_rights_clarification"
        else:
            route = "requires_source_group_clarification"
        entry = {
            "review_id": f"S2C-{len(ledger)+1:03d}", "source": r["source"],
            "image_id": r["id"], "raw_image_path": r["path"].replace("\\", "/"),
            "original_id": facts["original_id"], "original_split": facts["original_split"],
            "source_version": facts["source_version"],
            "source_url": facts["source_url"], "original_url": facts["original_url"],
            "license": facts["license"], "license_url": facts["license_url"],
            "image_rights_status": facts["rights_status"], "rights_clarification_required": facts["rights_blocker"],
            "rights_evidence_note": facts["rights_note"],
            "attribution_status": facts["attribution_status"], "attribution_blocker": facts["attribution_blocker"],
            "annotation_geometry_status": "passed_stage2b_mechanical_checks",
            "class_definition_status": "correction_proposed" if correction else ("targeted_review_required" if flags else "unreviewed"),
            "class_review_flags": flags,
            "missing_target_annotations_status": "not_verified_all_six_classes",
            "source_group_proposal": facts["group_proposal"],
            "source_group_evidence": facts["group_evidence"],
            "source_group_verified": False,
            "exact_duplicate_status": "no_sha256_match_within_187",
            "near_duplicate_status": "no_phash_match_at_distance_10_or_less_within_187; scene_overlap_unresolved",
            "synthetic_status": facts["synthetic_status"],
            "provisional_source_box_counts": source_counts,
            "source_boxes_xywh": r["boxes"],
            "sha256": r["sha256"], "phash": r["phash"],
            "visual_triage_issue": issue,
            "proposed_annotation_correction": proposal or ("Review all vest boxes against the fixed safety_vest definition; drop plain high-visibility shirts/jackets and check all six target classes." if vest else ""),
            "route": route,
            "human_decision": "pending", "human_reviewer": "", "human_notes": "",
            "accepted": False, "split_eligible": False,
        }
        for key in ("author", "title", "landing_url", "landing_http_status_stage2b"):
            if key in facts: entry[key] = facts[key]
        ledger.append(entry)
        if entry["proposed_annotation_correction"]:
            corrections.append({"review_id": entry["review_id"], "image_id": r["id"],
                                "source": r["source"], "issue": issue or "vest_scope",
                                "proposal": entry["proposed_annotation_correction"],
                                "applied": False, "approved": False})
        attribution.append({"review_id": entry["review_id"], "image_id": r["id"],
                            "source": r["source"], "source_url": facts["source_url"],
                            "landing_url": facts.get("landing_url", ""),
                            "author": facts.get("author", "Ultralytics team"),
                            "title": facts.get("title", "Ultralytics Construction-PPE Dataset"),
                            "license": facts["license"], "license_url": facts["license_url"],
                            "changes": "None to raw image; proposed labels not applied"})
    payload = {"schema_version": "stage2c-ledger-v1", "fixed_classes": ["person", "helmet", "safety_vest", "backpack", "knife", "pistol"],
               "accepted_count": 0, "records": ledger}
    (OUT / "review_ledger.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "review_ledger.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [k for k in ledger[0] if k not in ("source_boxes_xywh", "provisional_source_box_counts")]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ledger:
            flat = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            writer.writerow(flat)
    (OUT / "proposed_corrections.json").write_text(json.dumps(corrections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "attribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=attribution[0].keys())
        writer.writeheader(); writer.writerows(attribution)
    write_html(ledger)
    counts = {
        "images": len(ledger), "routes": dict(Counter(x["route"] for x in ledger)),
        "rights_clarification": sum(x["rights_clarification_required"] for x in ledger),
        "attribution_blocker": sum(x["attribution_blocker"] for x in ledger),
        "geometry_failure": 0,
        "class_correction_proposed": sum(x["class_definition_status"] == "correction_proposed" for x in ledger),
        "class_targeted_review": sum(x["class_definition_status"] == "targeted_review_required" for x in ledger),
        "missing_six_class_review": sum(x["missing_target_annotations_status"] != "verified_complete" for x in ledger),
        "source_group_unverified": sum(not x["source_group_verified"] for x in ledger),
        "synthetic_status_unknown": sum(x["synthetic_status"] == "unknown_per_image" for x in ledger),
        "exact_duplicate_matches": 0, "phash_near_matches_at_10": 0,
        "provisional_group_buckets": len(set(x["source_group_proposal"] for x in ledger)),
        "proposed_correction_records": len(corrections),
    }
    (OUT / "blocker_counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    with (OUT / "checksums.sha256").open("w", encoding="utf-8") as f:
        for name in ("review_ledger.json", "review_ledger.csv", "proposed_corrections.json", "attribution.csv", "blocker_counts.json", "review.html"):
            path = OUT / name
            f.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")
    print(json.dumps(counts, indent=2))


def write_html(ledger):
    ui_data = [{k: row[k] for k in ("review_id", "source", "image_id", "raw_image_path", "license",
                                   "image_rights_status", "attribution_status", "annotation_geometry_status",
                                   "class_definition_status", "class_review_flags", "missing_target_annotations_status",
                                   "source_group_proposal", "source_group_evidence", "exact_duplicate_status",
                                   "near_duplicate_status", "synthetic_status", "source_boxes_xywh",
                                   "proposed_annotation_correction", "route")}
               for row in ledger]
    data = json.dumps(ui_data, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2C — Quarantine review</title><style>
body{margin:0;font:15px system-ui,sans-serif;color:#17252c;background:#f3f6f5}header{padding:14px 22px;background:#143b4c;color:white}header h1{font-size:22px;margin:0}header p{margin:6px 0 0}main{display:grid;grid-template-columns:270px minmax(430px,1fr) 390px;gap:12px;padding:12px;height:calc(100vh - 104px)}section,aside{background:white;border:1px solid #cbd7d9;border-radius:8px;overflow:auto;padding:12px}.list button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #e4e9e9;background:white;padding:8px;cursor:pointer}.list button.active{background:#cce8e8}.list button:hover{background:#eaf3f3}.small{font-size:12px;color:#53666c}.tag{display:inline-block;background:#f7e7bb;border-radius:4px;padding:2px 5px;margin:2px}.viewer{display:flex;flex-direction:column;gap:8px}canvas{max-width:100%;max-height:75vh;background:#222;object-fit:contain;border-radius:5px}label{display:block;margin:8px 0 3px;font-weight:600}input[type=text],textarea,select{box-sizing:border-box;width:100%;padding:7px;border:1px solid #afbdc0;border-radius:4px;font:inherit}textarea{min-height:70px}button.action{margin:3px;padding:7px 11px;background:#155b6a;color:white;border:0;border-radius:4px;cursor:pointer}.checks label{font-weight:400}.checks input{margin-right:6px}.note{border-left:4px solid #d78b19;padding:7px;background:#fff7e6}.meta{white-space:pre-wrap;word-break:break-word}.toolbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.toolbar input{max-width:170px}.muted{color:#667}.boxlist{max-height:230px;overflow:auto;border:1px solid #ddd}.boxrow{display:grid;grid-template-columns:1fr 120px 190px;gap:5px;padding:4px;border-bottom:1px solid #ddd;font-size:12px}.boxrow input,.boxrow select{font-size:12px;padding:3px}
@media(max-width:1050px){main{grid-template-columns:1fr;height:auto}section,aside{max-height:none}.list{max-height:200px}}
</style></head><body><header><h1>Stage 2C — Quarantine review</h1><p>187 candidates. All decisions start pending. Source boxes are proposals. Export a decision file; raw images and labels stay unchanged.</p></header>
<main><section><div class="toolbar"><input id="search" type="text" placeholder="Filter ID or source"><select id="routeFilter"><option value="">All routes</option><option>requires_annotation_correction</option><option>requires_rights_clarification</option><option>requires_source_group_clarification</option></select></div><div id="list" class="list"></div></section>
<section class="viewer"><div><strong id="title"></strong> <span id="source" class="small"></span></div><canvas id="canvas"></canvas><div id="imageLink"></div><div class="small">Box colors: person green · helmet yellow · safety vest orange · backpack blue · knife pink · pistol purple. Drawn from unapproved source labels.</div><div id="tags"></div><div id="evidence" class="meta"></div><strong>Source boxes: decide each one</strong><div class="small">For resize, enter normalized x-center,y-center,width,height. Unreviewed is the default.</div><div id="boxEditor" class="boxlist"></div></section>
<aside><div class="note"><strong>Human decision only.</strong> Selecting “approve” here records your review decision in an exported draft; it does not release an image into the dataset. Complete the independent gates and export the result.</div><label>Decision</label><select id="decision"><option value="pending">Pending</option><option value="approve">Approve after corrections and gates</option><option value="correct">Needs annotation correction</option><option value="reject">Reject</option></select><label>Reviewer name</label><input id="reviewer" type="text" placeholder="Your name"><label>Rights evidence / page note</label><textarea id="rightsNote"></textarea><label>Confirmed source group ID</label><input id="group" type="text" placeholder="Leave blank until evidence exists"><label>Synthetic status</label><select id="synthetic"><option value="unknown">Unknown</option><option value="real">Real photo</option><option value="generated">Generated</option><option value="mixed">Composite / mixed</option></select><div class="checks"><label><input id="rights" type="checkbox">Exact-image rights checked</label><label><input id="attribution" type="checkbox">Attribution captured</label><label><input id="geometry" type="checkbox">Boxes and geometry reviewed</label><label><input id="classes" type="checkbox">Class definitions reviewed</label><label><input id="complete" type="checkbox">All six target classes checked</label><label><input id="groupCheck" type="checkbox">Source group verified</label><label><input id="duplicates" type="checkbox">Related-image duplicates reviewed</label></div><label>New boxes, one per line: class_id,cx,cy,w,h</label><textarea id="addedBoxes"></textarea><label>Annotation corrections / rejection reason</label><textarea id="corrections"></textarea><label>Other notes</label><textarea id="notes"></textarea><div class="toolbar"><button class="action" id="save">Save draft decision</button><button class="action" id="exportJson">Export JSON</button><button class="action" id="exportCsv">Export CSV</button></div><label>Import saved JSON decisions</label><input type="file" id="importJson" accept="application/json,.json"><p class="small">A browser draft is also stored locally. Exports contain reviewer decisions only; combine them with the immutable ledger by review_id after adjudication.</p></aside></main>
<script id="dataset" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('dataset').textContent), names=['person','helmet','safety_vest','backpack','knife','pistol'], colors=['#20da5d','#edc846','#ff8a26','#54c5ff','#f04c8b','#af63ff'];
let decisions={};try{decisions=JSON.parse(localStorage.getItem('stage2c_draft_v1')||'{}')}catch(e){}let current=0;
const $=id=>document.getElementById(id);const fields=['decision','reviewer','rightsNote','group','synthetic','rights','attribution','geometry','classes','complete','groupCheck','duplicates','addedBoxes','corrections','notes'];
function fileUrl(p){return '../../'+p.replaceAll('\\','/').split('/').map(encodeURIComponent).join('/')}
function draw(row){const image=new Image();image.onload=()=>{const c=$('canvas'),ctx=c.getContext('2d');const scale=Math.min(1000/image.width,720/image.height,1);c.width=Math.round(image.width*scale);c.height=Math.round(image.height*scale);ctx.drawImage(image,0,0,c.width,c.height);ctx.lineWidth=Math.max(2,Math.round(c.width/350));ctx.font='bold 14px sans-serif';row.source_boxes_xywh.forEach(b=>{const [cls,x,y,w,h]=b;const x0=(x-w/2)*c.width,y0=(y-h/2)*c.height,bw=w*c.width,bh=h*c.height;ctx.strokeStyle=colors[cls];ctx.fillStyle=colors[cls];ctx.strokeRect(x0,y0,bw,bh);ctx.fillText(names[cls],x0+2,Math.max(13,y0+13))})};image.onerror=()=>{$('canvas').width=1;$('canvas').height=1};image.src=fileUrl(row.raw_image_path)}
function renderList(){const q=$('search').value.toLowerCase(),route=$('routeFilter').value;const wrap=$('list');wrap.innerHTML='';rows.forEach((r,i)=>{if(q&&!(`${r.image_id} ${r.source} ${r.review_id}`.toLowerCase().includes(q)))return;if(route&&r.route!==route)return;const b=document.createElement('button');b.className=i===current?'active':'';b.textContent=`${r.review_id} ${r.image_id} · ${r.source} · ${decisions[r.review_id]?.decision||'pending'}`;b.onclick=()=>show(i);wrap.appendChild(b)})}
function show(i){current=i;const r=rows[i],d=decisions[r.review_id]||{};$('title').textContent=`${r.review_id} · ${r.image_id}`;$('source').textContent=r.source;$('imageLink').innerHTML=`<a href="${fileUrl(r.raw_image_path)}" target="_blank">Open original at full size</a>`;$('tags').innerHTML=r.class_review_flags.map(x=>`<span class="tag">${x}</span>`).join('');$('evidence').textContent=`Route: ${r.route}\nLicense: ${r.license}\nRights: ${r.image_rights_status}\nAttribution: ${r.attribution_status}\nGeometry: ${r.annotation_geometry_status}\nClass status: ${r.class_definition_status}\nSix-class completeness: ${r.missing_target_annotations_status}\nGroup proposal: ${r.source_group_proposal}\nGroup evidence: ${r.source_group_evidence}\nDuplicates: ${r.exact_duplicate_status}; ${r.near_duplicate_status}\nSynthetic: ${r.synthetic_status}\nProposed correction: ${r.proposed_annotation_correction||'None'}`;fields.forEach(f=>{$(f).value=d[f]||((f==='decision')?'pending':(f==='synthetic'?'unknown':''));if($(f).type==='checkbox')$(f).checked=Boolean(d[f])});const boxRoot=$('boxEditor');boxRoot.innerHTML='';r.source_boxes_xywh.forEach((b,idx)=>{const row=document.createElement('div');row.className='boxrow';const label=document.createElement('span');label.textContent=`#${idx+1} ${names[b[0]]} [${b.slice(1).map(x=>x.toFixed(3)).join(', ')}]`;const action=document.createElement('select');action.innerHTML='<option value="unreviewed">Unreviewed</option><option value="keep">Keep</option><option value="drop">Drop</option><option value="resize">Resize</option>';action.value=d.boxActions?.[idx]?.action||'unreviewed';const coords=document.createElement('input');coords.type='text';coords.placeholder='cx,cy,w,h if resize';coords.value=d.boxActions?.[idx]?.coords||'';row.append(label,action,coords);boxRoot.appendChild(row)});draw(r);renderList()}
function save(){const r=rows[current],d={review_id:r.review_id,image_id:r.image_id};fields.forEach(f=>d[f]=$(f).type==='checkbox'?$(f).checked:$(f).value);d.boxActions=[...$('boxEditor').children].map((row,idx)=>({index:idx,class_id:r.source_boxes_xywh[idx][0],action:row.querySelector('select').value,coords:row.querySelector('input').value}));if(d.decision==='approve'&&(!d.reviewer||!d.group||!['rights','attribution','geometry','classes','complete','groupCheck','duplicates'].every(k=>d[k])||d.boxActions.some(b=>b.action==='unreviewed'||(b.action==='resize'&&!b.coords)))){alert('Complete reviewer, group, every gate and every source-box decision before approving.');return false}decisions[r.review_id]=d;localStorage.setItem('stage2c_draft_v1',JSON.stringify(decisions));renderList();return true}
function exportFile(type){if(!save())return;const values=Object.values(decisions);let body,mime,name;if(type==='json'){body=JSON.stringify({schema:'stage2c-human-decisions-v1',exported_at:new Date().toISOString(),decisions:values},null,2);mime='application/json';name='stage2c_human_decisions.json'}else{const keys=['review_id','image_id',...fields,'boxActions'];const esc=x=>'"'+String(typeof x==='object'?JSON.stringify(x):x??'').replaceAll('"','""')+'"';body=[keys.join(','),...values.map(v=>keys.map(k=>esc(v[k])).join(','))].join('\r\n');mime='text/csv';name='stage2c_human_decisions.csv'}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:mime}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('save').onclick=save;$('exportJson').onclick=()=>exportFile('json');$('exportCsv').onclick=()=>exportFile('csv');$('search').oninput=renderList;$('routeFilter').onchange=renderList;$('importJson').onchange=async e=>{const file=e.target.files[0];if(!file)return;const obj=JSON.parse(await file.text());if(obj.schema!=='stage2c-human-decisions-v1')throw Error('Wrong decision schema');const ids=new Set(rows.map(x=>x.review_id));obj.decisions.forEach(d=>{if(ids.has(d.review_id))decisions[d.review_id]=d});localStorage.setItem('stage2c_draft_v1',JSON.stringify(decisions));show(current)};show(0);
</script></body></html>'''
    (OUT / "review.html").write_text(template.replace("__DATA__", data), encoding="utf-8")


if __name__ == "__main__":
    main()
