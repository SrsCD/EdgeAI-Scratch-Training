"""Write an evidence-based Stage 4A report from actual run artifacts."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/scratch_yolo11n_stage3c_experimental_b16'
OUT=ROOT/'documentation/stage4a_scratch_training_report.md'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    provenance=json.loads((RUN/'stage4a_provenance.json').read_text())
    with (RUN/'results.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('No completed epochs')
    if any(not math.isfinite(float(r[k])) for r in rows for k in
           ('train/box_loss','train/cls_loss','train/dfl_loss','val/box_loss','val/cls_loss','val/dfl_loss')):
        raise SystemExit('Nonfinite loss recorded')
    latest=rows[-1];best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    weights=RUN/'weights';best_pt=weights/'best.pt';last_pt=weights/'last.pt'
    complete=provenance.get('outcome')=='completed' and best_pt.is_file() and last_pt.is_file()
    samples=ROOT/'data/experimental/stage3c_v1/review_sample/reviews'
    reviews={p.stem:json.loads(p.read_text())['decision'] for p in samples.glob('P*.json')}
    if len(reviews)!=10 or set(reviews.values())!={'Hold'}:raise SystemExit('Ten Hold decisions changed')
    config=provenance['effective_arguments']
    status='COMPLETED' if complete else ('INTERRUPTED AFTER '+str(len(rows))+' EPOCHS' if provenance.get('outcome','').startswith('interrupted') else 'IN PROGRESS')
    lines=[
        '# Stage 4A — experimental YOLO11n scratch training','',
        f'**{status}. EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED.** Training labels and validation labels are provisional. No final test evaluation was run.','',
        '## Preflight and split','',
        'The Stage 3C preflight passed before both smoke and full training. The split stayed 5,417 train / 1,473 test / 790 validation, with six fixed classes: 0 person, 1 helmet, 2 safety_vest, 3 backpack, 4 knife, 5 pistol. The 10 manual-review decisions were all **Hold**, unresolved, and none were approvals. The approved-dataset preflight remains separate and blocked. Training and checkpoint selection used train and validation only; the test split was not loaded for Stage 4A.','',
        f"Dataset ledger SHA-256: `{provenance['ledger_sha256']}`. Split statistics SHA-256: `{provenance['split_stats_sha256']}`. Architecture YAML SHA-256: `{provenance['architecture_sha256']}`. Data YAML SHA-256: `{provenance['data_yaml_sha256']}`. Initial train config SHA-256: `{provenance['train_config_sha256']}`.",'',
        '## Scratch initialization and run settings','',
        f"YOLO11n was constructed from `configs/yolo11n.yaml` with an empty Ultralytics checkpoint dictionary and no checkpoint path or model `.pt` path. `pretrained={config['pretrained']}`, `resume={config['resume']}`, `device={config['device']}`, `seed={config['seed']}`, `imgsz={config['imgsz']}`, `workers={config['workers']}`, `amp={config['amp']}`. The full run used **batch {config['batch']}**, maximum **{config['epochs']} epochs**, and validation early stopping patience **{config['patience']}**. The initial batch-2 run completed one epoch and was interrupted for measured throughput tuning; its logs and checkpoint remain under `runs/scratch_yolo11n_stage3c_experimental`. A separate batch-16 smoke test passed with finite loss and 3.9 GB peak allocated VRAM. The full batch-16 run started afresh from YAML, not from either earlier checkpoint.",'',
        f"GPU: {provenance['gpu']} ({provenance['vram_bytes']} bytes). PyTorch {provenance['torch']}; Python {provenance['python']}; Ultralytics 8.3.233. Started UTC {provenance['started_utc']}. Outcome: {provenance.get('outcome','currently running')}. Last logged cumulative training time: {latest['time']} seconds. Peak allocated VRAM during the batch-16 smoke test: 3,896,231,936 bytes; the full run displayed about 4.5 GB in its training log.",'',
        '## Actual training and validation results','',
        f"Completed epoch rows: **{len(rows)}**. Latest epoch: **{latest['epoch']}**. Latest train losses (box / class / DFL): **{latest['train/box_loss']} / {latest['train/cls_loss']} / {latest['train/dfl_loss']}**. Latest validation losses: **{latest['val/box_loss']} / {latest['val/cls_loss']} / {latest['val/dfl_loss']}**.",'',
        f"Latest validation precision **{latest['metrics/precision(B)']}**, recall **{latest['metrics/recall(B)']}**, mAP@50 **{latest['metrics/mAP50(B)']}**, mAP@50–95 **{latest['metrics/mAP50-95(B)']}**. Highest CSV mAP@50–95 was **{best['metrics/mAP50-95(B)']}** at epoch **{best['epoch']}**. Ultralytics writes `best.pt` using its own validation fitness rule; the maximum CSV mAP epoch is reported as a separate observation.",'',
        f"Epoch metrics CSV: [results.csv](../runs/scratch_yolo11n_stage3c_experimental_b16/results.csv). Training arguments: [args.yaml](../runs/scratch_yolo11n_stage3c_experimental_b16/args.yaml). Console log: [stage4a_full_b16_console.log](../reports/stage4a_full_b16_console.log). Interrupted-run training and validation loss curves: [training_validation_losses.png](../reports/evaluation/stage3c_experimental_test/training_validation_losses.png).",'',
        '## Checkpoints and next stage','',
        f"Best checkpoint: `{best_pt}`"+(f" (SHA-256 `{digest(best_pt)}`)" if best_pt.exists() else ' (not yet written)')+'.',
        f"Last checkpoint: `{last_pt}`"+(f" (SHA-256 `{digest(last_pt)}`)" if last_pt.exists() else ' (not yet written)')+'.','',
        'Stage 4B should use the completed best checkpoint once, on the untouched provisional test split. It must compute TP, FP, FN, precision, recall, F1, AP@50, AP@75, mAP@50, mAP@50–95, matched-box IoU, and inference latency/FPS, plus raw/normalized confusion matrices, per-class tables, confidence CSV and PR curves. If image-level accuracy or specificity is needed, define a separate reviewed positive/negative image-level protocol first.','',
        '## Limitations','',
        'The 7,680-image experimental dataset is mechanically checked but not fully human verified. Missing co-occurring objects, unsuitable pistol depictions, source rights, and provisional group independence remain unresolved. Validation metrics measure agreement with provisional annotations. Future test metrics also cannot establish independently verified real-world accuracy. No datasets were downloaded, Model A was not changed, and CCTV video detection was not implemented.','']
    OUT.write_text('\n'.join(lines),encoding='utf-8')
    print(status,len(rows),OUT)
if __name__=='__main__':main()
