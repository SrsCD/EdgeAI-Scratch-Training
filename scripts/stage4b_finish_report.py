"""Package measured Stage 4B provisional-test results without rerunning inference."""
import csv,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/evaluation/stage3c_experimental_test'
RUN=ROOT/'runs/scratch_yolo11n_stage3c_experimental_b16'
DOC=ROOT/'documentation/stage4b_experimental_evaluation_report.md'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def f(v):return f'{float(v):.4f}' if v is not None else 'N/A'
def main():
    metrics=json.loads((OUT/'metrics.json').read_text())
    stats=rows(ROOT/'data/experimental/stage3c_v1/split_stats.csv')
    epoch=rows(RUN/'results.csv')
    provenance=json.loads((RUN/'stage4a_provenance.json').read_text())
    inputs=json.loads((ROOT/'reports/evaluation_inputs/stage3c/stage3c_experimental_test_provenance.json').read_text())
    if len(epoch)!=12 or provenance['outcome']!='interrupted_by_user_after_12_completed_epochs':
        raise SystemExit('Training outcome differs from recorded interruption')
    fig,ax=plt.subplots(2,1,figsize=(9,8),sharex=True)
    x=[int(r['epoch']) for r in epoch]
    for key,label in [('metrics/precision(B)','Precision'),('metrics/recall(B)','Recall'),
                      ('metrics/mAP50(B)','mAP@50'),('metrics/mAP50-95(B)','mAP@50–95')]:
        ax[0].plot(x,[float(r[key]) for r in epoch],label=label)
    for key,label in [('val/box_loss','Box'),('val/cls_loss','Class'),('val/dfl_loss','DFL')]:
        ax[1].plot(x,[float(r[key]) for r in epoch],label=label)
    ax[0].set(ylabel='Validation metric',title='Experimental validation curves (provisional labels)')
    ax[1].set(xlabel='Completed epoch',ylabel='Validation loss')
    for a in ax:a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation_metric_curves.png',dpi=160);plt.close(fig)
    o=metrics['overall'];pc=metrics['per_class']
    lines=['# Stage 4B — experimental six-class test evaluation','',
           '**Measured against provisional annotations; not independently verified real-world accuracy.**'
           ' The 12-epoch scratch run was interrupted at the user’s four-hour limit. No training was resumed or restarted for this evaluation.','',
           '## Checkpoint and test protocol','',
           f"The locally saved `best.pt` and `last.pt` both loaded and have SHA-256 `{inputs['checkpoint_sha256']}`. Checkpoint metadata records zero-based epoch 11, corresponding to **epoch 12** in the preserved [training CSV](../runs/scratch_yolo11n_stage3c_experimental_b16/results.csv). That epoch recorded validation precision **0.54031**, recall **0.39859**, mAP@50 **0.45081**, and mAP@50–95 **0.29567**. The best checkpoint was selected with validation, not test, data. [Training report](stage4a_scratch_training_report.md) records the interruption, config, GPU and hashes.",'',
           f"The [Stage 3C ledger](../data/experimental/stage3c_v1/ledger.csv) SHA-256 is `{provenance['ledger_sha256']}`. The existing provisional test split contains **1,473 images**; its group assignment was unchanged. The 10 manual-review records are all **Hold**, never approvals. Inference used image size 640, batch one, a 0.001 detection export floor and NMS IoU 0.7; the operating point below is confidence **0.25** with classwise one-to-one matching IoU **0.50**. The exporter saved **198,805** post-NMS detections. The test was evaluated once from those saved predictions. Test scores were not used to train, tune thresholds or choose the checkpoint.",'',
           '## Actual class distribution','',
           '| Class | Train images | Train boxes | Test images | Test boxes | Validation images | Validation boxes |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        lines.append('| '+r['class_name']+' | '+' | '.join(r[k] for k in
            ('train_positive_images','train_boxes','test_positive_images','test_boxes','val_positive_images','val_boxes'))+' |')
    lines+=['','## Test metrics at the stated operating point','',
            '| Class | TP | FP | FN | Precision | Recall | F1 | AP@50 | AP@75 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in pc:
        lines.append(f"| {r['class_name']} | {r['tp']} | {r['fp']} | {r['fn']} | {f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} | {f(r['ap50'])} | {f(r['ap75'])} |")
    lines += ['',f"**Overall:** TP **{o['tp']}**, FP **{o['fp']}**, FN **{o['fn']}**; precision **{f(o['precision'])}**, recall **{f(o['recall'])}**, F1 **{f(o['f1'])}**. mAP@50 **{f(o['map50'])}**, mAP@50–95 **{f(o['map50_95'])}**; mean matched-box IoU **{f(o['mean_matched_iou'])}**. Mean model-only inference latency **{o['timing']['mean_latency_ms']:.3f} ms/image**, median **{o['timing']['median_latency_ms']:.3f} ms**, throughput **{o['timing']['fps_model_only']:.2f} FPS** over 1,473 images. Timing excludes image loading, preprocessing and NMS, so it is not end-to-end CCTV FPS.",'',
              'TP is a confidence-filtered prediction matched once to a same-class provisional box at IoU ≥0.50; unmatched predictions are FP and unmatched boxes are FN. Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is 2PR/(P+R). AP@50 and AP@75 are 101-point interpolated precision–recall areas at IoU 0.50 and 0.75 using the exported detections. mAP@50 averages six class AP@50 values; mAP@50–95 averages class AP over IoU 0.50:0.05:0.95. Mean matched IoU averages the one-to-one true-positive overlaps at confidence 0.25. FPS is 1,000 times image count divided by summed synchronized model-reported inference milliseconds.','',
              '## Artifacts','',
              '- [Raw confusion matrix](../reports/evaluation/stage3c_experimental_test/confusion_raw.csv) and [normalized confusion matrix](../reports/evaluation/stage3c_experimental_test/confusion_normalized.csv) (rows are provisional ground truth, columns predictions; background row/column represent FP/FN).',
              '- [Per-class metrics](../reports/evaluation/stage3c_experimental_test/per_class_metrics.csv), [per-detection confidence scores](../reports/evaluation/stage3c_experimental_test/per_prediction_confidence.csv), and [full metrics JSON](../reports/evaluation/stage3c_experimental_test/metrics.json).',
              '- [Precision–recall curve](../reports/evaluation/stage3c_experimental_test/precision_recall.png), [F1/confidence](../reports/evaluation/stage3c_experimental_test/f1_confidence.png), [precision/confidence](../reports/evaluation/stage3c_experimental_test/precision_confidence.png), and [recall/confidence](../reports/evaluation/stage3c_experimental_test/recall_confidence.png).',
              '- [Training and validation losses](../reports/evaluation/stage3c_experimental_test/training_validation_losses.png), [validation metric curves](../reports/evaluation/stage3c_experimental_test/validation_metric_curves.png), and [annotated predictions](../reports/evaluation/stage3c_experimental_test/prediction_examples).',
              '- [Inference inputs and checkpoint provenance](../reports/evaluation_inputs/stage3c) and [Stage 4A console log](../reports/stage4a_full_b16_console.log).','',
              '## Interpretation and limits','',
              'Person and knife recall are low at the fixed 0.25 confidence threshold; these values should be interpreted alongside their AP and PR curves. The image labels can omit co-occurring targets, and the source-group components are provisional, so some scored FP/FN may reflect annotation omissions or leakage. Pistol suitability and image rights are also unresolved. These test metrics measure agreement with provisional labels, not independently verified real-world accuracy. The 10 Hold samples do not validate the remaining images. Image-level accuracy and specificity are omitted because no separately reviewed positive/negative image-level protocol exists. No external dataset, pretrained checkpoint, new training run or CCTV video demo was used.','']
    DOC.write_text('\n'.join(lines),encoding='utf-8')
    print(DOC)
if __name__=='__main__':main()
