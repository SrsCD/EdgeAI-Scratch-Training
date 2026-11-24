# EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED; provisional annotations: test evaluation

Split ledger SHA-256: `aa8ea6a9888101587554a96ead9270dcdfa160d27e074c55a6076cfdd369d0bb`. Scratch checkpoint SHA-256: `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`. Evaluated 1473 images with confidence ≥0.25 and IoU matching ≥0.5.

TP=1019; FP=700; FN=1055; precision=0.5927865037812682; recall=0.491321118611379; F1=0.5373055628789876; mAP@50=0.44271381263474135; mAP@50–95=0.2902925145883096; mean matched-box IoU=0.8274582892246862. Per-class AP@50/AP@75 and support are in `per_class_metrics.csv`. Mean latency/FPS: {'mean_latency_ms': 9.156313713509844, 'median_latency_ms': 7.9967, 'fps_model_only': 109.2142571004893, 'image_count': 1473}.

Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). AP uses 101-point interpolated precision at each recall level; mAP averages supported classes and IoU thresholds 0.50:0.05:0.95. Mean matched IoU averages one-to-one same-class matched boxes at the stated operating threshold. Timing CSV must be synchronized batch-one model-only latency; FPS=1000×images/sum(latency_ms). Confusion matrix rows are ground truth and columns are predictions, including background. Unsupported denominators are unavailable, not zero.

Training/validation loss plot generated: True. Accuracy/specificity are omitted because detection true negatives are undefined; a separate verified image-level presence/absence task would be required. Test outputs must not be used to tune thresholds or checkpoints.
