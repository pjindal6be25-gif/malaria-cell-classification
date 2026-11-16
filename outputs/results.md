Test set: 4,134 images, Parasitized = positive class. Run of 2026-10-18 16:25 on NVIDIA GeForce RTX 5050 Laptop GPU. Tuned thresholds were chosen on the validation set only.

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | 0.50 (default) | 96.57 % | 97.34 % | 95.74 % | 97.39 % | 0.9654 | 0.9927 | 88 | 54 |
| Custom CNN | 0.176 (tuned) | 95.55 % | 93.89 % | 97.44 % | 93.66 % | 0.9563 | 0.9927 | 53 | 131 |
| MobileNetV2 (transfer) | 0.50 (default) | 96.15 % | 97.70 % | 94.53 % | 97.77 % | 0.9609 | 0.9937 | 113 | 46 |
| MobileNetV2 (transfer) | 0.281 (tuned) | 96.52 % | 95.89 % | 97.19 % | 95.84 % | 0.9654 | 0.9937 | 58 | 86 |

| Model | Parameters | Training time | Epochs | ms/image, batch 1 (model only) | ms/image, batch 64 (model only) | ms/image end to end* |
|---|---|---|---|---|---|---|
| Custom CNN | 0.25 M | 7.1 min | stopped early at epoch 46 of 50 (best epoch 44) | 0.74 | 0.072 | 1.44 |
| MobileNetV2 (transfer) | 2.23 M | 3.8 min | frozen: ran all 5 epochs (best epoch 4); fine-tune: stopped early at epoch 18 of 20 (best epoch 13) | 5.00 | 0.111 | 4.72 |

\* one image at a time, including reading and resizing the PNG (what predict.py does).

Total run time (training + evaluation): 12.0 min. Headline model and `predict.py` default: **Custom CNN**. Highest validation AUC: MobileNetV2 (transfer).

Custom CNN vs MobileNetV2: 8.9× fewer parameters, 3.3× faster end to end, 6.8× faster at batch 1 and 1.5× faster at batch 64 (model only).

Is the gap bigger than noise? Paired bootstrap over the test images (1,000 resamples), MobileNetV2 minus Custom CNN:

| Metric | Threshold | Difference | 95 % interval | Verdict |
|---|---|---|---|---|
| ROC-AUC | - | +0.0010 | -0.0003 to +0.0025 | within noise |
| Accuracy | 0.50 | -0.4112 pts | -0.8466 to +0.0490 pts | within noise |
| F1 | 0.50 | -0.0045 | -0.0089 to +0.0003 | within noise |
| Accuracy | tuned | +0.9676 pts | +0.4112 to +1.5481 pts | MobileNetV2 better |
| F1 | tuned | +0.0091 | +0.0037 to +0.0148 | MobileNetV2 better |