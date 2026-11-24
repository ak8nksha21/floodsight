# FloodSight

Uncertainty-aware flood impact assessment on SpaceNet 8 (Germany + Louisiana-East) with a Siamese multi-task U-Net.

Pre- and post-flood images go through a shared ResNet-34 encoder with residual change fusion. The model outputs
building, road, flooded-building and flooded-road masks. MC Dropout (20 passes) gives uncertainty maps and
flooded-building count, flooded-road length and flooded-area ranges. A validation-calibrated 90% range and a
grid priority score are added on top.

## Results (whole-area split: train ger_0 + lou_1, val lou_2, test lou_0)
| Model | Val mIoU | Test mIoU |
|---|---|---|
| U-Net / U-Net++ / DeepLabV3+ / single-image multi-task (post only) | 0.13-0.15 | 0.10-0.11 |
| 3D ResNet (pre+post) | 0.364 | 0.249 |
| U-Net (pre only) | 0.375 | 0.271 |
| Siamese v2 (35 ep, flood loss x2, TTA) | 0.376 | 0.231 |
| **Siamese multi-task (final)** | **0.398** | 0.260 |
| Siamese + val regions in training | n/a | 0.268 |

Calibrated 90% ranges on the unseen test area cover the true flooded-building count in 86% of scenes
(raw MC Dropout ranges: 36%); road length and area: 63% and 67%.

## Limitations
Only 2 regions for training; flooded detection drops on the unseen test area. A pre-image-only model scores
similarly on test, so part of the flooded signal is location context. No flood-water mask exists in SpaceNet 8.
Road length has almost no per-image skill. Metres assume about 0.4 m per pixel.

## Files
data.py, models.py, train.py (train + evaluate), infer.py (MC Dropout, impact, grid), calibrate.py,
save_results.py, app.py (Streamlit dashboard: `streamlit run app.py`), results/ (weights in fp16, metrics, outputs).# FloodSight
