# Real-time Explainable Automatic Target Recognition System (REATS)

Deep-learning ATR pipeline for IR imagery — detection, classification, explainability,
battlefield threat analysis, and an operator dashboard — built on top of Do et al.
(2025) / JKSCI Vol. 30 No. 1, and extended with a heterogeneous model ensemble,
false-alarm/miss-rate reporting, hard-negative mining, and a confidence-driven CMS
policy tier (Warning / Track / Engagement).

---

## Source paper

> **A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System**
> Gyu-Seok Do, Ju-Mi Park, Won-Seok Jang, Young-Sub Yang, Ji-Seok Yoon
> Naval R&D Center, Hanwha Systems, Pangyo, Korea
> *Journal of The Korea Society of Computer and Information (JKSCI)*, Vol. 30 No. 1, pp. 33–40, January 2025
> DOI: `10.9708/jksci.2025.30.01.033`

The paper trains 6 distinct architectures — ResNet18, VGG16, ResNeXt50_32x4d,
ConvNeXt_tiny, ViT_b_16, Swin_T — and softmax-averages their predictions. A single
ConvNeXt-tiny reaches ~90.25% accuracy on the paper's 6-class subset; the 6-architecture
ensemble reaches **92%**. REATS's Module B implements that same heterogeneous ensemble
over its own, much larger, 43-class taxonomy — see [`docs/gap_analysis_report.md`](docs/gap_analysis_report.md)
for the full paper-vs-implementation comparison.

---

## Two codebases in this repo

| Directory | Purpose | Entry point |
|---|---|---|
| `REATS/` | Primary system — full ATR pipeline (Modules A–E) | `modules/module_*.py` |
| `cadet_atr_project/cadet_atr/` | Research scaffold — domain adaptation experiments | `run_experiment.py` |

They use **different class taxonomies**: REATS's 43-class AIR/GROUND/NAVAL taxonomy
lives in `REATS/config/targets.yaml` (single source of truth — every module loads
from it); cadet_atr uses a 6-class set (`fixed_wing, rotary_wing, uav, vessel,
vehicle_ground, vehicle_apc`) mapped onto REATS classes via `ingestion/label_maps.yaml`.

---

## REATS architecture

```
IR Frame
  └─► Module A  module_a_detector.py     YOLOv4 (CSPDarknet53 + SPP + PANet), pure PyTorch
                                          detect() → list of {bbox, conf, class_id}
                                          crop_roi() pads each detection

  └─► Module B  module_b_classifier.py   Heterogeneous 6-architecture softmax-averaging
                                          ensemble: ConvNeXt_tiny, ResNeXt50, ViT_b_16,
                                          Swin_T, VGG16, ResNet18 — one model per
                                          architecture, fusing local CNN features with
                                          global Transformer attention
                                          TemperatureScaler — post-hoc ECE calibration
                                          hard_negative_mining.py — extra fine-tune pass
                                          on the F16/MiG19/MiG21 confusion the paper's
                                          own confusion matrix identifies

  └─► Module C  module_c_xai.py          GradCAM / GradCAM++ / EigenCAM, SHAP, LIME
                                          MCDropoutWrapper — predictive entropy → OOD flag
                                          faithfulness deletion/insertion AUC

  └─► Module D  module_d_dashboard.py    Streamlit 5-tab UI: Live Analysis | Batch
                                          Processing | Calibration | About | iPhone Live
                                          threat_policy.py — confidence + threat_level →
                                          Warning/Track/Engagement CMS policy tier
                                          threat_metrics.py — per-class False Alarm Rate
                                          / Miss Rate battlefield threat analysis

  └─► Module E  module_e_streamer.py     FastAPI + WebSocket server — serves an HTML
                                          capture page to a phone, /frame + /status
                                          endpoints polled by Module D's iPhone Live tab
```

**Data flow**: Module A produces ROI crops → Module B classifies → Module C explains →
Module D displays in real time. Module E feeds live phone-camera frames into Module D.

---

## Target taxonomy

43 classes across 3 domains, defined in `REATS/config/targets.yaml` — the single
source of truth all modules load from automatically.

| Domain | Classes | Threat levels |
|---|---:|---|
| AIR | 24 | fighters, bombers, attack/utility helicopters, UAVs |
| GROUND | 13 | MBTs, IFVs/APCs, artillery, air defense |
| NAVAL | 6 | patrol/fast-attack craft, surface combatants |

Each class carries a `threat_level` (RED / ORANGE / YELLOW — 36 / 6 / 1 classes) used
for dashboard color-coding, the RED-threat-weighted FAR/MR aggregate, and the
Warning/Track/Engagement policy ceiling. Add new classes by editing `targets.yaml`
directly — no code changes needed elsewhere.

---

## Battlefield threat analysis

Beyond the paper's accuracy/ECE/mAP targets, REATS reports the two failure modes that
matter operationally on a naval CMS (`modules/threat_metrics.py`):

- **False Alarm Rate** `FAR = FP / (FP + TP) = 1 − Precision` — a civilian/friendly
  target misidentified as an enemy threat class.
- **Miss Rate** `MR = FN / (FN + TP) = 1 − Recall` — an actual threat that is never
  flagged. RED-threat-class MR is reported separately as the critical figure: a missed
  enemy fighter never raises an alarm.

Every detection also resolves to a CMS action tier (`modules/threat_policy.py`):

| Tier | Confidence | Ceiling | Meaning |
|---|---|---|---|
| NONE | < 0.50 | — | Below detection floor — no operator action |
| WARNING | ≥ 0.50 | YELLOW | Alert only — log + display, no tasking |
| TRACK | ≥ 0.75 | ORANGE | Cue sensors, maintain track — no engagement authority |
| ENGAGEMENT | ≥ 0.90 | RED only | Engagement-eligible — operator confirms per ROE |

ORANGE/YELLOW targets can never reach ENGAGEMENT regardless of confidence — a false
engagement call on a misclassified civilian/friendly target is the costliest failure
the system can produce. Thresholds live in `targets.yaml`'s `operational_policy`
section, not hardcoded in application code.

---

## Performance — targets vs. reported results

| Metric | Target | Reported result | Verdict |
|---|---|---|---|
| Classification accuracy | ≥ 92% | **93.12%** | PASS |
| ECE (calibration) | ≤ 0.05 | 0.0395 temperature-scaled (raw 0.1226) | PASS (scaled) |
| mAP@0.5 (detection) | ≥ 75% | not evaluated — bootstrapped detector untrained on IR by design | — |
| End-to-end latency | ≤ 40 ms/frame | 111.9 ms | FAIL (~2.8×) |
| FPS | ≥ 20 | ≈8.9 (derived from the latency figure, not independently benchmarked) | FAIL |
| Faithfulness AUC | ≥ 0.80 | 0.49 deletion / 0.75 insertion | FAIL |

Reported results are from the 2026-07-04 full 300-epoch training run
(`real-time-ex-03`, Tesla T4) — the training regime the paper itself specifies,
and the only run whose figures are quoted here. Only 8 of 43 classes currently
have real-image backing (the rest are synthetic-only), so treat classification
accuracy as validated architecture performance on the current data mix, not a
claim about real-world field accuracy. Detection has not produced a working
model in any run to date — a separate, faster detector-training attempt
reached mAP@0.5 of 0.0011. FAR/MR have no paper-given target; they're reported
alongside these, not pass/fail gated.

A separate fast-training-mode run measured a higher headline accuracy
(95.50%) — deliberately **not** used above. Fast mode trades training quality
for speed (an expected accuracy *drop*, not a gain), so its higher number
reflects data-composition inflation rather than a better result. Full
reasoning, per-domain breakdown, and visual evidence from both runs:
[`docs/kaggle_run_report.md`](docs/kaggle_run_report.md).

---

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended (CPU works for smoke testing and architecture validation)

### Installation

```bash
python -m venv /home/user/reats_env
source /home/user/reats_env/bin/activate
pip install -r REATS/requirements.txt
```

> **Note:** System Python on Debian ships a managed `blinker 1.7.0` that blocks
> `pip install`. Always activate the venv first.

Verify the environment, then run the full smoke test (no GPU or real data required):

```bash
python REATS/verify_env.py
python REATS/smoke_test.py
```

---

## REATS quickstart

### 1. Prepare data

Expected split: **170 train / 30 val / 200 test** per class.

```bash
# Ingest raw datasets into the standard split
cd REATS && python -m ingestion.pipeline \
    --datasets FLIR_Thermal:/path/to/flir HIT_UAV:/path/to/hit_uav \
    --out data/ --train 170 --val 30 --test 200

# Validate an existing split
python REATS/dataset_validator.py

# Auto-organise images from a raw folder
python REATS/dataset_validator.py --source raw/ --organize
```

If no real IR data is available yet:

```bash
python REATS/generate_flir_fallback.py --out REATS/data/                       # synthetic only
python REATS/generate_flir_fallback.py --flir /path/to/flir_adas/ --out REATS/data/  # FLIR-remapped
```

### 2. Bootstrap the detector

```bash
python REATS/bootstrap_detector_weights.py                              # downloads + converts COCO darknet weights
python REATS/bootstrap_detector_weights.py --weights checkpoints/my_yolov4.pt
python REATS/bootstrap_detector_weights.py --darknet ~/yolov4.weights
```

### 3. Train Module B

```bash
cd REATS

# Single model (fast iteration)
python modules/module_b_classifier.py

# Full 6-architecture heterogeneous ensemble
python -c "from modules.module_b_classifier import train_ensemble, CONFIG; train_ensemble(CONFIG)"

# Hard-negative fine-tune pass on the F16/MiG19/MiG21 confusion
python modules/hard_negative_mining.py --checkpoint checkpoints/convnext_tiny_0.pth --arch convnext_tiny
```

MLflow logs → `REATS/runs/` | Checkpoints → `REATS/checkpoints/`

### 4. Evaluate

```bash
python REATS/metrics_report.py --cls-weights checkpoints/convnext_tiny_0.pth   # accuracy, ECE, FAR/MR, mAP, latency
python REATS/metrics_report.py --quick                                        # skip slow faithfulness + mAP
```

### 5. Run the operator dashboard

```bash
cd REATS && streamlit run modules/module_d_dashboard.py
```

### 6. iPhone live feed (Module E)

```bash
cd REATS && python modules/module_e_streamer.py --ngrok   # HTTPS via ngrok
cd REATS && python modules/module_e_streamer.py           # HTTP on LAN only
```

---

## Docker deployment

```bash
docker compose up --build
```

- **Dashboard** → http://localhost:8501 (Modules A–D; reaches the streamer via `REATS_STREAMER_URL=http://streamer:7860`)
- **Streamer** (Module E) → http://localhost:7860

`docker-compose.yml` bind-mounts `REATS/checkpoints`, `REATS/data`, `REATS/runs` so
they persist across container restarts. GPU support requires uncommenting the
`deploy:` block (needs `nvidia-container-toolkit`) — otherwise both services run on CPU.

---

## Kaggle workflow

The primary GPU execution environment is `REATS/notebooks/01_kaggle_full_pipeline.ipynb`,
run natively as a Kaggle Notebook. Attach the datasets listed below via the right
panel's **+ Add Input** before running `c-config` — see [`CLAUDE.md`](CLAUDE.md)'s
"Kaggle notebook workflow" section for the full mount-path table. Run cells in order:

| Cell | Purpose |
|---|---|
| `c-gpu` | Detect GPU, set `device` |
| `c-install` | pip install all deps (streamlit, watchdog, pyngrok, …) |
| `c-clone` | git clone/pull + clear `__pycache__` + evict stale modules |
| `c-config` | Set dataset mount paths, warm-start source, and hyperparameters |
| `c-ingest` | Run ingestion pipeline across all attached datasets |
| `c-train-single` / `c-train-ensemble` | Train ConvNeXt_tiny alone, or the full 6-architecture ensemble (~18–24h on T4/P100) |
| `c-eval-metrics` | Test accuracy, ECE, per-domain breakdown |
| `c-faithfulness` | Run GradCAM + faithfulness AUC evaluation |
| `c-module-a` / `c-pipeline` | Module A detection demo + full A→B→C latency benchmark |
| `c-dashboard` | Launch Streamlit + ngrok tunnel |

### Ingestion pipeline datasets

`REATS/ingestion/pipeline.py` orchestrates format parsers, label resolution, patch
extraction, and a stratified train/val/test split. It supports COCO JSON, YOLO txt,
Pascal VOC XML, CSV, folder-per-class, and frame-sampled video sources:

| Dataset key | Format | Source |
|---|---|---|
| `FLIR_Thermal` | COCO JSON | Thermal street-level (civilian) |
| `FLIR_ADAS_v2` | COCO JSON | Teledyne FLIR road dataset |
| `HIT_UAV` | YOLO txt | HIT-UAV infrared, UAV top-down |
| `HIT_UAV_v2` | COCO JSON | HIT-UAV v1.2.1 |
| `Dataset2_Folders` | Video folder | Fixed/rotary wing mp4 clips |
| `HRSC2016` | Pascal VOC XML | High-resolution ship collection |
| `Ships_Aerial` | YOLO txt | Ship detection from aerial images |
| `Ships_Google_Earth` | Folder | Ships in Google Earth |
| `Ships_Vessels_Aerial` | CSV | Ships/vessels in aerial images |
| `SWIM` | Folder | Ship wake imagery |
| `CGI_Planes` | Folder | CGI planes in satellite imagery |
| `Airbus_Aircraft` | CSV | Airbus aircraft detection |
| `SwimmingPool_Car` | Folder | Swimming pool + car detection |
| `Vehicle_Dataset` | Folder | Vehicle dataset (car/truck/tank/APC) |
| `Aerial_Segmentation` | Folder | Semantic segmentation of aerial imagery |
| `Aerial_Roof_Seg` | Folder | Roof segmentation (contributes 0 annotations) |
| `Ships_Satellite` | *(TBD)* | Ships in satellite imagery |
| `SARScope_Maritime` | *(TBD)* | SAR maritime landscape |
| `Thermal_Ships` | *(TBD)* | Thermal (genuinely IR) ship imagery |
| `Aerial_Vehicle_Detection` | *(TBD)* | Aerial vehicle detection |
| `Battle_Tank_UAV` | *(TBD)* | UAV/aerial battle-tank detection — targets the T72/Abrams/Leopard2/BMP2/Bradley/K21 GROUND-domain confusion |

The 5 rows above (added 2026-07-05) have no confirmed format or `label_maps.yaml`
entry yet — run `c-ingest` and read its UNMAPPED report before assuming either.

Every generated image is tagged in `data/provenance.json` as `real`, `remapped`, or
`synthetic` — the Kaggle notebook's provenance cell splits test accuracy by bucket,
since most classes are currently synthetic-only and headline accuracy is an
architecture-validation number until label-map coverage grows.

---

## cadet_atr quickstart

Run all commands from `cadet_atr_project/cadet_atr/`.

```bash
python smoke_test.py                    # no GPU or real data required
python run_experiment.py --mode baseline_only
python run_experiment.py --mode adapt --strategy histogram    --checkpoint checkpoints/baseline_best.pt
python run_experiment.py --mode adapt --strategy domain_random
python run_experiment.py --mode adapt --strategy finetune     --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode adapt --strategy dann         --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode gap_only --checkpoint checkpoints/dann_best.pt
python run_experiment.py --mode full    # all 4 strategies sequentially
```

### Domain adaptation strategies

| # | Strategy | Key component |
|---|---|---|
| 1 | Histogram matching | `build_reference_histogram` + `apply_histogram_matching` |
| 2 | Domain randomisation | `BackgroundSwapDataset` (extended augmentation, no checkpoint needed) |
| 3 | Fine-tuning on real IR | `RealDataFinetuner.finetune(mode=head_only\|full\|layer_wise)` |
| 4 | DANN / GRL | `DANNModel` (ConvNeXt + Gradient Reversal Layer + domain classifier) |

`DANNModel.forward(x, return_domain=False)` returns class logits only for deployment;
pass `return_domain=True` during training to get both class and domain logits.

```python
from adaptation.strategies import DANNModel
import torch

model = DANNModel(num_classes=6)
dummy = torch.randn(4, 3, 224, 224)

cls_logits = model(dummy)                              # inference — domain head inactive
assert cls_logits.shape == (4, 6)

cls_logits, dom_logits = model(dummy, return_domain=True)  # training — both heads active
assert dom_logits.shape == (4, 2)
dom_logits.sum().backward()
```

---

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — developer/agent guide: environment setup, commands, module implementation details, known pitfalls
- [`docs/gap_analysis_report.md`](docs/gap_analysis_report.md) — paper-vs-implementation gap analysis (ensemble heterogeneity, FAR/MR, hard-negative mining, operational policy), each gap grounded in a paper citation
- [`docs/gap_analysis_slides.md`](docs/gap_analysis_slides.md) — the same analysis as a slide deck
- [`docs/kaggle_run_report.md`](docs/kaggle_run_report.md) — results from the real Kaggle GPU runs to date (accuracy/ECE/faithfulness/latency/mAP vs. paper targets, per-domain breakdown, real-vs-synthetic-data caveat)
- [`MEMORY.md`](MEMORY.md) — running log of architectural decisions and bug fixes across sessions

---

## References

- Do, G.-S., Park, J.-M., Jang, W.-S., Yang, Y.-S., & Yoon, J.-S. (2025). A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System. *JKSCI*, 30(1), 33–40. https://doi.org/10.9708/jksci.2025.30.01.033
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556.
- Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. *CVPR*.
- Liu, Z., Mao, H., Wu, C., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR*.
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929.
- Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
- Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. *JMLR*, 17(59), 1–35.
