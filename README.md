# Real-time Explainable Automatic Target Recognition System (REATS)

Deep-learning ATR pipeline for IR imagery — detection, classification, explainability, and operator dashboard — based on Do et al. (2025) / JKSCI Vol. 30 No. 1.

---

## Source paper

> **A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System**
> Gyu-Seok Do, Ju-Mi Park, Won-Seok Jang, Young-Sub Yang, Ji-Seok Yoon
> Naval R&D Center, Hanwha Systems, Pangyo, Korea
> *Journal of The Korea Society of Computer and Information (JKSCI)*, Vol. 30 No. 1, pp. 33–40, January 2025
> DOI: `10.9708/jksci.2025.30.01.033`

Key results: single ConvNeXt-tiny → **90.25%** accuracy; 6-model softmax ensemble → **≥ 92%**.

---

## Repository layout

```
Real-time-Explainable-Automatic-Target-Recognition-System/
├── REATS/                          # Primary system — full ATR pipeline
│   ├── modules/
│   │   ├── module_a_detector.py    # YOLOv4 IR detector (pure PyTorch)
│   │   ├── module_b_classifier.py  # ConvNeXt-tiny × 6 ensemble classifier
│   │   ├── module_c_xai.py         # Grad-CAM / SHAP / MC Dropout XAI engine
│   │   └── module_d_dashboard.py   # Streamlit operator dashboard
│   ├── notebooks/
│   │   └── 00_baseline.ipynb       # End-to-end training walkthrough
│   ├── data/                       # Excluded from git — use .gitkeep placeholders
│   ├── checkpoints/                # Excluded from git
│   ├── runs/                       # MLflow logs — excluded from git
│   ├── dataset_validator.py        # Validates / organises train/val/test split
│   ├── generate_flir_fallback.py   # Synthetic + FLIR-remapped data generator
│   ├── verify_env.py               # Environment smoke test
│   └── requirements.txt
└── cadet_atr_project/              # Research scaffold — domain adaptation experiments
    └── cadet_atr/
        ├── adaptation/strategies.py  # 4 domain adaptation strategies
        ├── data/                     # Dataset classes + Kornia augmentation
        ├── models/convnext.py        # ConvNeXt builder
        ├── training/trainer.py       # Training loop + W&B logging
        ├── evaluation/evaluator.py   # Domain gap measurement
        ├── utils/                    # Config dataclass + visualisation
        ├── generate_synthetic.py     # Stable Diffusion synthetic IR generator
        └── run_experiment.py         # CLI entry point
```

---

## Target classes

### REATS (Naval CMS — Do et al. 2025)
| Class | Description |
|-------|-------------|
| `F16` | Fixed-wing fighter |
| `LYNX` | Naval helicopter |
| `MiG19` | Fixed-wing fighter |
| `MiG21` | Fixed-wing fighter |
| `PKG` | Anti-ship missile |
| `PTG` | Patrol torpedo boat |

### cadet_atr (expanded airborne + ground set)
| Class | Type |
|-------|------|
| `fixed_wing` | Airborne |
| `rotary_wing` | Airborne |
| `uav` | Airborne |
| `vessel` | Surface |
| `vehicle_ground` | Ground |
| `vehicle_apc` | Ground |

---

## Performance targets

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 92% |
| ECE (calibration error) | ≤ 0.05 |
| mAP@0.5 (detection) | ≥ 75% |
| End-to-end latency | ≤ 40 ms/frame |
| Faithfulness AUC | ≥ 0.80 |
| FPS | ≥ 20 |

---

## REATS architecture

```
IR Frame
  └─► Module A  module_a_detector.py
                YOLOv4 (CSPDarknet53 + SPP + PANet) — pure PyTorch, no ultralytics
                detect()   → list of {bbox, conf, class_id}
                crop_roi() → padded ROI crops per detection

  └─► Module B  module_b_classifier.py
                ConvNeXt-tiny × 6, softmax-averaging ensemble
                train_full_pipeline() — single model
                train_ensemble()      — all 6 models
                TemperatureScaler     — post-hoc ECE calibration

  └─► Module C  module_c_xai.py
                GradCAMExplainer / GradCAMPlusPlusExplainer / EigenCAMExplainer
                SHAPExplainer (DeepExplainer), LIMEExplainer
                MCDropoutWrapper → predictive entropy → OOD flag
                faithfulness_auc() — deletion / insertion curve AUC

  └─► Module D  module_d_dashboard.py
                Streamlit 4-tab UI
                  Live Analysis | Batch Processing | Calibration | About
```

Data flows: Module A crops → Module B classifies → Module C explains → Module D displays.

---

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended

### Installation

```bash
python -m venv /home/user/reats_env
source /home/user/reats_env/bin/activate
pip install -r REATS/requirements.txt
```

> **Note:** The system Python on Debian ships with a managed `blinker 1.7.0` that blocks `pip install`. Always activate the venv first.

Verify the environment:

```bash
python REATS/verify_env.py
```

---

## REATS quickstart

### 1. Prepare data

Expected split: **170 train / 30 val / 200 test** per class.

```bash
# Validate existing split
python REATS/dataset_validator.py

# Auto-organise images from a raw folder
python REATS/dataset_validator.py --source raw/ --organize
```

### 2. Generate training data (if no real IR data)

```bash
# Synthetic only
python REATS/generate_flir_fallback.py --out REATS/data/

# Remap from FLIR ADAS dataset
python REATS/generate_flir_fallback.py --flir /path/to/flir_adas/ --out REATS/data/
```

### 3. Train Module B classifier

```bash
cd REATS
python modules/module_b_classifier.py
```

MLflow logs → `REATS/runs/` | Checkpoints → `REATS/checkpoints/`

### 4. Run the operator dashboard

```bash
cd REATS
streamlit run modules/module_d_dashboard.py
```

---

## cadet_atr quickstart

Run all commands from `cadet_atr_project/cadet_atr/`.

```bash
# Smoke test — no GPU or real data required
python smoke_test.py

# Train synthetic baseline
python run_experiment.py --mode baseline_only

# Run a single domain adaptation strategy
python run_experiment.py --mode adapt --strategy histogram     --checkpoint checkpoints/baseline_best.pt
python run_experiment.py --mode adapt --strategy domain_random
python run_experiment.py --mode adapt --strategy finetune      --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode adapt --strategy dann          --checkpoint checkpoints/domain_random_best.pt

# Measure domain gap on a saved checkpoint
python run_experiment.py --mode gap_only --checkpoint checkpoints/dann_best.pt

# Full pipeline (all 4 strategies sequentially)
python run_experiment.py --mode full
```

### Domain adaptation strategies

| # | Strategy | Key component |
|---|----------|---------------|
| 1 | Histogram matching | `build_reference_histogram` + `apply_histogram_matching` |
| 2 | Domain randomisation | `BackgroundSwapDataset` (extended augmentation) |
| 3 | Fine-tuning on real IR | `RealDataFinetuner.finetune(mode=head_only\|full\|layer_wise)` |
| 4 | DANN / GRL | `DANNModel` (ConvNeXt + Gradient Reversal Layer + domain classifier) |

`DANNModel.forward(x)` returns class logits for deployment; pass `return_domain=True` during training to get both class and domain logits.

---

## DANN smoke test

```python
from adaptation.strategies import DANNModel
import torch

model = DANNModel(num_classes=6)
dummy = torch.randn(4, 3, 224, 224)

# Inference — domain head inactive
cls_logits = model(dummy)
assert cls_logits.shape == (4, 6)

# Training — both heads active
cls_logits, dom_logits = model(dummy, return_domain=True)
assert dom_logits.shape == (4, 2)

dom_logits.sum().backward()
print("GRL gradients flow correctly.")
```

---

## ONNX export

```python
import torch
from adaptation.strategies import DANNModel

model = DANNModel(num_classes=6)
model.load_state_dict(torch.load("checkpoints/dann_best.pt"))
model.eval()

torch.onnx.export(
    model,
    torch.randn(1, 3, 224, 224),
    "checkpoints/dann_best.onnx",
    input_names=["ir_image"],
    output_names=["class_logits"],
    opset_version=17,
)
```

---

## Results

| Experiment | Synth acc | Real acc | Domain gap |
|------------|-----------|----------|------------|
| Paper baseline — Do et al. (2025) | ~90.25% | — | — |
| Synthetic baseline | TBD | TBD | TBD |
| + Histogram matching | TBD | TBD | TBD |
| + Domain randomisation | TBD | TBD | TBD |
| + Fine-tuning on real data | TBD | TBD | TBD |
| + DANN (Strategy 4) | TBD | TBD | TBD |
| 6-model ensemble (REATS) | — | ≥ 92% | — |

Target domain gap: < 10% on the 6-class set.

---

## References

- Do, G.-S., Park, J.-M., Jang, W.-S., Yang, Y.-S., & Yoon, J.-S. (2025). A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System. *JKSCI*, 30(1), 33–40. https://doi.org/10.9708/jksci.2025.30.01.033
- Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. *JMLR*, 17(59), 1–35. https://arxiv.org/abs/1505.07818
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*. https://arxiv.org/abs/2201.03545
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
