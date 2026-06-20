# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Two codebases in one repo

| Directory | Purpose | Entry point |
|---|---|---|
| `REATS/` | Primary system — full ATR pipeline (Modules A–D) | `modules/module_*.py`, `notebooks/00_baseline.ipynb` |
| `cadet_atr_project/cadet_atr/` | Research scaffold — domain adaptation experiments | `run_experiment.py` |

The two systems use **different class taxonomies**:
- **REATS**: `F16, LYNX, MiG19, MiG21, PKG, PTG` (Naval CMS targets from Do et al. 2025)
- **cadet_atr**: `fixed_wing, rotary_wing, uav, vessel, vehicle_ground, vehicle_apc` (expanded airborne+ground set)

---

## Local development environment

System Python has a Debian-managed `blinker 1.7.0` that blocks `pip install`. Always use the venv:

```bash
source /home/user/reats_env/bin/activate
```

Install once:
```bash
python -m venv /home/user/reats_env
source /home/user/reats_env/bin/activate
pip install -r REATS/requirements.txt
```

Verify:
```bash
python REATS/verify_env.py
```

---

## Common REATS commands

```bash
# Validate dataset split (170 train / 30 val / 200 test per class)
python REATS/dataset_validator.py

# Auto-organise images from a raw folder into the split
python REATS/dataset_validator.py --source raw/ --organize

# Generate synthetic / FLIR-remapped training data
python REATS/generate_flir_fallback.py --out REATS/data/              # synth-only
python REATS/generate_flir_fallback.py --flir /path/to/flir_adas/ --out REATS/data/  # FLIR

# Train Module B (ConvNeXt, single model)
cd REATS && python modules/module_b_classifier.py

# Run the Streamlit dashboard
cd REATS && streamlit run modules/module_d_dashboard.py
```

MLflow experiment logs write to `REATS/runs/`; checkpoints to `REATS/checkpoints/`.

---

## Common cadet_atr commands

Run from `cadet_atr_project/cadet_atr/`:

```bash
# Smoke test (no GPU or real data required)
python smoke_test.py

# Train synthetic baseline
python run_experiment.py --mode baseline_only

# Run a single adaptation strategy
python run_experiment.py --mode adapt --strategy histogram    --checkpoint checkpoints/baseline_best.pt
python run_experiment.py --mode adapt --strategy domain_random
python run_experiment.py --mode adapt --strategy finetune    --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode adapt --strategy dann        --checkpoint checkpoints/domain_random_best.pt

# Measure domain gap on a saved checkpoint
python run_experiment.py --mode gap_only --checkpoint checkpoints/dann_best.pt

# Full pipeline (all 4 strategies sequentially)
python run_experiment.py --mode full
```

---

## REATS architecture

```
IR Frame
  └─► Module A  module_a_detector.py     YOLOv4 (CSPDarknet53 + SPP + PANet)
                                          detect() → list of {bbox, conf, class_id}
                                          crop_roi() pads each detection
  └─► Module B  module_b_classifier.py   ConvNeXt_tiny × 6 (softmax-averaging ensemble)
                                          train_full_pipeline() for one model
                                          train_ensemble() for all 6
                                          TemperatureScaler for post-hoc ECE calibration
  └─► Module C  module_c_xai.py          GradCAM / GradCAM++ / EigenCAM
                                          SHAP DeepExplainer, LIME
                                          MCDropoutWrapper (entropy → OOD flag)
                                          faithfulness deletion/insertion AUC
  └─► Module D  module_d_dashboard.py    Streamlit 4-tab UI
                                          (Live Analysis | Batch | Calibration | About)
```

**Data flow between modules**: Module A produces ROI crops → Module B classifies them → Module C explains the classification → Module D displays everything in real time.

**Module A is pure PyTorch** — no ultralytics. The `ultralytics` line in `requirements.txt` is only used by `verify_env.py`. `IRDetector.detect()` runs the full YOLOv4 forward pass + NMS internally.

**Module B training quirks**:
- `train_one_epoch` accepts optional `scaler` (AMP GradScaler) and `ema` (ModelEMA) — both `None` by default.
- Validation and checkpointing only run from epoch `CONFIG["best_epoch_start"]` (225/300) — this is intentional per the paper.
- `compute_ece` accepts `is_probs=False`; set `True` when passing ensemble output (already softmax).

---

## cadet_atr architecture

```
generate_synthetic.py        Stable Diffusion → data/synthetic/{class}/
data/dataset.py              SyntheticIRDataset / make_loaders() / make_real_loader()
data/augmentation.py         Kornia IR augmentation pipeline
models/convnext.py           build_model(model_name, num_classes) → ConvNeXt
training/trainer.py          Trainer.fit(train_loader, val_loader) → ckpt_path
evaluation/evaluator.py      measure_domain_gap() → {synth_acc, real_acc, domain_gap}
adaptation/strategies.py     4 strategies (see below)
utils/config.py              Config dataclass (cfg singleton) — all hyperparams
utils/visualise.py           plot_tsne, GradCAM, _extract_features
run_experiment.py            CLI dispatcher → run_adapt_strategy() / run_full_pipeline()
```

**4 domain adaptation strategies** in `adaptation/strategies.py`:
1. `histogram` — `build_reference_histogram` + `apply_histogram_matching`
2. `domain_random` — `BackgroundSwapDataset` (extended aug, no checkpoint needed)
3. `finetune` — `RealDataFinetuner.finetune(mode=head_only|full|layer_wise)`
4. `dann` — `DANNModel` (ConvNeXt backbone + GRL + domain classifier), `DANNTrainer.train()`

`DANNModel.forward(x, return_domain=False)` returns only class logits for deployment; pass `return_domain=True` during DANN training to get both class and domain logits.

---

## Git / data layout

`REATS/.gitignore` excludes `data/**`, `checkpoints/**`, `runs/**` (large files). Placeholder `.gitkeep` files track directory structure. The pattern used is:
```
data/**
!data/
!data/**/
!data/**/.gitkeep
```
Force-adding `.gitkeep` files requires `git add -f`; ordinary `git add` will ignore them.

Active development branch: `claude/friendly-gauss-howk55`

---

## Performance targets (from Do et al. 2025)

| Metric | Target |
|---|---|
| Classification accuracy | ≥ 92% |
| ECE (calibration) | ≤ 0.05 |
| mAP@0.5 (detection) | ≥ 75% |
| End-to-end latency | ≤ 40 ms/frame |
| Faithfulness AUC | ≥ 0.80 |
| FPS | ≥ 20 |

Single ConvNeXt_tiny achieves ~90.25%; the 6-model softmax ensemble pushes to ~92%.
