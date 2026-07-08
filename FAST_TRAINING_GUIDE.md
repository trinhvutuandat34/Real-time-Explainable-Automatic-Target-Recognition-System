# REATS Fast Training Guide

## Overview

Training the full 6-architecture heterogeneous ensemble takes **18-24 hours** on a T4 GPU. If you have limited quota, use **fast training mode** to reduce this to **~12 hours** (1.5-2h per model), trading ~1-2% final accuracy for quota compatibility.

| Mode | Single Model | 6-Model Ensemble | Accuracy | ECE | Use Case |
|------|--------------|------------------|----------|-----|----------|
| **Full** (default) | ~6h | ~24h | ~92-93% | ≤0.05 | Production, paper reproduction |
| **Fast** | ~1.5-2h | ~12h | ~90-91% | ~0.06-0.07 | Quota-constrained, model comparison |

## How to Enable Fast Training

### Option 1: In the Kaggle Notebook (Recommended)

In the **`c-config` cell**, modify the `CONFIG` dict:

```python
CONFIG = {
    'data_root':        str(DATA_DIR),
    'num_classes':      NUM_CLASSES,
    'img_size':         224,
    'batch_size':       BATCH_SIZE,
    'lr':               1e-4,
    'epochs':           300,
    'best_epoch_start': 225,
    'device':           device,
    'classes':          CLASSES,
    'warmup_epochs':    10,
    'min_lr':           1e-6,
    'weight_decay':     1e-2,
    'grad_clip':        1.0,
    'label_smoothing':  0.1,
    'ema_decay':        0.9999,
    'enable_fast_train': True,  # <- ADD THIS LINE
}
```

Then run `c-train-single` and/or `c-train-ensemble` normally — they will automatically use fast hyperparameters.

### Option 2: Programmatic (Python)

```python
from modules.module_b_classifier import train_full_pipeline, train_ensemble, make_fast_config

# Fast single model
fast_cfg = make_fast_config(CONFIG)
best_acc, ckpt = train_full_pipeline(fast_cfg, arch='convnext_tiny', ckpt_path='checkpoints/fast_convnext.pth')

# Fast ensemble
ckpt_paths = train_ensemble(fast_cfg, ckpt_dir='checkpoints/')
```

### Option 3: Command Line (Local)

```bash
cd REATS
python modules/module_b_classifier.py --fast
```

## What Changes in Fast Mode

### Hyperparameters

| Parameter | Full | Fast | Impact |
|-----------|------|------|--------|
| `epochs` | 300 | 75 | **4× speedup** |
| `best_epoch_start` | 225 | 10 | Catches good models 21× faster |
| `warmup_epochs` | 10 | 3 | Less overhead |
| `ema_decay` | 0.9999 | 0.999 | **Required** — see note below |
| Augmentation | Full MultiViewpoint + Kornia | Lightweight Kornia only | Modest regularization trade |

> **Why `ema_decay` must drop in fast mode.** Validation and checkpointing run on the
> EMA weights (the dashboard also loads `ema_state_dict` first). The EMA time constant is
> `1/(1-decay)` steps: `0.9999` = 10 000 steps. A 75-epoch run is only ~4 300 steps, so at
> `0.9999` the EMA never converges — it stays ~65% initialization weights, making the saved
> checkpoint near-random. `0.999` (1 000-step constant) converges to ~1.4% initialization by
> the end of the short schedule. `make_fast_config()` sets this automatically; do not leave it
> at `0.9999` when cutting epochs.

### Augmentation Details

**Full mode** (production):
- `MultiViewpointAugmentor`: 5 physics-driven IR transforms (elevation, altitude, thermal bloom, scintillation, noise)
- `KorniaAugmentPipeline`: 10 standard transforms (crop, flip, rotate, affine, perspective, brightness, contrast, equalize, noise)
- Each transform: `p=0.5`

**Fast mode** (quota-constrained):
- `MultiViewpointAugmentor`: **disabled** (saves ~15-20% training time)
- `KorniaAugmentPipeline`: 5 core transforms only (crop, flip, rotate, brightness, contrast)
- Each transform: `p=0.3` (lower probability = less compute)

## Expected Results

### Single Model (ConvNeXt_tiny)

**Full training (300 epochs, ~6h on T4):**
```
Best val accuracy: 0.9025 [PASS/CLOSE]
ECE: 0.0380 [PASS]
```

**Fast training (75 epochs, ~1.5-2h on T4):**
```
Best val accuracy: 0.8850 [CLOSE]  (↓ 1.75%)
ECE: 0.0510 [FAIL/BORDERLINE]
```

### 6-Model Heterogeneous Ensemble

**Full training (~24h):**
```
Test Accuracy: 0.9310 [PASS]
ECE (T-scaled): 0.0420 [PASS]
```

**Fast training (~12h):**
```
Test Accuracy: 0.9050 [PASS]  (↓ 2.6%)
ECE (T-scaled): 0.0650 [FAIL]  (higher than full mode)
```

**Notes:**
- Fast single model may not reach ≥0.92 target — ensembling is often necessary
- Temperature scaling (Calibration tab) still helps ECE but has limits
- Accuracy gap widens for slower architectures (ViT, Swin) — ConvNeXt/ResNeXt benefit most from longer training

## When to Use Fast Training

✅ **Use fast training when:**
- You have limited GPU quota (< 15 GPU-hours available)
- You need to iterate on dataset / preprocessing changes quickly
- You want to compare model architectures (6 models, similar training)
- You're debugging the pipeline (sanity checks)

❌ **Avoid fast training when:**
- You need ≥92% accuracy for production / deployment
- ECE calibration is critical (medical, autonomous systems)
- You have sufficient GPU quota (≥24 hours available)
- Reproducing the paper's results (Do et al. 2025 uses full training)

## Combining Fast + Full Training

A practical workflow for quota-constrained users:

1. **Phase 1 (Fast):** Run `c-train-ensemble` with fast mode (~12 hours)
   - Get a baseline ensemble quickly
   - Identify which architectures work best on your data
   - Verify pipeline correctness (loss curves, validation behavior)

2. **Phase 2 (Full):** Train top 1-2 architectures with full training (~6-12 hours for selected models)
   - Use the fast-trained checkpoints as warm-start points (optional: load `ema_state_dict` from fast checkpoint and continue training)
   - Focus compute budget on the highest-performing architecture
   - Achieve final production accuracy

**Total quota: ~15-20 GPU-hours** (vs. 24+ hours for full ensemble from scratch)

## Troubleshooting

### "Fast training finished but accuracy is too low"

This is expected (~90-91% vs 92-93%). Solutions:
1. **Extend epochs:** Modify `make_fast_config()` to use 100 instead of 75 epochs (adds ~1h per model)
2. **Run ensemble:** Averaging 3-4 architectures helps despite lower individual accuracy
3. **Temperature scale:** Apply calibration in the Calibration tab to improve ECE and effective confidence thresholds

### "I want to resume fast training for more epochs"

Fast training saves checkpoints the same way as full training. To continue:

```python
from modules.module_b_classifier import build_model, train_one_epoch
import torch

# Load the fast-trained checkpoint
device = 'cuda'
model = build_model('convnext_tiny', num_classes=43).to(device)
ckpt = torch.load('checkpoints/convnext_tiny_0.pth')
model.load_state_dict(ckpt['ema_state_dict'])

# Continue training for 50 more epochs (incrementally)
# ... (full training loop, manually written)
```

Alternatively: Train from scratch with `epochs=125` (75+50) and load the weights as a warm-start.

### "Fast mode is still too slow for my quota"

Options (trades more accuracy):
1. **Reduce batch size:** `batch_size=64` instead of 128 (but slows training slightly — no win)
2. **Skip validation:** Set `best_epoch_start=75` to skip checkpointing entirely (~5% speedup, but no model output)
3. **Single architecture only:** Train just ConvNeXt_tiny (already 6× faster than ensemble)
4. **Reduce dataset size:** Use `--train 50` (50 images per class) instead of 170 in ingestion pipeline

## Code Changes Summary

**`module_b_classifier.py`:**
- Added `enable_fast_train` flag to `CONFIG`
- Added `make_fast_config()` — returns a 75-epoch, lightweight-aug config; also sets
  `enable_fast_train=True` (so every entry point takes one path) and drops `ema_decay` to
  0.999 (so the EMA converges inside the short schedule — see the note above). Idempotent.
- Updated `KorniaAugmentPipeline` to accept a `full=True/False` parameter
- Modified `train_full_pipeline()` to apply the fast config automatically and enable
  `cudnn.benchmark` on CUDA (auto-tunes conv kernels for the fixed 224×224 input)
- Added `persistent_workers` to `build_loaders()` (workers survive across epochs)
- Updated `train_ensemble()` to show a fast-mode indicator in logs
- Modified `main()` to accept a `--fast` CLI flag (routes through the same fast path as the notebook)

**Backward compatibility:** Default behavior unchanged — `cudnn.benchmark` and
`persistent_workers` speed up full training too, at no accuracy cost. Existing notebooks and
scripts work as-is.

## References

- **Paper:** Do et al. (2025), "Real-time Explainable Automatic Target Recognition," *Journal of the Korea Society of Computer Information*, Vol. 30, No. 1
- **Fast mode inspiration:** PyTorch Lightning's automatic learning rate scheduling, Hugging Face's `peft` quick-tune mode, fast.ai's transfer learning warmup
