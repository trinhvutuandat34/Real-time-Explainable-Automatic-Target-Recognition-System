# Kaggle Fast Training Notebook Cells

Copy-paste these cells into your Kaggle notebook to enable fast training. Replace the existing cells with the same ID.

## Modified `c-config` Cell

Replace the existing `CONFIG` dict definition with this version:

```python
# ── Module B hyperparams ────────────────────────────────────────────────
# Set FAST_TRAINING = True to train in ~1.5-2h per model instead of ~6h
# This trades ~1-2% final accuracy for quota compatibility.
from modules.module_b_classifier import make_fast_config

FAST_TRAINING = False   # <- SET TO True FOR FAST MODE

CONFIG = {
    'data_root':        str(DATA_DIR),
    'num_classes':      NUM_CLASSES,
    'img_size':         224,
    'batch_size':       BATCH_SIZE,
    'lr':               1e-4,
    'epochs':           300,
    'best_epoch_start': 225,   # checkpoint only from epoch 225, per paper
    'device':           device,
    'classes':          CLASSES,
    'warmup_epochs':    10,
    'min_lr':           1e-6,
    'weight_decay':     1e-2,
    'grad_clip':        1.0,
    'label_smoothing':  0.1,
    'ema_decay':        0.9999,
    'enable_fast_train': False,
}

# Let make_fast_config apply the full set of fast overrides (epochs, warmup,
# best_epoch_start AND the critical ema_decay=0.999). Do NOT hand-edit just the
# epochs — leaving ema_decay at 0.9999 makes the short-schedule EMA (which gets
# validated and saved) never converge. This one call keeps them consistent.
if FAST_TRAINING:
    CONFIG = make_fast_config(CONFIG)

available_ds = [k for k, p in DATASET_INPUTS.items() if p.exists()]
missing_ds   = [k for k in DATASET_INPUTS if k not in available_ds]
print(f'Available datasets ({len(available_ds)}/{len(DATASET_INPUTS)}): {available_ds}')
if missing_ds:
    print(f'Missing (skipped): {missing_ds}')

mode_str = "FAST (75 epochs, lightweight aug, ema=0.999, ~1.5-2h/model)" if FAST_TRAINING else "FULL (300 epochs, ~6h/model)"
print(f'Training mode: {mode_str}')
print(f'Classes: {NUM_CLASSES} | Batch: {BATCH_SIZE} | Epochs: {CONFIG["epochs"]}')
```

## New Cell: Quick Ensemble Training (Fast)

Insert this before `c-train-ensemble` if you want to always train the ensemble in fast mode with minimal configuration:

```python
# ── QUICK ENSEMBLE (Fast Mode) ──────────────────────────────────────────
# Trains all 6 architectures in ~12 GPU-hours instead of ~24.
# Accuracy trades ~1-2% for quota compatibility.
# Disable FAST_TRAINING above to run full 300-epoch training instead.

import mlflow, time
from modules.module_b_classifier import ARCHITECTURES, train_ensemble

mlflow.set_tracking_uri(f'sqlite:///{RUNS_DIR}/mlflow.db')

if FAST_TRAINING:
    print(f'Fast ensemble: {len(ARCHITECTURES)} architectures, 75 epochs each')
    print(f'Expected wall time: ~12 GPU-hours on T4')
else:
    print(f'Full ensemble: {len(ARCHITECTURES)} architectures, 300 epochs each')
    print(f'Expected wall time: ~24 GPU-hours on T4')

print()

t0 = time.time()
ckpt_paths = train_ensemble(CONFIG, ckpt_dir=str(CKPT_DIR))
ensemble   = load_ensemble(ckpt_paths, num_classes=NUM_CLASSES, device=device)
ensemble.eval()
USE_ENSEMBLE = True
elapsed_h = (time.time() - t0) / 3600

print()
print(f'Ensemble ready: {len(ckpt_paths)} models')
print(f'Wall time: {elapsed_h:.1f} hours')
```

## Modified `c-train-single` Cell (For Reference)

If you want to also train a single model with fast mode, use:

```python
import mlflow, time

mlflow.set_tracking_uri(f'sqlite:///{RUNS_DIR}/mlflow.db')

mode_str = "fast" if FAST_TRAINING else "full"
print(f'Training ConvNeXt_tiny ({mode_str} mode, {CONFIG["epochs"]} epochs)')
print(f'Augmentor: {"lightweight Kornia (fast)" if FAST_TRAINING else "MultiViewpointAugmentor + KorniaAugmentPipeline (full)"}')
print(f'AMP + EMA + WarmupCosine | checkpoint -> {CKPT_SINGLE}')
print()

t0 = time.time()
best_val_acc, ckpt_path = train_full_pipeline(CONFIG, seed=42, ckpt_path=CKPT_SINGLE)
elapsed_min = (time.time() - t0) / 60

if best_val_acc >= 0.92:
    status = 'PASS'
elif best_val_acc >= 0.90:
    status = 'CLOSE  -> run ensemble cell below'
else:
    status = 'FAIL   -> check data, run ensemble cell below'

print(f'Best val acc : {best_val_acc:.4f}  [{status}]  (target >= 0.92 for full, >= 0.90 for fast)')
print(f'Wall time    : {elapsed_min:.1f} min')
```

## Usage Instructions

### For Fast Training (Quota-Constrained)

1. **In `c-config` cell:** Set `FAST_TRAINING = True`
2. **Run cells in order:**
   - `c-gpu` → `c-install` → `c-clone` → `c-config` (with `FAST_TRAINING = True`)
   - `c-ingest` → `c-fallback` → `c-train-single` (optional) → `c-train-ensemble` or Quick Ensemble cell above
   - `c-eval-metrics` → remaining evaluation cells
3. **Expected total time:** ~15-18 GPU-hours (vs. 30+ for full pipeline)

### For Full Training (Production)

1. **In `c-config` cell:** Set `FAST_TRAINING = False` (default)
2. Run cells normally — no other changes needed
3. **Expected total time:** ~30+ GPU-hours for single model + ensemble

## Example: Hybrid Approach

Best for limited quota (< 20 GPU-hours available):

**Cell 1: Fast Ensemble** (12 hours)
```python
FAST_TRAINING = True
# Run c-config, c-ingest, c-fallback, c-train-ensemble
```

**Cell 2: Evaluate & Cherry-Pick** (1-2 hours)
```python
# c-eval-metrics shows which architecture performed best
# e.g. ConvNeXt_tiny achieved 90.5%, ResNeXt50 achieved 89.8%, etc.
```

**Cell 3: Full Training of Winner** (6 hours)
```python
FAST_TRAINING = False
# Re-run c-config with the original 300-epoch settings
# Run c-train-single for just the best architecture (ConvNeXt_tiny)
# Re-run c-eval-metrics on the full-trained checkpoint
```

**Total: ~19 GPU-hours, still leaves room to tune and re-run if needed.**

## Monitoring Progress

Both full and fast training print epoch summaries every 10 epochs (full) or 5 epochs (fast):

```
[Epoch  10] loss=0.8234 acc=0.6521
[Epoch  15] loss=0.6123 acc=0.7314
...
[Epoch  70] New best: 0.8847 → checkpoints/convnext_tiny_0.pth
[Epoch  75] loss=0.3891 acc=0.8901
```

Best validation accuracy is logged at every epoch when it improves, with the checkpoint path.

## Reverting to Full Training

If you accidentally enabled fast training and want full training:

```python
# In a new cell:
FAST_TRAINING = False
from modules.module_b_classifier import make_fast_config
CONFIG = {
    'data_root':        str(DATA_DIR),
    # ... (rest of full CONFIG)
}
# (no enable_fast_train key, or set it to False)
# Then re-run c-train-single or c-train-ensemble
```

## Notes

- Fast training uses the same data loaders, architectures, and loss functions as full training
- Only hyperparameters (`epochs`, `best_epoch_start`) and augmentation complexity differ
- Checkpoints are 100% compatible — a fast-trained checkpoint loads normally into the dashboard
- MLflow experiment tracking continues to work (logs to the `REATS-Baseline_fast` experiment)
