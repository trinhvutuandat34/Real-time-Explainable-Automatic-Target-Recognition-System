# REATS Gap Analysis: Implementation vs. KCI (2025) Paper vs. Professor's Requirements

**Date:** 2026-07-03
**Scope:** Four gaps identified between the previously-committed REATS codebase, Do et
al., *"A Study on Deep Learning-based Automatic Target Recognition System in IR Image
for Intelligent Combat Management System"* (JKSCI Vol. 30 No. 1, pp. 33–40, Jan. 2025),
and the professor's additional requirements. All four gaps are now closed in code on
branch `claude/model-gaps-ensemble-metrics-e4xysn`.

---

## Summary table

| # | Gap | Was | Now | Files |
|---|-----|-----|-----|-------|
| 1 | Ensemble configuration | 6 seeds of one architecture (ConvNeXt_tiny) | 6 distinct architectures (ConvNeXt_tiny, ResNeXt50, ViT_b_16, Swin_T, VGG16, ResNet18) | `modules/module_b_classifier.py`, `modules/module_d_dashboard.py`, `metrics_report.py` |
| 2 | FAR/MR threat metrics | Precision/Recall only | Explicit FAR (=1-Precision) and MR (=1-Recall), RED-threat-weighted | `modules/threat_metrics.py`, `metrics_report.py`, `modules/module_d_dashboard.py` |
| 3 | Hard negative mining | Plain cross-entropy over the full dataset | Confusable-class mining (F16/MiG19/MiG21) + oversampled fine-tune pass | `modules/hard_negative_mining.py` |
| 4 | Operational policy mapping | Static RED/ORANGE/YELLOW color only | Confidence + threat_level → Warning/Track/Engagement CMS tier | `config/targets.yaml`, `config/__init__.py`, `modules/threat_policy.py`, `modules/module_d_dashboard.py` |

---

## Gap 1 — Ensemble Model Configuration Method

### What the paper actually specifies

Do et al. (2025), Section II.1.1 ("Transfer Learning") and Section III.2 ("Experimental
Environment"), state the system uses **six distinct pretrained architectures**, not six
seeds of one:

> "따라서 제안하는 시스템은 총 6종의 모델로 전통적인 CNN 기반 모델로 ResNet[6]과
> VGG[7] 모델을 선정하였고, 최신의 CNN 기반 모델로는 ResNeXt[8]와 ConvNeXt[9]를
> 선정하였다. 추가적으로 Transformer 기반 모델로 Vision Transformer[10], Swin
> Transformer[11] 모델을 선정하였다."
>
> *("The proposed system selects a total of 6 models: traditional CNN-based models
> ResNet and VGG; state-of-the-art CNN-based models ResNeXt and ConvNeXt; and
> Transformer-based models Vision Transformer and Swin Transformer.")*

Table 3/4 confirms the exact variants used: **ResNet18, ConvNeXt_tiny, VGG16,
ViT_b_16, ResNeXt50_32x4d, Swin_T** (PyTorch Hub weights). Table 4 reports each
architecture's individual accuracy with and without augmentation; Table 6 reports 5
ensembling strategies (Geometric/Voting/Averaging and two hybrids) applied on top of
those 6 models, with **Averaging** the best performer at 0.92 accuracy — which is why
REATS already used softmax averaging as the combination rule. The gap was never the
combination rule; it was that all 6 inputs to that averaging were the same architecture
with only the random seed varied.

### Why heterogeneity matters (not just paper-compliance)

Six seeds of ConvNeXt_tiny only reduces variance from random initialization/data
order — the models share the same inductive bias (local convolutional receptive
fields) and tend to make correlated errors on the same hard examples. Six different
architectures — CNNs (ResNet18, VGG16, ResNeXt50, ConvNeXt_tiny) plus Transformers
(ViT_b_16, Swin_T) — combine local texture/edge features with global
self-attention context, so their error modes are less correlated and averaging
actually cancels more mistakes rather than just averaging noise.

### What changed in code

`modules/module_b_classifier.py`:
- Added `build_resnet18`, `build_vgg16`, `build_resnext50`, `build_vit_b_16`,
  `build_swin_t` alongside the existing `build_convnext`, each replacing the
  ImageNet-pretrained head with a `Linear(·, NUM_CLASSES)` for the REATS taxonomy.
- Added `ARCHITECTURES = ["convnext_tiny", "resnext50", "vit_b_16", "swin_t", "vgg16", "resnet18"]`
  and a `build_model(arch, num_classes, pretrained)` dispatcher.
- `train_ensemble(cfg, architectures=None)` now trains **one model per architecture**
  (default: all 6) instead of 6 seeds of one; each checkpoint records its own `"arch"`
  key.
- `load_ensemble()` and the dashboard's `load_pipeline()` read `arch` back out of each
  checkpoint and reconstruct the matching network, so a heterogeneous ensemble loads
  correctly. Checkpoints saved before this field existed default to `convnext_tiny`
  for backward compatibility with any already-trained single-architecture runs.
- `EnsembleClassifier.forward()` needed **no change** — it already only calls
  `m(x)` and softmax-averages, so it was architecture-agnostic from the start.

Verified: all 6 builders instantiate correctly, forward a `(1, 3, 224, 224)` tensor to
`(1, 43)` logits, and the resulting `EnsembleClassifier` over all 6 produces a valid
probability distribution (`smoke_test.py::module_b.heterogeneous_ensemble`).

### What is *not* done (requires GPU)

Actually training all 6 architectures to convergence (300 epochs each per the paper's
hyperparameters) requires a GPU run on Kaggle, matching the existing pending item in
`MEMORY.md` for the homogeneous ensemble. The architecture and training code changes
are complete and tested at the unit level; the multi-day training run itself is future
work, unchanged from the project's existing GPU-training backlog.

---

## Gap 2 — Missing False Alarm Rate (FAR) / Miss Rate (MR) Analysis

### The problem with precision/recall alone

The existing `per_class_metrics()` in `module_b_classifier.py` reports precision,
recall, and F1 per class — mathematically sufficient, but it does not name or
foreground the two failure modes that matter operationally on a naval CMS:

- **False Alarm (FAR = 1 − Precision = FP / (FP + TP))** — the dashboard raises a
  threat alarm on something that isn't actually that threat class (e.g. a civilian
  aircraft or friendly vessel misclassified as an enemy platform). Costly because it
  erodes operator trust and can trigger an unwarranted track/engagement recommendation.
- **Miss / Omission (MR = 1 − Recall = FN / (FN + TP))** — an actual instance of the
  class is never flagged at all. For a RED-threat class (fighter, bomber, attack
  helicopter, armed UAV/vessel) this is the worst possible failure: an infiltrating
  enemy platform crosses undetected.

### What changed in code

New module `modules/threat_metrics.py`:
- `compute_far_mr(labels, preds, classes=CLASSES)` builds a confusion matrix and
  derives per-class `{FAR, MR, TP, FP, FN}`, plus `macro_FAR`/`macro_MR` (mean over
  all 43 classes) and `red_threat_FAR`/`red_threat_MR` (mean restricted to
  `config.RED_THREATS` — the highest-consequence subset).
- `far_mr_from_model(model, loader, device)` runs a model over a labeled loader and
  calls `compute_far_mr`.
- `format_report(report, top_n)` renders a sorted (worst-MR-first) text table for CLI
  use.

Wired in two places:
- `metrics_report.py` — a new `[Battlefield threat analysis — False Alarm Rate / Miss
  Rate]` section runs immediately after accuracy/ECE, printing the top-10
  highest-miss-rate classes plus macro and RED-threat aggregates.
- `modules/module_d_dashboard.py` Calibration tab — this tab already runs labeled
  inference over an uploaded test-set ZIP to compute ECE; it now also tracks per-sample
  true/predicted indices and renders a "Battlefield Threat Analysis — FAR / MR" table
  with `st.metric` call-outs for macro and RED-threat FAR/MR.

No paper or professor-specified numeric target exists for FAR/MR (unlike accuracy/ECE/
mAP), so `metrics_report.py`'s `TARGETS` dict is intentionally left unchanged — FAR/MR
are reported, not pass/fail gated.

Verified: `compute_far_mr` unit-tested against a hand-constructed confusion case
(`smoke_test.py::threat_metrics.far_mr`) and against the full 43-class identity + 2
forced errors, producing the expected FN/FP counts.

---

## Gap 3 — Hard Negative Mining Not Implemented

### The problem the paper's own results point to

Do et al.'s confusion matrix (Table 5, Geometric-ensemble ConvNeXt on the paper's
6-class subset: F-16, LYNX, MiG-19, MiG-21, PKG, PTG) shows F-16, MiG-19, and MiG-21
are the three lowest-accuracy classes, and explicitly calls out cross-confusion:

> "MiG-19, MiG-21 라벨은 종종 F-16 라벨로 인식함을 함께 확인할 수 있었다. 이를 통해
> 대공 표적 중 헬기를 제외한 전투기 표적 간의 유사도가 가장 높다는 것을 알 수
> 있다."
>
> *("MiG-19 and MiG-21 labels were often recognized as F-16 — the fighter jets [as
> opposed to helicopters] have the highest visual similarity among air targets.")*

The previously-committed training code (`train_one_epoch` /
`LabelSmoothingCrossEntropy`) treats every training sample identically. Rare,
visually-ambiguous examples from a confusable class group get the same gradient
weight as easy, unambiguous ones, so the model has no extra pressure to resolve
exactly the confusion the paper measured.

### What changed in code

New module `modules/hard_negative_mining.py`:
- `CONFUSABLE_GROUPS = [{"F16", "MiG19", "MiG21"}]` — extensible list of class-name
  sets known to be visually similar; add more groups as new confusions are found
  (e.g. from a future confusion-matrix run over the full 43-class REATS taxonomy).
- `mine_hard_negatives(model, dataset, device, confusable_ids=None, margin_thresh=0.15)`
  — for every sample whose true label falls in a confusable group, flags it as a hard
  negative if the model misclassifies it, **or** if its softmax top-1/top-2 margin is
  below `margin_thresh` (the model is unsure even when it happens to be right).
- `HardNegativeDataset(base_dataset, hard_indices, oversample_factor=4)` — wraps the
  base training set, repeating only the flagged hard-negative indices so easy
  examples are not crowded out and the model does not catastrophically forget other
  classes.
- `finetune_on_hard_negatives(model, base_train_dataset, val_loader, device, ...)` —
  a short (default 15-epoch), low-LR (default `1e-5`) fine-tuning pass over the
  oversampled subset, run **on top of** an already fully-trained checkpoint — this is
  a post-hoc addition to the paper's 300-epoch schedule, not a replacement for it.
- CLI entry point: `python modules/hard_negative_mining.py --checkpoint <ckpt> --arch <arch>`.

Verified: `mine_hard_negatives` unit-tested end-to-end against a random-init ConvNeXt
model and a synthetic dataset spanning the confusable-group indices
(`smoke_test.py::hard_negative_mining.mine`).

---

## Gap 4 — Threshold-Operational Policy (Warning / Track / Engagement) Not Implemented

### The gap

The dashboard previously mapped a detected class straight to a static RED/ORANGE/
YELLOW color sourced from `targets.yaml` metadata — a description of *what kind* of
target it might be, with no link to *what the operator should do about it*. The
professor's requirement is to connect real-time confidence (and, implicitly, the
FAR/MR trade-off from Gap 2) to the actual CMS code-of-conduct action tiers: **Warning
→ Track → Engagement**.

### What changed in code

`config/targets.yaml` gained an `operational_policy` section (single source of truth,
same pattern as the class taxonomy):

```yaml
operational_policy:
  confidence_thresholds:
    WARNING: 0.50
    TRACK: 0.75
    ENGAGEMENT: 0.90
  threat_level_ceiling:
    RED: ENGAGEMENT
    ORANGE: TRACK
    YELLOW: WARNING
```

`config/__init__.py` loads and exports it as `OPERATIONAL_POLICY`.

New module `modules/threat_policy.py`:
- `map_confidence_to_policy(confidence, threat_level) -> str` — returns one of
  `NONE / WARNING / TRACK / ENGAGEMENT`. The confidence thresholds set the tier a
  detection would earn on its own; the `threat_level_ceiling` then caps it, so an
  ORANGE or YELLOW target can never reach `ENGAGEMENT` no matter how confident the
  classifier is — a false engagement recommendation on a misclassified civilian or
  friendly target is the costliest failure mode the system can produce, which is
  exactly the risk Gap 2's FAR metric quantifies.
- `POLICY_DESCRIPTION` — human-readable description of each tier for display.

Wired into `modules/module_d_dashboard.py`:
- Live Analysis tab: each per-detection card now shows a `CMS policy` metric and
  description, and the expander label is suffixed with the policy tier and icon.
- iPhone Live Feed tab: each `det_cards` entry carries its computed `policy`, shown
  next to the threat-level icon in the live markdown feed.
- About tab: a new "CMS operational policy" table renders the confidence thresholds,
  the threat_level each tier caps out at, and its description, plus a caption
  explaining the RED-only engagement-authority rule.

Verified: `map_confidence_to_policy` unit-tested against all three cases —
high-confidence RED → `ENGAGEMENT`; high-confidence YELLOW capped to `WARNING`;
low-confidence RED → `NONE` (`smoke_test.py::threat_policy.map_confidence`). The
dashboard module was also import-checked end-to-end after the change.

---

## Verification

All four gaps are covered by new or extended smoke tests in `REATS/smoke_test.py`,
run alongside the full existing suite:

```
Results: 18/18 passed
  ...
  ✓ module_b.heterogeneous_ensemble
  ✓ threat_metrics.far_mr
  ✓ hard_negative_mining.mine
  ✓ threat_policy.map_confidence
  ...
```

`REATS/modules/module_d_dashboard.py` was import-checked directly (not just
`py_compile`) to confirm the new `threat_policy`/`threat_metrics` wiring doesn't break
module load. `config/targets.yaml` was re-parsed to confirm the new
`operational_policy` section coexists with the existing 43-class taxonomy without
disturbing `CLASSES`/`NUM_CLASSES`/`RED_THREATS`.

## What remains as future GPU work

- Actually training the 6 heterogeneous architectures to convergence and confirming
  the ensemble accuracy against the paper's reported 92% (Gap 1's architecture code is
  ready; the training run itself needs a GPU session, same as the existing homogeneous
  ensemble backlog in `MEMORY.md`).
- Running `hard_negative_mining.py` against a real trained checkpoint to confirm the
  fine-tune pass measurably reduces F16/MiG19/MiG21 confusion (the mining/fine-tune
  logic is implemented and unit-tested against synthetic data, but its effect on real
  accuracy can only be measured with a trained model and real val data).
- Tuning the `operational_policy` confidence thresholds against real FAR/MR
  measurements once a trained ensemble is available — the current 0.50/0.75/0.90
  values are a reasonable starting point, not empirically fit.
