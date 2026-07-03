---
marp: true
theme: default
paginate: true
title: REATS Gap Analysis
---

<!-- 1 -->
# Closing the distance to KCI (2025) and the professor's brief

Four gaps identified between the committed REATS codebase, Do et al. 2025
(JKSCI Vol. 30 No. 1), and the professor's additional instructions — each
traced to a paper citation or a stated requirement, then closed in code.

- Paper: Do et al., JKSCI 30(1), 2025 — target accuracy 92%
- Gaps closed: 4 / 4 — Smoke tests: 18 / 18 passing
- Branch: `claude/model-gaps-ensemble-metrics-e4xysn`

---

<!-- 2 -->
## Overview — what was missing

| # | Gap | Status |
|---|-----|--------|
| 1 | Ensemble configuration — 6 seeds of ConvNeXt_tiny instead of 6 distinct architectures | Closed |
| 2 | FAR/MR threat analysis — Precision/Recall only, never separated or threat-weighted | Closed |
| 3 | Hard negative mining — plain cross-entropy, no F16/MiG19/MiG21 confusion pressure | Closed |
| 4 | Operational policy mapping — static color only, no Warning/Track/Engagement link | Closed |

---

<!-- 3 -->
## Gap 1 — Six architectures, not six seeds

**Was:** ConvNeXt_tiny × 6 (seeds 42–47) — same inductive bias, correlated errors.

**Now:** ConvNeXt_tiny, ResNeXt50_32x4d, ViT_b_16, Swin_T, VGG16, ResNet18 — local
CNN features + global Transformer attention, same softmax-averaging combiner
(the paper's own best strategy, Table 6).

> "전통적인 CNN 기반 모델로 ResNet과 VGG 모델을 선정하였고, 최신의 CNN 기반
> 모델로는 ResNeXt와 ConvNeXt를 선정하였다. 추가적으로 Transformer 기반
> 모델로 Vision Transformer, Swin Transformer 모델을 선정하였다."
> — Do et al. 2025, §II.1.1 Transfer Learning

---

<!-- 4 -->
## Gap 1 — Implementation

`build_model(arch, num_classes, pretrained)` dispatches to a per-architecture
builder. Every checkpoint carries its own `"arch"` key so `load_ensemble()`
and the dashboard reconstruct the right network per file.

| Architecture | Family | Params | Verified |
|---|---|---:|---|
| convnext_tiny | CNN (modern) | 27,853,195 | (1,3,224,224)→(1,43) ✓ |
| resnext50 | CNN (modern) | 23,068,011 | ✓ |
| vit_b_16 | Transformer | 85,831,723 | ✓ |
| swin_t | Transformer | 27,552,421 | ✓ |
| vgg16 | CNN (traditional) | 134,436,715 | ✓ |
| resnet18 | CNN (traditional) | 11,198,571 | ✓ |

Ensemble over all 6 verified to output a valid probability distribution.
Training to convergence is GPU work (see slide 12).

---

<!-- 5 -->
## Gap 2 — Name the two failure modes that matter

**FAR (False Alarm Rate)** = FP / (FP + TP) = 1 − Precision
A civilian aircraft or friendly vessel misidentified as an enemy threat —
erodes operator trust, can trigger an unwarranted track/engagement call.

**MR (Miss / Omission Rate)** = FN / (FN + TP) = 1 − Recall
An actual instance of the class is never flagged. For a RED-threat class,
this is the worst failure: an infiltrating enemy platform crosses undetected.

No numeric target exists for FAR/MR in the paper — reported alongside
accuracy/ECE/mAP, not pass/fail gated.

---

<!-- 6 -->
## Gap 2 — Implementation

`modules/threat_metrics.py` → `compute_far_mr()` builds a confusion matrix →
per-class FAR/MR + macro + RED-threat aggregates.

- **metrics_report.py** — new section prints the 10 worst-MR classes plus
  macro/RED-threat FAR/MR alongside accuracy + ECE.
- **Dashboard Calibration tab** — uploaded labeled test-set ZIP now also
  renders a FAR/MR table with RED-threat call-out metrics.

36 RED-threat classes weighted separately · 43 total classes · 0 hardcoded
thresholds (sourced from `config`).

---

<!-- 7 -->
## Gap 3 — F-16 / MiG-19 / MiG-21 share a silhouette, and the errors

Plain cross-entropy weights every sample equally — rare, visually-ambiguous
fighters get the same gradient as easy, unambiguous ones.

> "MiG-19, MiG-21 라벨은 종종 F-16 라벨로 인식함을 함께 확인할 수 있었다.
> 이를 통해 대공 표적 중 헬기를 제외한 전투기 표적 간의 유사도가 가장
> 높다는 것을 알 수 있다."
> — Do et al. 2025, §III.3 Experimental Results (Table 5, Confusion Matrix)

---

<!-- 8 -->
## Gap 3 — Mine → oversample → fine-tune

1. **Mine** — `mine_hard_negatives()` flags samples in `CONFUSABLE_GROUPS`
   that are misclassified, or whose top1/top2 softmax margin is below 0.15.
2. **Oversample** — `HardNegativeDataset` repeats only the flagged indices
   (×4 default); easy examples keep their normal rate, no catastrophic forgetting.
3. **Fine-tune** — `finetune_on_hard_negatives()` runs a short, low-LR (1e-5)
   pass on top of an already-trained checkpoint — post-hoc, not a replacement
   for the 300-epoch schedule.

CLI: `python modules/hard_negative_mining.py --checkpoint <ckpt> --arch <arch>`

---

<!-- 9 -->
## Gap 4 — Confidence + threat level → CMS action tier

| Tier | Confidence | Ceiling | Meaning |
|---|---|---|---|
| NONE | < 0.50 | — | Below detection floor — no action |
| WARNING | ≥ 0.50 | YELLOW | Alert only — log + display, no tasking |
| TRACK | ≥ 0.75 | ORANGE | Cue sensors, maintain track — no engagement |
| ENGAGEMENT | ≥ 0.90 | RED only | Engagement-eligible — operator confirms per ROE |

ORANGE/YELLOW targets can never reach ENGAGEMENT regardless of confidence —
a false engagement call on a misclassified civilian/friendly target is the
costliest failure the system can produce.

---

<!-- 10 -->
## Gap 4 — Implementation

**config/targets.yaml** — new `operational_policy` section (thresholds +
threat_level ceiling), loaded and exported as `config.OPERATIONAL_POLICY`.
No thresholds hardcoded in application code.

**modules/threat_policy.py** — `map_confidence_to_policy(confidence,
threat_level)`, wired into the Live Analysis cards, iPhone Live Feed, and the
About tab's policy table.

Verified: RED@0.95→ENGAGEMENT, YELLOW@0.95→WARNING (capped), RED@0.10→NONE.

---

<!-- 11 -->
## Verification — 18/18 smoke tests, including 4 new ones

- `module_b.heterogeneous_ensemble` — all 6 architectures build, forward,
  and ensemble-average to a valid distribution
- `threat_metrics.far_mr` — `compute_far_mr()` checked against a hand-built
  confusion case
- `hard_negative_mining.mine` — `mine_hard_negatives()` run end-to-end on a
  synthetic dataset
- `threat_policy.map_confidence` — all three tier/ceiling cases verified

4 files newly added · 7 existing files touched. Dashboard module
import-checked directly (not just syntax-checked) after all wiring changes.

---

<!-- 12 -->
## Next — code is closed, training is next

1. **Train the 6 heterogeneous architectures to convergence** on Kaggle and
   confirm ensemble accuracy against the paper's reported 92% — architecture
   code is ready and unit-tested; the multi-day run is unchanged from the
   existing GPU-training backlog.
2. **Run the hard-negative fine-tune pass against a real trained checkpoint**
   to confirm it measurably reduces F16/MiG19/MiG21 confusion.
3. **Tune the 0.50/0.75/0.90 policy thresholds** against real FAR/MR
   measurements once a trained ensemble exists.

Full detail: `docs/gap_analysis_report.md`
