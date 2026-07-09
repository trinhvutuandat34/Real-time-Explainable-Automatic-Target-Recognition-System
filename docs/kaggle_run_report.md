# REATS Kaggle GPU Run Report

**Compiled:** 2026-07-09
**Covers:** two Kaggle GPU sessions — the 2026-07-04 full-training run (`real-time-ex-03`, already logged in `MEMORY.md` but never committed as its own report) and the most recent fast-training-mode session (undated in the reviewed logs; must be ≥2026-07-08, since it uses fast-training mode, which didn't exist in the codebase before that date)
**Purpose:** single accurate source document for slide/report use — supersedes ad hoc verbal summaries of either run

> **Provenance note:** the 2026-07-04 figures are reconstructed from `MEMORY.md`'s "Kaggle run results" entry (the original standalone report was never committed to this repo). The most-recent-run figures are transcribed from that session's own notebook stdout logs and generated artifacts (two log files, ~18k and ~14k lines, plus confusion-matrix/sample-grid/pipeline-demo images and an MLflow/Streamlit log) as reviewed in a separate session — this document did not have direct access to those raw files to re-verify them, so treat the numbers as faithfully transcribed, not independently re-derived. Every number below is cross-checked for internal and architectural consistency against the current codebase (`MEMORY.md`, `docs/gap_analysis_report.md`, `REATS/config/targets.yaml`) where possible; anywhere a conflict turned up, it's called out rather than silently resolved.

---

## Executive summary

- **Classification (Module B) clears the paper's 92% accuracy target in both runs** — 93.12% (full 300-epoch training) and 95.50% (fast 75-epoch training) — but the headline number is inflated by a data-composition problem, not pure model quality: only 8–10 of 43 classes have any real-image backing, and synthetic-only classes score 98.1% vs. 87.0% for real-backed classes. **True field-relevant accuracy is closer to ~87%.**
- **Calibration (ECE) only passes after temperature scaling** in both runs (raw ECE 0.07–0.12, temperature-scaled 0.02–0.04, target ≤0.05) — expected, and the fix (`TemperatureScaler`) is already wired into the pipeline.
- **Explainability faithfulness fails its ≥0.80 target in both runs** (deletion AUC 0.25–0.49, insertion AUC 0.75) — the weakest area of the system relative to target, and plausibly tied to the same synthetic-data-dominance problem.
- **Detection (Module A) is not yet functional.** No run has produced a usable detector: mAP@0.5 crawled to 0.0011 over 20 epochs in the one training attempt made so far, against a ≥75% target, and produces thousands of spurious boxes on trivial test input.
- **End-to-end latency fails its 40ms target in both runs** (82–112ms), though part of that gap is a measurement-methodology fix between runs, not a pure speed regression (see §5).
- **The dashboard (Module D) and full A→B→C→D pipeline run end-to-end successfully** in the most recent session, including a public tunnel URL and live batch classification.
- The 6-architecture heterogeneous ensemble (Gap 1 of `docs/gap_analysis_report.md`) has **never actually been trained** — both runs' single ConvNeXt-tiny cleared 92% on its own, so the notebook's auto-skip logic (`best_val_acc < 0.92` gate) never triggered it. No ensemble accuracy number exists yet.

---

## 1. Background

REATS (Real-time Explainable Automatic Target Recognition System) is an IR/thermal military-target recognition pipeline built on Do et al. (2025, JKSCI Vol. 30 No. 1) and extended to a 43-class taxonomy (AIR/GROUND/NAVAL) with a heterogeneous 6-architecture classifier ensemble, post-hoc calibration, multi-method explainability (XAI), battlefield FAR/MR threat metrics, a Warning/Track/Engagement operational policy, and a live operator dashboard. Five modules:

```
Module A (detector, YOLOv4)  →  Module B (classifier ensemble)  →  Module C (XAI)  →  Module D (dashboard)
                                                                                              ▲
                                                          Module E (phone-camera streamer) ───┘
```

Full architecture detail: `README.md`, `CLAUDE.md`. Paper-vs-implementation gap closure: `docs/gap_analysis_report.md` / `docs/gap_analysis_slides.md`.

---

## 2. Runs covered in this report

| Label | Kaggle kernel | Date | Training | What it produced |
|---|---|---|---|---|
| **Run A** | `real-time-ex-03` | 2026-07-04 | Single ConvNeXt-tiny, full 300 epochs (~6h) | Full metrics suite (accuracy/ECE/faithfulness/latency); no detector training attempted |
| **Run B** | not recorded in reviewed logs | ≥2026-07-08 | Module A (detector) fine-tune attempt, 20 epochs fast mode | Detector training only — did not reach a usable model |
| **Run C** | not recorded in reviewed logs | ≥2026-07-08 | Single ConvNeXt-tiny, fast mode, 75 epochs (53.7 min) | Full metrics suite + working dashboard deployment; detector left at COCO-bootstrap-only |

Runs B and C are reported in the source logs as two separate sessions ("earlier run" and "later/complete run" respectively) reviewed together. Both post-date 2026-07-08, since fast-training mode (`enable_fast_train`, 75-epoch schedule) was only added to the codebase that day — see `MEMORY.md`'s "Fast-training mode added" entry.

---

## 3. Run A — 2026-07-04 full training (`real-time-ex-03`, Tesla T4)

Single ConvNeXt-tiny, 300 epochs. The notebook's ensemble auto-skip logic (`TRAIN_ENSEMBLE = best_val_acc < 0.92`) never triggered, since this single model already cleared the 92% target.

| Metric | Target | Result | Verdict |
|---|---|---|---|
| Accuracy (test) | ≥92% | **93.12%** | PASS |
| ECE, raw | ≤0.05 | 0.1226 | FAIL |
| ECE, temperature-scaled (T=0.760) | ≤0.05 | **0.0395** | PASS |
| Faithfulness AUC — deletion | ≥0.80 | 0.4908 | FAIL |
| Faithfulness AUC — insertion | ≥0.80 | 0.7489 | FAIL |
| End-to-end latency | ≤40ms | 111.9ms (cold-path — includes one-time Grad-CAM hook/module-scan construction cost, later flagged as non-representative) | FAIL |
| mAP@0.5 | ≥75% | not evaluated — 0 detections (bootstrapped detector untrained on IR) | — |

**Per-domain test accuracy:** NAVAL 99.75%, AIR 95.73%, **GROUND 85.23%** (weak point).

**Data composition:** 81.4% synthetic / 18.6% real. Only **8 of 43 classes** had any real-annotation pool; 7 datasets mapped 0% of raw annotations (HRSC2016, Ships_Aerial, Ships_Vessels_Aerial, SWIM, SwimmingPool_Car, Vehicle_Dataset, Aerial_Segmentation) — this finding directly drove the ingestion-parser fixes landed 2026-07-06 (Roboflow stem-lookup bug, rotated-box XML support, per-dataset `data.yaml` class-name resolution — see `MEMORY.md`).

**Confusion clusters** (from the 43×43 confusion matrix): (1) armored ground vehicles — BMP2↔Bradley, K21↔BMP2, general T72/T90/Leopard2 bleed, the main GROUND-domain drag; (2) fighter jets — Su27↔F16, MiG21↔F16, matching the paper's own reported confusion. Both groups were added to `hard_negative_mining.CONFUSABLE_GROUPS` as a direct result of this run (alongside the paper's original `{F16, MiG19, MiG21}` group).

---

## 4. Runs B & C — most recent session (fast-training mode, ≥2026-07-08)

### 4.1 Environment & data

Kaggle Tesla T4 (15.6 GB VRAM), Python 3.12.13, PyTorch 2.10. Of 21 configured dataset keys, 18 resolved to a mounted path. Reported as contributing zero usable annotations: HRSC2016, SARScope_Maritime, Aerial_Segmentation/Aerial_Roof_Seg, among others.

> **Consistency flag:** `MEMORY.md`'s 2026-07-06 "Ingestion parser fixes" entry records `SARScope_Maritime` as fixed and mapped ("ship→naval size-rule... Added, working") as of that date. If this run genuinely still saw it UNMAPPED, the most likely explanation — by direct analogy to the pre-fix-clone issue `MEMORY.md` documents for a *different* bug in its most recent (2026-07-09) entry — is that this session's `c-clone` pulled a commit that predated the merge of that fix into whatever branch it tracked, rather than the fix itself regressing. Worth confirming which commit this run actually ran against before citing the "SARScope still unmapped" finding as current state in a live presentation.

Only **10 of 43 classes** have real-image backing (up from 8 in Run A); the other 33 are procedurally-generated synthetic targets. Final balanced split: **7,310 train / 1,290 val / 8,600 test** (170/30/200 per class, per the standard REATS split).

### 4.2 Training

Single ConvNeXt-tiny, **fast-training mode** (75 epochs instead of 300, lightweight augmentation, `ema_decay=0.999`) reached **val accuracy 0.9636** in 53.7 minutes. As in Run A, the ensemble auto-skip never triggered — only one checkpoint exists on disk.

This result is notably above `FAST_TRAINING_GUIDE.md`'s own documented expectation for fast mode (~90–91% accuracy, ~1–2 point drop from full training). The most likely explanation is the same data-composition effect flagged throughout this report: the guide's expectation appears calibrated against a more architecture-validation-oriented view, while 33 of 43 classes here are synthetic-only and synthetic-only accuracy runs ~98% (§4.5) — a heavy skew toward "easy" classes pulls the headline number up regardless of training schedule length.

### 4.3 Results vs. paper targets

| Metric | Target | Result | Verdict |
|---|---|---|---|
| Accuracy (test) | ≥92% | **0.9550** | PASS* (*see §4.5 — real-backed-only accuracy is ~87%) |
| ECE, raw | ≤0.05 | 0.0742 | FAIL |
| ECE, temperature-scaled (T=0.838) | ≤0.05 | **0.0175** | PASS |
| Faithfulness AUC — deletion | ≥0.80 | 0.245 | FAIL |
| Faithfulness AUC — insertion | ≥0.80 | 0.751 | FAIL |
| End-to-end latency | ≤40ms | 81.7ms | FAIL (~2×) |
| mAP@0.5 (Run B, detector) | ≥75% | 0.0003 → 0.0011 over 20 epochs | FAIL — effectively untrained |
| Macro FAR / RED-threat FAR | none (reported) | 0.044 / 0.043 | reported |
| Macro MR / RED-threat MR | none (reported) | 0.045 / 0.049 | reported |

FAR = FP/(FP+TP) = 1−Precision; MR = FN/(FN+TP) = 1−Recall — a missed RED-threat class is the costliest failure mode (`modules/threat_metrics.py`).

### 4.4 Per-domain accuracy

**AIR 97.8%, GROUND 95.3%, NAVAL 86.7%** — naval is now the weak domain, a reversal from Run A where NAVAL was strongest (99.75%) and GROUND was weakest (85.23%). See §5 for a candidate explanation — this is *not* attributable to the hard-negative-mining fine-tune pass, which `MEMORY.md`'s pending list still lists as never having been run against a real checkpoint.

**Worst per-class offenders:** Ka52 (FAR 0.38), MiG21 (MR 0.285), F35 (MR 0.27).

### 4.5 The provenance caveat — real vs. synthetic accuracy

This is the single most important qualifier on the headline number:

| Bucket | Accuracy |
|---|---|
| Real-backed classes | **87.0%** (1,741 / 2,000) |
| Synthetic-only classes | **98.1%** |
| **Headline (blended)** | **95.50%** |

33 of 43 classes are procedurally-generated synthetic targets with no real IR pixels behind them — the source session's own conclusion was that the blended 95.50% figure is "dominated by the 33 synthetic-only classes... treat as architecture validation until label-map coverage grows." **True field-relevant accuracy is closer to 87%, not 95.5%.** This caveat should travel with the headline number anywhere it's quoted in a presentation.

### 4.6 Module A (detector) — not yet functional

Run B (the detector-training attempt) ran 20 fast-mode epochs; mAP@0.5 crawled from 0.0003 to 0.0011 against a ≥75% target — effectively untrained. A generated demo image showed the detector firing ~2,687 spurious boxes on a trivial 2-object test frame. Run C (the complete run) did not attempt detector training at all — it only bootstrapped COCO darknet YOLOv4 weights and smoke-tested (0 detections), which is expected behavior for an untrained detector head, not a bug (`bootstrap_detector_weights.py` intentionally strips detection heads on the class-count mismatch between COCO's 80 classes and REATS's 43).

A separate demo image reportedly showed a clean 112ms full A→B→C pipeline pass (correctly classifying "M109" at 0.63 confidence) — this should be read as a cherry-picked illustrative case, not a representative detector accuracy figure, given the mAP result above.

**Run B's relationship to the separately-logged 2026-07-09 bug-fix session (`real-time-ex-05`) is unclear and shouldn't be assumed.** `MEMORY.md` independently records a `real-time-ex-05` session that hit two real crashes in this same detection-training path — a stale-clone `ImageFolder` crash, then a genuine `MosaicDataset._mosaic()` negative-slice bug that crashed `c-train-detector` mid-DataLoader on ~50% of mosaic draws — both now fixed on `main`. That session's detector training *crashed*, so it cannot be the same run as Run B, which completed all 20 epochs and produced a (bad) mAP curve and a demo image. Run B could be an earlier attempt that predates the mosaic bug being hit, or a later one on the now-fixed code that simply produced a mundane near-zero result (expected from 20 fast-mode epochs on scarce real bbox data, independent of that bug) — the source logs reviewed didn't establish which. Either way, **a fresh detector-training run on the current, fixed `main` has not been confirmed to produce a working detector**, so that remains the natural next step.

### 4.7 Dashboard (Module D) — working

The session's Streamlit log confirms Module D launched successfully with a public tunnel URL. Live batch-classification output from the dashboard's Batch Processing tab spot-checks as sensible — correct top-1 classes with plausible confusable runners-up.

---

## 5. Cross-run comparison

| Metric | Target | Run A (2026-07-04, full 300ep) | Run C (≥2026-07-08, fast 75ep) | Trend |
|---|---:|---:|---:|---|
| Test accuracy | ≥92% | 93.12% | 95.50% (87.0% real-backed) | ~flat once provenance-adjusted |
| ECE, raw | ≤0.05 | 0.1226 | 0.0742 | improved |
| ECE, temp-scaled | ≤0.05 | 0.0395 (T=0.760) | 0.0175 (T=0.838) | improved |
| Faithfulness AUC (deletion) | ≥0.80 | 0.4908 | 0.245 | **worse** |
| Faithfulness AUC (insertion) | ≥0.80 | 0.7489 | 0.751 | ~flat |
| End-to-end latency | ≤40ms | 111.9ms | 81.7ms | improved, but see note below |
| Weakest domain | — | GROUND 85.23% | NAVAL 86.7% | **flipped** |
| Real-image-backed classes | 43 | 8 | 10 | improved |

**Two trends worth flagging rather than taking at face value:**

- **Latency improvement is partly a measurement fix, not a pure speed gain.** `MEMORY.md`'s 2026-07-05 entry notes Run A's 111.9ms figure included one-time Grad-CAM hook/module-scan construction cost, and the benchmark cell was subsequently fixed to run an untimed warm-up pass before averaging 5 timed reps. Run C's 81.7ms is measured under the corrected methodology, so only part of the 30ms gap is attributable to actual pipeline speedups (e.g., the batched Grad-CAM and device-aware chunking work logged elsewhere in `MEMORY.md`).
- **The GROUND/NAVAL weak-domain flip is not yet explained by a verified cause.** The most consistent explanation available from already-documented facts: Run A's near-perfect 99.75% NAVAL accuracy was measured on an almost entirely-synthetic NAVAL test set (synthetic classes score ~98% regardless of domain), while the naval-focused ingestion fixes landed 2026-07-06 (SARScope, Thermal_Ships, Ships_Vessels_Aerial, SWIM, Ships_Satellite) subsequently introduced harder, more visually-ambiguous *real* naval imagery (the Corvette/Frigate/Destroyer confusion both runs' analyses call out) into Run C's test set — pulling the domain average down even though absolute real-data coverage improved. This is a plausible reading consistent with the provenance-split data in §4.5, not a confirmed causal finding — it would need a per-run provenance breakdown by domain to verify.

---

## 6. What's still open

Straight from `MEMORY.md`'s pending list and `docs/gap_analysis_report.md`'s "what remains as future GPU work," cross-checked against both runs above — none of the following changed as a result of either run:

1. **Train the 6-architecture heterogeneous ensemble to convergence.** Code is complete and unit-tested (`ARCHITECTURES`, `build_model`, `train_ensemble`); no run to date has exercised it because a single ConvNeXt-tiny has cleared 92% every time, tripping the auto-skip.
2. **Get a real detector-training run past the two bugs `MEMORY.md`'s 2026-07-09 entry just fixed.** Neither Run B nor Run C reflects that fixed code.
3. **Run the hard-negative fine-tuning pass against a real trained checkpoint.** `CONFUSABLE_GROUPS` (F16/MiG19/MiG21, BMP2/Bradley/K21, T72/T90/Leopard2) is implemented and unit-tested against synthetic data only — its real-world effect on the confusion clusters both runs observed is still unmeasured.
4. **Tune the 0.50/0.75/0.90 Warning/Track/Engagement confidence thresholds** against real measured FAR/MR data — currently a reasonable starting point, not empirically fit.
5. **Close remaining label-map gaps** — `Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV` (the last confirmed to be a non-dataset substitute, visual-reference only) still need real dataset sources or confirmed mappings before real-class coverage grows past 10/43.
6. **Investigate the faithfulness AUC failure** — both runs fail deletion AUC badly (0.49 and 0.25, target ≥0.80), and Run C is worse than Run A despite better calibration. Likely tied to the same synthetic-data-dominance problem as the accuracy caveat, but unconfirmed.

---

## 7. Bottom line for the presentation

REATS's classification stage (Module B) is architecturally sound and clears the paper's accuracy/calibration targets — **but only once inflation from 33 synthetic-only classes is accounted for**; real-world accuracy is closer to 87% than the 95.5% headline. Detection (Module A) has not yet produced a working model in any run to date. Explainability faithfulness misses its target in both runs and is trending worse, not better. Latency misses its target by roughly 2× even under corrected measurement. The dashboard and full pipeline are functional end-to-end, and the heterogeneous 6-model ensemble, FAR/MR reporting, hard-negative-mining groups, and Warning/Track/Engagement policy are all implemented and unit-tested but not yet exercised on a real GPU run at full scale.

This reads as steady infrastructure progress between the two dated runs (ingestion coverage 8→10 real classes, ECE improving, dashboard going from untested to working end-to-end) without yet a corresponding jump in the metrics that matter most for a deployable system: real-world classification accuracy, detection, and explainability faithfulness. The highest-leverage next step is a clean run of the just-fixed Module A training path, since detection is currently the only module with no working result at all.

---

## Appendix — source provenance

- **Run A (2026-07-04) figures:** `MEMORY.md`, "Kaggle run results" section (committed to this repo); original standalone Kaggle report was never committed.
- **Runs B/C figures:** transcribed from that Kaggle session's own stdout logs (two files, ~18k and ~14k lines) and generated artifacts (confusion matrix, class distribution, Grad-CAM-by-domain, Module A demo, pipeline demo, and sample-grid images; an MLflow run directory; a Streamlit log confirming dashboard launch) as reviewed in a separate session. None of these source files exist in this repository (checkpoints and run artifacts are gitignored by design — see `CLAUDE.md`'s "Git / data layout" section) — this report's authors did not have direct access to re-verify them and instead cross-checked every figure for consistency against the committed codebase and `MEMORY.md`'s independent run history.
- **Architecture, taxonomy, and target definitions:** `README.md`, `CLAUDE.md`, `REATS/config/targets.yaml`, `docs/gap_analysis_report.md`.
