# REATS Kaggle GPU Run Report

**Compiled:** 2026-07-09
**Reported result — use this in the presentation:** **93.12% test accuracy**, 2026-07-04 full-training run (`real-time-ex-03`), single ConvNeXt-tiny, 300 epochs.
**Reference benchmark only — do not quote as system performance:** a later fast-training-mode run (75 epochs) is kept in this report solely to illustrate the training-time/quality tradeoff. Its 95.50% headline accuracy is inflated by data composition, not a genuine quality improvement, and is excluded from reporting — see §3.

> **Provenance note:** the 2026-07-04 figures are reconstructed from `MEMORY.md`'s "Kaggle run results" entry (the original standalone report for that run was never committed to this repo). The fast-training-mode benchmark's tabular figures are transcribed from that session's own notebook stdout logs (two files, ~18k and ~14k lines) as reviewed in a separate session; its five generated images (class distribution, confusion matrix, Grad-CAM by domain, Module A demo, A→B→C pipeline demo) were reviewed directly for this revision and are described from that direct review in §4.4. Every figure is cross-checked for internal and architectural consistency against the current codebase (`MEMORY.md`, `docs/gap_analysis_report.md`, `FAST_TRAINING_GUIDE.md`, `REATS/config/targets.yaml`); conflicts are called out, not silently resolved.

---

## Executive summary

- **The reported system result is 93.12% test accuracy**, from a full 300-epoch training run on 2026-07-04 (`real-time-ex-03`) — clears the paper's ≥92% target under the training regime the paper itself specifies. **This is the number for the presentation.**
- **A separate fast-training-mode run (75 epochs) measured a higher headline number (95.50%) and is deliberately excluded from reporting.** Fast mode trades training quality for time — `FAST_TRAINING_GUIDE.md` documents an *expected* ~1–2 point **drop** from full training, not a gain. A result that went up instead of down under a regime built to be lower-fidelity is a sign of data-composition inflation, not better modeling. Full reasoning in §3.
- **Both runs' headline numbers depend heavily on synthetic data.** Only 8–10 of 43 classes have any real-image backing; in the benchmark run, real-backed classes scored 87.0% vs. 98.1% for synthetic-only classes. The same caveat applies to the reported 93.12% figure, which comes from an even more synthetic-heavy split (only 8 real-backed classes).
- **Calibration (ECE) only passes after temperature scaling** in the reported run (raw 0.1226, temperature-scaled 0.0395, target ≤0.05) — expected, and already handled by `TemperatureScaler` in the pipeline.
- **Explainability faithfulness fails its ≥0.80 target** in the reported run (deletion AUC 0.49, insertion AUC 0.75) — the weakest area of the system relative to target.
- **Detection (Module A) is not functional.** The one training attempt made to date (in the benchmark session) reached mAP@0.5 of 0.0011 against a ≥75% target and produces thousands of spurious boxes on trivial input — visually confirmed in §4.4.
- **End-to-end latency fails its 40ms target** in the reported run (111.9ms, ~2.8× target).
- The 6-architecture heterogeneous ensemble has **never actually been trained** in any run — a single ConvNeXt-tiny cleared 92% both times, so the notebook's auto-skip logic never triggered it.

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

## 2. Reported result — 2026-07-04 full training (`real-time-ex-03`, Tesla T4)

**This is the figure to use in the presentation.** Single ConvNeXt-tiny, 300 epochs (~6h) — the training schedule the paper itself specifies. The notebook's ensemble auto-skip logic (`TRAIN_ENSEMBLE = best_val_acc < 0.92`) never triggered, since this single model already cleared the 92% target.

| Metric | Target | Result | Verdict |
|---|---|---|---|
| Accuracy (test) | ≥92% | **93.12%** | PASS |
| ECE, raw | ≤0.05 | 0.1226 | FAIL |
| ECE, temperature-scaled (T=0.760) | ≤0.05 | **0.0395** | PASS |
| Faithfulness AUC — deletion | ≥0.80 | 0.4908 | FAIL |
| Faithfulness AUC — insertion | ≥0.80 | 0.7489 | FAIL |
| End-to-end latency | ≤40ms | 111.9ms (~2.8× target; cold-path figure — includes one-time Grad-CAM hook/module-scan construction cost, later flagged as non-representative and fixed in the benchmarking cell, see §5) | FAIL |
| mAP@0.5 | ≥75% | not evaluated — 0 detections (bootstrapped detector untrained on IR) | — |

**Per-domain test accuracy:** NAVAL 99.75%, AIR 95.73%, **GROUND 85.23%** (weak point).

**Data composition:** 81.4% synthetic / 18.6% real. Only **8 of 43 classes** had any real-annotation pool; 7 datasets mapped 0% of raw annotations (HRSC2016, Ships_Aerial, Ships_Vessels_Aerial, SWIM, SwimmingPool_Car, Vehicle_Dataset, Aerial_Segmentation) — this finding directly drove the ingestion-parser fixes landed 2026-07-06 (Roboflow stem-lookup bug, rotated-box XML support, per-dataset `data.yaml` class-name resolution — see `MEMORY.md`).

**Confusion clusters** (from the 43×43 confusion matrix): (1) armored ground vehicles — BMP2↔Bradley, K21↔BMP2, general T72/T90/Leopard2 bleed, the main GROUND-domain drag; (2) fighter jets — Su27↔F16, MiG21↔F16, matching the paper's own reported confusion. Both groups were added to `hard_negative_mining.CONFUSABLE_GROUPS` as a direct result of this run (alongside the paper's original `{F16, MiG19, MiG21}` group).

**Even this reported figure carries the same synthetic-data caveat as everything else in this report** — only 8 of 43 classes have real-image backing, so 93.12% should be read as validated architecture performance on the current data mix, not a claim about real-world field accuracy. It is nonetheless the correct figure to report, because it was produced under the paper's own specified training regime rather than a speed-optimized shortcut.

---

## 3. Why the fast-training-mode run is not used as the reported result

A second real GPU session, using fast-training mode (75 epochs instead of 300, lightweight augmentation — added to the codebase 2026-07-08), measured a *higher* headline accuracy (95.50% tabular / 0.953 as labeled on its own confusion-matrix image — see the discrepancy note in §4.3) than the full 300-epoch run above. That is a reason for suspicion, not celebration:

- `FAST_TRAINING_GUIDE.md` documents fast mode's own expected trade: **~90–91% accuracy, a 1–2 point drop** from full training's ~92–93%, in exchange for roughly 4× speed. A result that went *up* instead of down under a regime explicitly built to be lower-fidelity indicates the two runs aren't measuring comparably, not that fast mode is secretly better.
- The most likely driver is data composition, not modeling quality: the benchmark run's own provenance breakdown (§4.5) shows synthetic-only classes scoring 98.1% vs. 87.0% for real-backed classes. With roughly three-quarters of all classes still synthetic-only, small run-to-run shifts in exactly which classes carry real data can swing the blended headline number by several points independent of anything the model actually learned.
- Its faithfulness AUC is *worse* than the full-training run's (deletion 0.245 vs. 0.4908), and its one detector-training attempt (mAP 0.0011) is no better. Neither result supports fast mode producing a higher-quality system.

**Per this decision, the benchmark run's 95.50% figure should not be quoted as REATS's accuracy in the presentation.** It is retained below as engineering evidence — it demonstrates the dashboard deploying successfully end-to-end, and gives directional signal on calibration behavior and confusion patterns (now visually confirmed, §4.4) — not as a competing headline result.

---

## 4. Reference benchmark — fast-training-mode run (not for reporting)

### 4.1 Environment & data

Kaggle Tesla T4 (15.6 GB VRAM), Python 3.12.13, PyTorch 2.10. Of 21 configured dataset keys, 18 resolved to a mounted path. Reported as contributing zero usable annotations: HRSC2016, SARScope_Maritime, Aerial_Segmentation/Aerial_Roof_Seg, among others.

> **Consistency flag:** `MEMORY.md`'s 2026-07-06 "Ingestion parser fixes" entry records `SARScope_Maritime` as fixed and mapped ("ship→naval size-rule... Added, working") as of that date. If this run genuinely still saw it UNMAPPED, the most likely explanation — by direct analogy to the pre-fix-clone issue `MEMORY.md` documents for a *different* bug in its most recent (2026-07-09) entry — is that this session's `c-clone` pulled a commit that predated the merge of that fix, rather than the fix itself regressing. Worth confirming which commit this run actually ran against before citing "SARScope still unmapped" as current state.

Only **10 of 43 classes** have real-image backing (up from 8 in the reported run); the other 33 are procedurally-generated synthetic targets — directly visible in the class-distribution image reviewed in §4.4, where every one of the 43 classes hits the identical 170/30/200 split regardless of whether any real annotation exists for it. Final balanced split: **7,310 train / 1,290 val / 8,600 test**.

### 4.2 Training

Single ConvNeXt-tiny, fast-training mode (75 epochs, lightweight augmentation, `ema_decay=0.999`) reached val accuracy 0.9636 in 53.7 minutes. As in the reported run, the ensemble auto-skip never triggered — only one checkpoint exists on disk. See §3 for why this run's accuracy figure is not used as the system's reported result.

### 4.3 Benchmark metrics (not for reporting)

| Metric | Target | Result | Verdict |
|---|---|---|---|
| Accuracy (test) | ≥92% | 0.9550 (tabular log) / **0.953** (labeled on its own confusion-matrix image, §4.4 — a genuine ~0.2-point discrepancy between the two, not just display rounding; unexplained in the source logs, flagged rather than resolved) | Not reported — see §3 |
| ECE, raw | ≤0.05 | 0.0742 | FAIL |
| ECE, temperature-scaled (T=0.838) | ≤0.05 | 0.0175 | PASS |
| Faithfulness AUC — deletion | ≥0.80 | 0.245 | FAIL |
| Faithfulness AUC — insertion | ≥0.80 | 0.751 | FAIL |
| End-to-end latency | ≤40ms | 81.7ms (~2× target) | FAIL |
| mAP@0.5 (separate 20-epoch detector attempt) | ≥75% | 0.0003 → 0.0011 | FAIL — effectively untrained |
| Macro FAR / RED-threat FAR | none (reported) | 0.044 / 0.043 | reported |
| Macro MR / RED-threat MR | none (reported) | 0.045 / 0.049 | reported |

FAR = FP/(FP+TP) = 1−Precision; MR = FN/(FN+TP) = 1−Recall — a missed RED-threat class is the costliest failure mode (`modules/threat_metrics.py`).

**Per-domain accuracy:** AIR 97.8%, GROUND 95.3%, **NAVAL 86.7%** — a reversal from the reported run, where NAVAL was strongest (99.75%) and GROUND was weakest (85.23%). See §5 for a candidate explanation. **Worst per-class offenders:** Ka52 (FAR 0.38), MiG21 (MR 0.285), F35 (MR 0.27).

### 4.4 Visual evidence reviewed

Five images generated by this run were reviewed directly for this revision (previously relayed only as text description):

**Class distribution** (`class_distribution.png`) — confirms the balanced 170/30/200-per-class split across all 43 classes, hit uniformly regardless of real-data availability: classes with zero real annotations were backfilled with synthetic data to the exact same target count as classes with real coverage. This is the visual mechanism behind the provenance-split gap in §4.5 — the split chart alone can't distinguish a real-backed class from a synthetic-only one, which is exactly the problem.

**Confusion matrix** (`confusion_matrix.png`, labeled acc=0.953). The single most visually prominent off-diagonal block is a tight three-way cluster among **Corvette, Destroyer, and Frigate** — precisely the naval domain that dropped from 99.75% (reported run) to 86.7% (this run), corroborating that finding by direct observation rather than secondhand description. Lighter, more scattered confusion is visible around the armored-vehicle group (BMP2↔Bradley, T90↔Leopard2) and the fighter-jet group (F16/MiG21/Su27), consistent with the three groups already defined in `hard_negative_mining.CONFUSABLE_GROUPS`.

**Grad-CAM by domain** (`gradcam_by_domain.png`) — concrete visual evidence for this report's central caveat. Three AH64 examples and three BM21 examples — simple, flat, iconographic synthetic renders (a white cross and a white rod on plain noisy backgrounds) — are all classified correctly with tight, sensibly-centered attention. Three Corvette examples — textured, silhouette-style renders, visibly more complex than the AIR/GROUND icons — split 1 correct / **2 misclassified as Frigate**, with attention landing on specific hull/silhouette features rather than a clean holistic match on the misclassified pair. This is direct evidence that the easiest, most iconographic synthetic classes drive the headline number up while the harder, more realistic-looking naval renders carry the true difficulty.

**Module A demo** (`module_a_demo.png`) — a synthetic test frame containing exactly 2 objects (two soft-edged glowing blobs) returns **2,687 detections** that visually cover essentially the entire frame edge-to-edge in dense, overlapping boxes, with no discernible concentration around the actual 2 objects. This is not "low precision" in the ordinary sense — the output is uncorrelated with scene content, consistent with an untrained detection head (COCO backbone/neck weights only, per `bootstrap_detector_weights.py`'s documented behavior of stripping heads on the 80→43-class mismatch) and with the near-zero mAP@0.5 (0.0011) reported alongside it.

**A→B→C pipeline demo** (`pipeline_demo.png`) — the "112ms, M109 @ 0.63 confidence" example is run on the same style of synthetic 2-blob test scene as the Module A demo, not real or even target-shaped IR imagery. It demonstrates the pipeline's plumbing works end-to-end (detector→crop→classifier→Grad-CAM→top-3 probabilities all execute and return a plausible-looking result), but given what the confusion matrix and Module A demo above show, it should be read as a plumbing check, not evidence of real-world detector or classifier accuracy.

### 4.5 The provenance caveat — real vs. synthetic accuracy

| Bucket | Accuracy |
|---|---|
| Real-backed classes | **87.0%** (1,741 / 2,000) |
| Synthetic-only classes | **98.1%** |
| Headline (blended) | 95.50% |

33 of 43 classes are procedurally-generated synthetic targets with no real IR pixels behind them. This is the data-composition effect §3 points to as the likely explanation for this run's inflated headline number, and it applies — to a lesser degree, since the split is 8/43 real rather than 10/43 — to the reported 93.12% figure as well.

### 4.6 Module A (detector) — not yet functional

The detector-training attempt in this session ran 20 fast-mode epochs; mAP@0.5 crawled from 0.0003 to 0.0011 against a ≥75% target — effectively untrained, and visually confirmed non-functional in §4.4. This session did not otherwise attempt detector training — it separately bootstrapped COCO darknet YOLOv4 weights and smoke-tested (0 detections), which is expected behavior for an untrained detector head, not a bug.

This detector-training attempt's relationship to the separately-logged 2026-07-09 bug-fix session (`real-time-ex-05`) is unclear and shouldn't be assumed. `MEMORY.md` independently records a `real-time-ex-05` session that hit two real crashes in this same detection-training path — a stale-clone `ImageFolder` crash, then a genuine `MosaicDataset._mosaic()` negative-slice bug that crashed `c-train-detector` mid-DataLoader on ~50% of mosaic draws — both now fixed on `main`. That session's detector training *crashed*, so it cannot be the same run as the one described here, which completed all 20 epochs. Either this run predates the mosaic bug being hit, or postdates the fix and simply produced a mundane near-zero result (expected from 20 fast-mode epochs on scarce real bbox data, independent of that bug) — unclear which. Either way, **a fresh detector-training run on the current, fixed `main` has not been confirmed to produce a working detector.**

### 4.7 Dashboard (Module D) — working

This session's Streamlit log confirms Module D launched successfully with a public tunnel URL. Live batch-classification output from the dashboard's Batch Processing tab spot-checks as sensible — correct top-1 classes with plausible confusable runners-up.

---

## 5. Cross-run comparison

Provided for engineering-trend context only — the benchmark run's figures are not used as reported results (§3).

| Metric | Target | Reported (2026-07-04, full 300ep) | Benchmark (≥2026-07-08, fast 75ep) | Trend |
|---|---:|---:|---:|---|
| Test accuracy | ≥92% | **93.12%** | 95.50% / 0.953 (not reported, §3) | benchmark inflated, not comparable |
| ECE, raw | ≤0.05 | 0.1226 | 0.0742 | improved |
| ECE, temp-scaled | ≤0.05 | 0.0395 (T=0.760) | 0.0175 (T=0.838) | improved |
| Faithfulness AUC (deletion) | ≥0.80 | 0.4908 | 0.245 | **worse** |
| Faithfulness AUC (insertion) | ≥0.80 | 0.7489 | 0.751 | ~flat |
| End-to-end latency | ≤40ms | 111.9ms | 81.7ms | improved, but see note below |
| Weakest domain | — | GROUND 85.23% | NAVAL 86.7% | **flipped** |
| Real-image-backed classes | 43 | 8 | 10 | improved |

**Two trends worth flagging rather than taking at face value:**

- **Latency improvement is partly a measurement fix, not a pure speed gain.** `MEMORY.md`'s 2026-07-05 entry notes the reported run's 111.9ms figure included one-time Grad-CAM hook/module-scan construction cost, and the benchmark cell was subsequently fixed to run an untimed warm-up pass before averaging 5 timed reps. The benchmark run's 81.7ms is measured under the corrected methodology, so only part of the 30ms gap is attributable to actual pipeline speedups (e.g., the batched Grad-CAM and device-aware chunking work logged elsewhere in `MEMORY.md`).
- **The GROUND/NAVAL weak-domain flip is not yet explained by a verified cause.** The most consistent explanation available from already-documented facts, now with direct visual support (§4.4's confusion-matrix review): the reported run's near-perfect 99.75% NAVAL accuracy was measured on an almost entirely-synthetic NAVAL test set (synthetic classes score ~98% regardless of domain), while the naval-focused ingestion fixes landed 2026-07-06 (SARScope, Thermal_Ships, Ships_Vessels_Aerial, SWIM, Ships_Satellite) subsequently introduced harder, more visually-ambiguous *real* naval imagery — the Corvette/Frigate/Destroyer confusion the confusion-matrix image shows directly — into the benchmark run's test set, pulling the domain average down even though absolute real-data coverage improved. Plausible and now visually corroborated, but still not a confirmed causal finding — it would need a per-run provenance breakdown by domain to fully verify.

---

## 6. What's still open

Straight from `MEMORY.md`'s pending list and `docs/gap_analysis_report.md`'s "what remains as future GPU work," cross-checked against both runs above — none of the following changed as a result of either run:

1. **Train the 6-architecture heterogeneous ensemble to convergence.** Code is complete and unit-tested (`ARCHITECTURES`, `build_model`, `train_ensemble`); no run to date has exercised it because a single ConvNeXt-tiny has cleared 92% every time, tripping the auto-skip.
2. **Get a real detector-training run past the two bugs `MEMORY.md`'s 2026-07-09 entry just fixed.** Neither run described in this report reflects that fixed code with confidence.
3. **Run the hard-negative fine-tuning pass against a real trained checkpoint.** `CONFUSABLE_GROUPS` (F16/MiG19/MiG21, BMP2/Bradley/K21, T72/T90/Leopard2) is implemented and unit-tested against synthetic data only — its real-world effect on the confusion clusters both runs observed (and the benchmark run's confusion matrix now visually confirms) is still unmeasured.
4. **Tune the 0.50/0.75/0.90 Warning/Track/Engagement confidence thresholds** against real measured FAR/MR data — currently a reasonable starting point, not empirically fit.
5. **Close remaining label-map gaps** — `Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV` (the last confirmed to be a non-dataset substitute, visual-reference only) still need real dataset sources or confirmed mappings before real-class coverage grows past 10/43.
6. **Investigate the faithfulness AUC failure** — both runs fail deletion AUC badly (0.49 and 0.25, target ≥0.80), and the benchmark run is worse despite better calibration. Likely tied to the same synthetic-data-dominance problem as the accuracy caveat, but unconfirmed.

---

## 7. Bottom line for the presentation

**Report 93.12% test accuracy** (2026-07-04, full 300-epoch training) as REATS's classification result — it clears the paper's 92% target under the training regime the paper itself specifies, without the additional speed/quality tradeoff fast-training mode introduces. Even this reported figure carries the same caveat as every number in this report: only 8 of 43 classes currently have real-image backing, so treat 93.12% as validated architecture performance on the current data mix, not a claim about real-world field accuracy.

Detection (Module A) has not produced a working model in any run to date — visually confirmed non-functional in §4.4 (2,687 boxes on a 2-object frame; mAP@0.5 of 0.0011). Explainability faithfulness misses its target (deletion AUC 0.49 vs. ≥0.80). Latency misses its target by roughly 2.8×. The dashboard and full pipeline run end-to-end. The heterogeneous 6-model ensemble, FAR/MR reporting, hard-negative-mining groups, and Warning/Track/Engagement policy are all implemented and unit-tested but not yet exercised on a real GPU run at full scale. The highest-leverage next step is a clean run of the just-fixed Module A training path, since detection is the only module with no working result in either run reviewed here.

---

## Appendix — source provenance

- **Reported result (2026-07-04) figures:** `MEMORY.md`, "Kaggle run results" section (committed to this repo); the original standalone Kaggle report was never committed.
- **Benchmark run figures:** transcribed from that Kaggle session's own stdout logs (two files, ~18k and ~14k lines), as reviewed in a separate session. Its five generated images — class distribution, confusion matrix, Grad-CAM by domain, Module A demo, A→B→C pipeline demo — were reviewed directly for this revision (§4.4); a sixth generated image (sample grid) was described only in the earlier session's text summary and has not been directly reviewed. None of these source files exist in this repository (checkpoints and run artifacts are gitignored by design — see `CLAUDE.md`'s "Git / data layout" section).
- **Architecture, taxonomy, and target definitions:** `README.md`, `CLAUDE.md`, `REATS/config/targets.yaml`, `docs/gap_analysis_report.md`, `FAST_TRAINING_GUIDE.md`.
