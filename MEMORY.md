# MEMORY.md — REATS Session State

Last updated: 2026-07-05
Active branch: `claude/model-gaps-ensemble-metrics-e4xysn`

---

## What this file is for

Running log of architectural decisions, bug fixes, and session context so that future Claude Code sessions can resume without re-deriving the same information.

---

## Current project state

`notebooks/01_kaggle_full_pipeline.ipynb` is the primary execution environment — **it now targets Google Colab, not Kaggle** (ported 2026-07-05; filename kept as-is, see "Colab port" below). All REATS modules (A–E) are implemented and the dashboard is functional. **A real GPU training run completed on Kaggle 2026-07-04** (before the Colab port) — see "Kaggle run results" below; it's the ground truth for what still needs work, superseding guesses made from synthetic-data smoke tests alone. That run has not yet been repeated on Colab.

### What is working
- Module A: `IRDetector` YOLOv4 pure-PyTorch, forward pass + NMS (bootstrapped from COCO darknet weights; heads untrained on IR — 0 detections on real IR input by design until fine-tuned)
- Module B: heterogeneous 6-architecture ensemble (ConvNeXt_tiny/ResNeXt50/ViT_b_16/Swin_T/VGG16/ResNet18), AMP + EMA, TemperatureScaler calibration, hard-negative mining (3 confusable groups)
- Module C: GradCAM / GradCAM++ / EigenCAM, SHAP, LIME, MCDropout, faithfulness AUC — Grad-CAM batched per-chunk in Module D
- Module D: Streamlit 5-tab dashboard (Live Analysis, Batch, Calibration, About, iPhone Live) — FAR/MR + Warning/Track/Engagement policy wired in; ROI classification batched device-aware (see below)
- Module E: FastAPI/WebSocket phone-camera streamer
- Ingestion pipeline: 20 dataset keys as of 2026-07-05 (was 16); wrapper-directory descent fixed 2026-07-04
- Notebook: runs on Colab, datasets pulled via `kagglehub` (see below)

### What is pending
- Fix the 7 zero-mapped-label datasets flagged by the 2026-07-04 run (partially addressed; HRSC2016 may need dataset-content verification, not just a code fix)
- **New**: add `ingestion/label_maps.yaml` entries for the 5 datasets added 2026-07-05 (`Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV`) — none have a mapping yet; inspect via the ingestion pipeline's own UNMAPPED report first, don't guess
- Fine-tune Module A on labeled IR detection data (mAP@0.5 currently unmeasured — bootstrapped detector fires on COCO classes, not IR blobs)
- Investigate faithfulness AUC failure (0.49 deletion, target ≥0.80) — likely tied to the 81%-synthetic corpus, needs real-data share to grow before re-testing
- Re-run hard-negative fine-tune (`hard_negative_mining.py`) against a real trained checkpoint to confirm it reduces the fighter-jet and armored-vehicle confusion the Kaggle run's confusion matrix showed
- **New**: re-run the full pipeline on Colab (datasets via kagglehub, including the 5 new ones + `Battle_Tank_UAV` specifically targeting the GROUND-domain confusion) to get a post-fix accuracy/FAR-MR baseline — the 93.12%/85.2%-GROUND numbers below are all pre-Colab-port, pre-new-dataset
- iPhone Live tab: requires second tunnel (Cloudflare) when phone is not on same WiFi as the notebook's GPU runtime

---

## Notebook crash bugs fixed pre-Colab-port (2026-07-05)

The gap-analysis work (heterogeneous ensemble, FAR/MR, hard-negative mining, threat policy — see `docs/gap_analysis_report.md`) changed `module_b_classifier.py`'s API, but the notebook that actually produces GPU runs was never updated to match. Found and fixed before the Colab port:

- `c-train-ensemble` called `train_ensemble(CONFIG, n_models=6, ckpt_dir=...)` — `n_models` no longer exists (replaced by `architectures`). Would have raised `TypeError` the moment a single model scored below 0.92.
- `c-eval-metrics`'s resume-safe check globbed for `convnext_[0-5].pth` to detect a saved ensemble on disk; new checkpoints are named `{arch}_{i}.pth` (e.g. `convnext_tiny_0.pth`, `resnext50_1.pth`). The glob would never match, so a resumed session would silently fail to find a previously-trained heterogeneous ensemble and retrain from scratch.
- Added a FAR/MR reporting cell right after `c-eval-metrics` (reuses `all_preds`/`all_labels` already collected there — zero extra inference) and an optional hard-negative-mining cell after the confusion matrix (off by default via a flag) — both exist in the codebase but were never wired into the notebook that produces the actual runs.
- `c-pipeline`'s `reats_pipeline()` now runs one untimed warm-up call before timing, then averages 5 warm reps — the 2026-07-04 run's 111.9ms latency figure included one-time GradCAM hook/module-scan construction cost, which the run's own report flagged as "not representative of steady-state."

---

## Colab port + kagglehub integration (2026-07-05)

The notebook was Kaggle-only: `/kaggle/working`/`/kaggle/input` paths throughout, and datasets sourced via Kaggle's "+ Add input" mount panel, which has no Colab equivalent. Ported in two passes:

**Pass 1 — path/environment port.** `/kaggle/working` → `/content` everywhere (c-config, c-sample-grid, c-dist-chart, c-confusion, c-gradcam, c-module-a, c-pipeline, 971d98df, c-dashboard, c-summary). New `c-kaggle-data` setup cell (inserted after `c-clone`, before `c-config`) handles Kaggle auth (Colab Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`, or an upload prompt for `kaggle.json`) and originally downloaded datasets via the `kaggle` CLI (`kaggle datasets download -p <dest> --unzip`) into `/content/datasets/<key>/`. Warm-start (previously a mounted notebook-output input) now uses `kaggle kernels output <owner>/<kernel>` — no kagglehub equivalent for kernel outputs, so the `kaggle` CLI stays installed for this one purpose.

**Bug caught by mock-exec testing (pass 1):** the download loop pre-created the destination directory before attempting the download, so a *failed* download left an empty dir behind — `.exists()` downstream would then wrongly report it as "available" with zero real files (Kaggle's own mount semantics never have this in-between state: a dataset either has content or the mount doesn't exist at all). Fixed by `rmdir()`-ing the empty placeholder on failure. Caught this via a 3-scenario mock-exec test (all-fail / all-succeed / already-present-skip) with `subprocess` stubbed, not by inspection — worth repeating this pattern for any future notebook-cell change that manages its own directory state.

**Pass 2 — switched to `kagglehub`.** User supplied a list of `kagglehub.dataset_download(handle)` snippets and asked to fuse them in. Rewrote `c-kaggle-data` to use `kagglehub.dataset_download()` instead of the `kaggle` CLI subprocess (simpler: no `-p`/`--unzip`/directory bookkeeping, auto-caches under `~/.cache/kagglehub/`, returns the local path directly). `KAGGLE_DATASET_HANDLES` values are now a single handle or a list of handles tried in order — used for 2 of the user's supplied handles that are fallback mirrors (`FLIR_ADAS_v2` → `rajababuadigarla/teledyne-flir-free-adas-thermal-dataset-v2`, `HRSC2016` → `weiming97/hrsc2016-ms-dataset`). 5 of the user's handles were genuinely new datasets, added as new keys: `SARScope_Maritime`, `Ships_Satellite`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV` (targets the T72/Abrams/Leopard2/BMP2/Bradley/K21 GROUND-domain confusion below), `Thermal_Ships` (genuinely IR, unlike most of the optical/RGB "aerial" sets already in the list). **None of these 5 have a `label_maps.yaml` entry yet** — deliberately not fabricated; the ingestion pipeline's UNMAPPED report is the correct way to discover real label strings, same principle as the 2026-07-04 label-map fixes below.

**Second bug caught by mock-exec testing (pass 2):** `c-config` still had its old line reconstructing `DATASET_INPUTS` from the now-removed shared `DATASETS_DIR` — since kagglehub returns a different cache path per dataset (no shared directory to reconstruct from), that line would have silently overwritten every real kagglehub path with a nonexistent stand-in, making every dataset look "missing" downstream despite successful downloads. Removed; `c-config` now trusts `DATASET_INPUTS` as built in `c-kaggle-data`. Also caught by mock-exec (fallback-to-mirror + all-handles-fail + available/missing-count assertions), not by inspection — same lesson as pass 1.

**Both passes:** notebook re-read is no longer possible via the Read tool in one shot once it grows past ~25k tokens (hit this mid-session) — edited the `.ipynb` JSON directly via a Python script instead (`json.load` → mutate `cells_by_id[...]["source"]` → `json.dump(..., indent=1, ensure_ascii=False)`, which round-trips byte-identical to Jupyter's own formatting for untouched cells). Always dry-run against a copy first, diff cell IDs before/after to confirm nothing was reordered/clobbered, and mock-execute any cell with nontrivial control flow (not just `ast.parse` for syntax) before touching the real file.

---

## Kaggle run results (2026-07-04, `real-time-ex-03`, Tesla T4)

Full report: `docs/gap_analysis_report.md`'s sibling Kaggle report (not committed to this repo as of this entry — summarized here so it isn't lost). Single ConvNeXt_tiny only (300 epochs); the notebook's auto-logic skips the 6-architecture ensemble once a single model clears 92%.

| Metric | Target | Result | Verdict |
|---|---|---|---|
| Accuracy (test) | ≥92% | **93.12%** | PASS |
| ECE, raw / temperature-scaled (T=0.760) | ≤0.05 | 0.1226 / **0.0395** | FAIL raw, PASS scaled |
| Faithfulness AUC (deletion / insertion) | ≥0.80 | 0.4908 / 0.7489 | FAIL both |
| End-to-end latency | ≤40ms | 111.9ms (cold-path, includes one-time Grad-CAM construction) | FAIL |
| mAP@0.5 | ≥75% | not evaluated (0 detections — bootstrapped detector untrained on IR) | — |

**Per-domain test accuracy:** NAVAL 99.75%, AIR 95.73%, **GROUND 85.23%** (weak point — armored-vehicle confusion, see below).

**Data composition:** 81.4% synthetic / 18.6% real. Only 8 of 43 classes got any real annotation pool; 7 datasets mapped **0%** of raw annotations (HRSC2016, Ships_Aerial, Ships_Vessels_Aerial, SWIM, SwimmingPool_Car, Vehicle_Dataset, Aerial_Segmentation) — root-caused and (mostly) fixed this session, see below.

**Confusion clusters** (from the 43×43 confusion matrix): (1) armored ground vehicles — BMP2↔Bradley, K21↔BMP2, general T72/T90/Leopard2 bleed, the main GROUND-domain drag; (2) fighter jets — Su27↔F16, MiG21↔F16, matching the paper's own confusable group. Both added to `hard_negative_mining.CONFUSABLE_GROUPS` this session.

---

## Bug fixes applied 2026-07-04 (Kaggle-run-driven)

| File | Fix |
|------|-----|
| `ingestion/label_maps.yaml` | Added `Ships_Aerial: boat`, `Vehicle_Dataset: minivan/pickup/bus`, `SwimmingPool_Car: "1"/"2"` (numeric-index fallback, best-effort car=1/pool=2 — verify against the dataset's own classes.txt if it looks backwards) |
| `ingestion/formats.py` | `parse_folder`/`parse_video_folder` now descend past media-less **wrapper directories** via `_iter_class_leaf_dirs()` — root's immediate children were being read as the class label even when the real per-class folders sat one or more levels deeper under a dataset-slug/version/tooling wrapper (observed raw labels: `'swim_dataset_1.0.0'`, `'ships-aerial-images'`, `'hrsc2016.part02'`/`'dev-tools'`). A directory is a leaf once it has ≥1 direct image/video file; otherwise it's assumed a wrapper and searched one level deeper (capped at 4). Already-flat datasets are provably unchanged (see `smoke_test.py::ingestion.wrapper_dir_descent`) |
| `modules/hard_negative_mining.py` | `CONFUSABLE_GROUPS` extended with `{BMP2, Bradley, K21}` and `{T72, T90, Leopard2}` — the armored-vehicle clusters the Kaggle run's confusion matrix surfaced (kept as 2 separate groups, not merged, per the report's own framing) |
| `modules/module_d_dashboard.py` | `_grad_cam_batch()` replaces per-detection `_grad_cam()` in `run_pipeline` — ONE forward + ONE backward for a whole chunk (sum per-sample target logits before a single `.backward()`, not an approximation: verified bit-identical to the old per-detection algorithm, 1.58× faster measured). `_find_target_layer()` caches the last-Conv2d lookup (`WeakKeyDictionary`), shared with `_eigen_cam`. **Caution for next session:** a naive `.backward(retain_graph=True)` loop (one call per sample) was tried first and measured **2.7× slower** — retaining the graph still backprops the *full* batch every call, so don't reach for that pattern again without re-benchmarking |

---

## Bug fixes applied earlier session, 2026-06-28ish (in order)

| Commit | File | Fix |
|--------|------|-----|
| `ff06531` | `module_c_xai.py` | GradCAM: added `torch.enable_grad()` + `requires_grad_(True)` so it works inside `torch.no_grad()` callers |
| `e19d856` | `01_kaggle_full_pipeline.ipynb` | `c-faithfulness` cell: `importlib.reload()` to bypass stale bytecode, removed outer `torch.no_grad()` |
| `3edc2f1` | `module_c_xai.py` | `faithfulness_*_auc`: added `.copy()` after `[::-1]` to avoid negative-stride numpy error |
| `4248ef1` | `module_c_xai.py` | `np.trapz` → `_trapezoid` alias (NumPy 2.0 renamed it to `np.trapezoid`) |
| `b5a3811` | `01_kaggle_full_pipeline.ipynb` | `c-module-a` cell: fixed `IRDetector(conf=0.25, iou=0.45)` (was using wrong kwarg names) |
| `7ec55df` | `01_kaggle_full_pipeline.ipynb` | `c-dashboard` cell: made self-contained, added inline `ROOT`/`REATS` definitions |
| `4446807` | `01_kaggle_full_pipeline.ipynb` | Added `streamlit` + `watchdog` to deps, added port-readiness poll before ngrok connect |
| `3ab6330` | `01_kaggle_full_pipeline.ipynb` | `c-clone` cell: added `device` fallback when `c-gpu` was skipped |
| `d30e354` | `module_d_dashboard.py` | `load_pipeline()`: unwrap checkpoint dict — try `ema_state_dict` → `model_state_dict` → `state_dict` → raw |
| `5c915a1` | `module_d_dashboard.py` | `run_pipeline()`: fixed `detector.detect(conf_thresh=, iou_thresh=)` (was using `conf=`, `iou=`) |

Earlier session fixes (committed before this log):
- `parse_coco()`: fallback for `filename`/`path` key variants (not just `file_name`)
- `parse_coco()`: `bbox[:4]` slice for rotated-box datasets with 5-element bbox
- `parse_video_folder()`: new function for mp4/avi/mov video datasets
- `process_annotation()`: detect `_frame_idx` field, use `cv2.VideoCapture` for video frames
- `_collect_by_class()`: dataset-level try/except to skip broken datasets
- `label_maps.yaml`: added `HIT_UAV_v2`, `Dataset2_Folders`, `Aerial_Roof_Seg` entries
- `pipeline.py`: added `video_folder` format handler, `Aerial_Roof_Seg` dataset entry

---

## IRDetector kwarg table (critical — asymmetric API)

| Location | Parameter names |
|----------|----------------|
| `IRDetector.__init__()` | `conf=0.25`, `iou=0.45` |
| `IRDetector.detect()` | `conf_thresh=0.25`, `iou_thresh=0.45` |

Never use `conf_thresh` in `__init__` or `conf` in `detect()`.

---

## Dataset handles — superseded, see CLAUDE.md

This section used to list literal `/kaggle/input/...` mount paths (as of 2026-06-28). Obsolete since the 2026-07-05 Colab port: there's no mount anymore, datasets come from `kagglehub.dataset_download(handle)` inside the notebook's `c-kaggle-data` cell. Current source of truth is CLAUDE.md's "Colab notebook workflow" dataset table (20 keys, includes 2 fallback mirrors and the 5 datasets added 2026-07-05) — don't hand-maintain a second copy of that list here.

<details>
<summary>Old snapshot (2026-06-28, kept only for historical diff context)</summary>

```python
DATASET_INPUTS = {
    # Thermal / infrared
    "FLIR_Thermal":       "/kaggle/input/flir-thermal-images-dataset/",
    "FLIR_ADAS_v2":       "/kaggle/input/thermal-dataset-adas/",
    "HIT_UAV":            "/kaggle/input/hit-uav/",
    "HIT_UAV_v2":         "/kaggle/input/datasets/trnhvtunt/dataset1/HIT-UAV-Infrared-Thermal-Dataset-v1.2.1/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c",

    # Video (mp4 clips, no bbox)
    "Dataset2_Folders":   "/kaggle/input/datasets/trnhvtunt/dataset2/",

    # Naval / ship
    "HRSC2016":           "/kaggle/input/hrsc2016/",
    "Ships_Aerial":       "/kaggle/input/ships-in-aerial-images/",
    "Ships_Google_Earth": "/kaggle/input/ships-in-google-earth/",
    "Ships_Vessels_Aerial": "/kaggle/input/ships-vessels-aerial/",
    "SWIM":               "/kaggle/input/swim-ship-wake-imagery/",

    # Air
    "CGI_Planes":         "/kaggle/input/cgi-planes-in-satellite-imagery-w-bboxes/",
    "Airbus_Aircraft":    "/kaggle/input/airbus-aircraft-detection/",

    # Ground vehicle
    "SwimmingPool_Car":   "/kaggle/input/swimming-pool-and-car-detection/",
    "Vehicle_Dataset":    "/kaggle/input/vehicle-dataset/",
    "Aerial_Segmentation": "/kaggle/input/semantic-segmentation-of-aerial-imagery/",

    # Roof segmentation (all labels → null, contributes 0 annotations)
    "Aerial_Roof_Seg":    "/kaggle/input/datasets/atilol/aerialimageryforroofsegmentation/",
}
```

</details>

---

## REATS 43-class taxonomy (from `config/targets.yaml`, corrected 2026-07-04 — the table here previously drifted out of sync with the actual YAML; always check `targets.yaml` directly for ground truth)

| Domain | Count | Classes |
|--------|------:|---------|
| AIR | 24 | F16, F15, F22, F35, Su27, Su35, MiG29, MiG19, MiG21, J20 (fighters); B52, Tu22M, Tu95 (bombers); AH64, Mi24, Ka52 (attack helis); LYNX, UH60, CH47 (utility helis); MQ9, TB2, Shahed136, RQ4, WZ7 (UAV) |
| GROUND | 13 | M1Abrams, T72, T90, Leopard2 (MBT); BMP2, Bradley, BTR80, K21 (IFV/APC); M109, BM21 (artillery); Patriot, Buk, Pantsir (air defense) |
| NAVAL | 6 | PKG, PTG, FastAttack (patrol/fast-attack); Destroyer, Frigate, Corvette (surface combatants) |

Threat levels: 36 RED, 6 ORANGE, 1 YELLOW. `operational_policy` section added 2026-07-03 (Warning/Track/Engagement thresholds + threat_level ceiling).

---

## Module C — XAI known constraints

- `GradCAMExplainer`: model params must have `requires_grad=True` (set in `__init__`); `explain()` uses `torch.enable_grad()` internally
- `faithfulness_deletion_auc` / `faithfulness_insertion_auc`: reversed argsort must be `.copy()` before passing to PyTorch indexing
- `_trapezoid` alias at top of `module_c_xai.py` handles both NumPy ≥2.0 (`trapezoid`) and <2.0 (`trapz`)

---

## Dashboard deployment notes

- **Port:** 8501 (Streamlit default)
- **Tunnel:** ngrok (auth token via env var, never hardcoded)
- **iPhone Live without same WiFi:** Cloudflare Tunnel on port 5000 for the MJPEG streamer
- **Log file:** `/content/streamlit.log` (was `/kaggle/working/streamlit.log` before the 2026-07-05 Colab port)
- **Security:** ngrok and Kaggle tokens go in **Colab Secrets** (left sidebar → 🔑 icon), not notebook cells

---

## File structure (key files only)

```
REATS/
  config/
    targets.yaml                    43-class taxonomy + operational_policy
  ingestion/
    formats.py                      parse_coco / parse_yolo / parse_xml / parse_csv / parse_folder / parse_video_folder
                                     + _iter_class_leaf_dirs (wrapper-dir descent)
    preprocessor.py                 process_annotation() — handles _frame_idx for video
    pipeline.py                     IngestPipeline / load_dataset_annotations / _collect_by_class
    label_maps.yaml                 label → class mapping per dataset key
  modules/
    module_a_detector.py            IRDetector (YOLOv4, pure PyTorch)
    module_b_classifier.py          Heterogeneous 6-arch EnsembleClassifier, TemperatureScaler
    module_c_xai.py                 GradCAMExplainer*, SHAPExplainer, LIMEExplainer, MCDropoutWrapper
    module_d_dashboard.py           Streamlit 5-tab UI; _grad_cam_batch, _find_target_layer cache
    module_e_streamer.py            FastAPI/WebSocket phone-camera streamer
    hard_negative_mining.py         CONFUSABLE_GROUPS (3), mine/oversample/finetune pipeline
    threat_metrics.py               compute_far_mr() — FAR/MR battlefield threat analysis
    threat_policy.py                map_confidence_to_policy() — Warning/Track/Engagement
  notebooks/
    00_baseline.ipynb               local training walkthrough
    01_kaggle_full_pipeline.ipynb   Colab GPU full pipeline (name predates the 2026-07-05
                                     Kaggle→Colab port; datasets still sourced from Kaggle,
                                     now via kagglehub instead of a mount)
  requirements.txt
  verify_env.py
  smoke_test.py                     22 checks, no GPU/data required
  metrics_report.py                 accuracy/ECE/FAR-MR/mAP/latency vs paper targets
```
