# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Two codebases in one repo

| Directory | Purpose | Entry point |
|---|---|---|
| `REATS/` | Primary system — full ATR pipeline (Modules A–E) | `modules/module_*.py` |
| `cadet_atr_project/cadet_atr/` | Research scaffold — domain adaptation experiments | `run_experiment.py` |

The two systems use **different class taxonomies**:
- **REATS**: 43-class taxonomy (AIR / GROUND / NAVAL) defined in `REATS/config/targets.yaml` — single source of truth; all modules load from it automatically
- **cadet_atr**: `fixed_wing, rotary_wing, uav, vessel, vehicle_ground, vehicle_apc` (mapped to REATS classes via `ingestion/label_maps.yaml`)

**Before assuming this file is exhaustive, check:**
- `MEMORY.md` — running log of architectural decisions, bug fixes, and session state across prior Claude Code sessions; the source of truth for what's currently in-flight, recently broken, or pending (e.g. unmapped datasets, unrepeated GPU runs)
- `docs/gap_analysis_report.md` / `docs/gap_analysis_slides.md` — paper-vs-implementation gap analysis (ensemble heterogeneity, FAR/MR, hard-negative mining, operational policy), each gap grounded in a citation from Do et al. (2025)

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
# End-to-end smoke test — no GPU or real data required
python REATS/smoke_test.py

# Evaluate all metrics against paper targets
python REATS/metrics_report.py --cls-weights checkpoints/convnext_best.pth
python REATS/metrics_report.py --quick   # skip slow faithfulness + mAP

# Bootstrap YOLOv4 detector weights (downloads COCO darknet weights, converts to PyTorch)
python REATS/bootstrap_detector_weights.py
# or from an existing checkpoint / darknet .weights file:
python REATS/bootstrap_detector_weights.py --weights checkpoints/my_yolov4.pt
python REATS/bootstrap_detector_weights.py --darknet ~/yolov4.weights

# Ingest raw datasets into the standard REATS split
cd REATS && python -m ingestion.pipeline \
    --datasets FLIR_Thermal:/path/to/flir HIT_UAV:/path/to/hit_uav \
    --out data/ --train 170 --val 30 --test 200

# Validate dataset split (170 train / 30 val / 200 test per class)
python REATS/dataset_validator.py

# Auto-organise images from a raw folder into the split
python REATS/dataset_validator.py --source raw/ --organize

# Generate synthetic / FLIR-remapped training data
python REATS/generate_flir_fallback.py --out REATS/data/
python REATS/generate_flir_fallback.py --flir /path/to/flir_adas/ --out REATS/data/

# Train Module B (ConvNeXt, single model)
cd REATS && python modules/module_b_classifier.py

# Train the full 6-architecture heterogeneous ensemble (ConvNeXt_tiny, ResNeXt50,
# ViT_b_16, Swin_T, VGG16, ResNet18 — one model per architecture, not 6 seeds of one)
cd REATS && python -c "from modules.module_b_classifier import train_ensemble, CONFIG; train_ensemble(CONFIG)"

# Hard-negative mining + fine-tune pass on confusable classes (F16/MiG19/MiG21)
cd REATS && python modules/hard_negative_mining.py --checkpoint checkpoints/convnext_tiny_0.pth --arch convnext_tiny

# FAR/MR battlefield threat report (alongside accuracy/ECE/mAP/latency)
python REATS/metrics_report.py --cls-weights checkpoints/convnext_tiny_0.pth

# Run the Streamlit dashboard
cd REATS && streamlit run modules/module_d_dashboard.py

# Start the iPhone live-feed streamer (Module E)
cd REATS && python modules/module_e_streamer.py --ngrok   # HTTPS via ngrok
cd REATS && python modules/module_e_streamer.py           # HTTP on LAN only
```

MLflow experiment logs write to `REATS/runs/`; checkpoints to `REATS/checkpoints/`.

---

## Common cadet_atr commands

Run from `cadet_atr_project/cadet_atr/`:

```bash
python smoke_test.py
python run_experiment.py --mode baseline_only
python run_experiment.py --mode adapt --strategy histogram    --checkpoint checkpoints/baseline_best.pt
python run_experiment.py --mode adapt --strategy domain_random
python run_experiment.py --mode adapt --strategy finetune    --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode adapt --strategy dann        --checkpoint checkpoints/domain_random_best.pt
python run_experiment.py --mode gap_only --checkpoint checkpoints/dann_best.pt
python run_experiment.py --mode full
```

---

## Testing notes

Neither codebase uses pytest/unittest. Each has one `smoke_test.py` — a flat script of sequential `check(name, fn)` calls (dotted names like `module_b.full_aug_pipeline`, `threat_policy.map_confidence`) with no `if __name__ == "__main__":` guard, so importing the module runs every check and then calls `sys.exit()`. There is no CLI flag to run a single check; to isolate one, either comment out the other `check(...)` calls at the bottom of the file, or copy the target `_test_*()` function's body into a `python -c` snippet with its own imports. `grep -n '^check(' REATS/smoke_test.py` lists all available check names. REATS's suite has ~22 checks (config, both classifiers, XAI, ingestion wrapper-dir descent, Grad-CAM batching, end-to-end latency); cadet_atr's is smaller (config/data/models/adaptation only).

---

## REATS architecture

```
IR Frame
  └─► Module A  module_a_detector.py     YOLOv4 (CSPDarknet53 + SPP + PANet)
                                          detect() → list of {bbox, conf, class_id}
                                          crop_roi() pads each detection
  └─► Module B  module_b_classifier.py   Heterogeneous 6-architecture softmax-averaging
                                          ensemble: ConvNeXt_tiny, ResNeXt50, ViT_b_16,
                                          Swin_T, VGG16, ResNet18 (one model per arch)
                                          train_full_pipeline() for one model
                                          train_ensemble() for all 6
                                          TemperatureScaler for post-hoc ECE calibration
                                          augmentation_viewpoint.py — MultiViewpointAugmentor,
                                          5 UAV/FLIR-physics transforms chained before Kornia aug
                                          hard_negative_mining.py — extra fine-tune pass
                                          on confusable classes (F16/MiG19/MiG21)
  └─► Module C  module_c_xai.py          GradCAM / GradCAM++ / EigenCAM (pytorch-grad-cam)
                                          SHAP DeepExplainer, LIME
                                          MCDropoutWrapper (entropy → OOD flag)
                                          faithfulness deletion/insertion AUC
  └─► Module D  module_d_dashboard.py    Streamlit 5-tab UI
                                          (Live Analysis | Batch | Calibration | About | iPhone Live Feed)
                                          threat_policy.py — confidence + threat_level →
                                          Warning/Track/Engagement CMS policy tier
                                          threat_metrics.py — per-class FAR/MR
                                          (False Alarm Rate / Miss Rate) reporting
  └─► Module E  module_e_streamer.py     FastAPI + WebSocket server
                                          serves HTML capture page to iPhone
                                          /frame → latest JPEG, /status → JSON stats
                                          polled by Module D's iPhone Live Feed tab
```

**Data flow**: Module A produces ROI crops → Module B classifies → Module C explains → Module D displays in real time. Module E feeds live iPhone camera frames into Module D.

### Key implementation details

**Module A** is pure PyTorch — no ultralytics at runtime. The `ultralytics` line in `requirements.txt` is only used by `verify_env.py`. `IRDetector.detect()` runs the full YOLOv4 forward pass + NMS internally. Training uses `MosaicDataset` which expects YOLO-format layout: `data/{split}/images/*.jpg` + `data/{split}/labels/*.txt` (class cx cy w h, normalised).  Default checkpoint: `checkpoints/detector_bootstrap.pt`. `IRDetector(weights=path)` raises `FileNotFoundError` if the path doesn't exist; only `weights=None` gives (intentionally) random init. Dashboard `load_pipeline()` likewise raises on any missing checkpoint — no silent random-weight fallback.

**Module B training quirks**:
- `train_one_epoch` accepts optional `scaler` (AMP GradScaler) and `ema` (ModelEMA) — both `None` by default
- Validation and checkpointing only run from epoch `CONFIG["best_epoch_start"]` (225/300) — intentional per the paper
- `compute_ece` accepts `is_probs=False`; set `True` when passing ensemble output (already softmax)
- `EnsembleClassifier.forward()` returns probabilities (post-softmax mean), not logits — architecture-agnostic, so it works for both the legacy homogeneous ensemble and the heterogeneous one. **Never re-softmax its output**: a double softmax flattens 43-class confidences toward uniform (max ≈ 0.06) and silently breaks the Warning/Track/Engagement confidence thresholds. Temperature-scale ensemble probabilities through log space (`softmax(log(p)/T)`), not by dividing them like logits.
- `preprocess_roi(roi)` converts a BGR/grayscale uint8 ROI to a normalized `(1, 3, 224, 224)` tensor via pure cv2 (no PIL round-trip, ~6× faster) — this is the real-time path used by Module D's iPhone Live Feed tab
- `ece_from_arrays(conf, correct, n_bins)` is the vectorised binning core of `compute_ece` (numpy `searchsorted` instead of an O(bins × N) Python list-comprehension scan per bin) — fuzz-tested to match the original `(lo, hi]`-inclusive binning exactly. `metrics_report.py`'s `collect_predictions()` reuses it: accuracy, ECE, and FAR/MR are all derived from a **single** inference pass over the val/test loader (previously 3 separate passes — 3× the ensemble forward-pass cost for no reason, since none of those three metrics need a second look at the data).
- `ARCHITECTURES = ["convnext_tiny", "resnext50", "vit_b_16", "swin_t", "vgg16", "resnet18"]` — the 6 architectures required by Do et al. (2025); `build_model(arch, num_classes, pretrained)` dispatches to the right builder. `train_ensemble(cfg, architectures=None)` trains one model per architecture (default: all 6) instead of 6 seeds of one architecture — each checkpoint saves its `arch` under the `"arch"` key so `load_ensemble()` / dashboard `load_pipeline()` reconstruct the right network per file. Checkpoints saved before this field existed default to `convnext_tiny` for backward compatibility.
- `train_full_pipeline`'s `aug_pipeline` chains `MultiViewpointAugmentor()` (`augmentation_viewpoint.py`) before `KorniaAugmentPipeline()` — 5 default-on probabilistic transforms modeling UAV/FLIR sensor physics rather than generic image augmentation: `ElevationForeshortening` (oblique-angle compression), `AltitudeVariance` (apparent target scale at high/low altitude, including zoom-out cases standard `RandomResizedCrop` never produces), `ThermalBloom` (hot-target heat bleed), `AtmosphericScintillation` (low-altitude heat shimmer), `IRFixedPatternNoise` (FPA row/column noise + dead pixels). Pass `MultiViewpointAugmentor(p_scale=...)` to uniformly scale every transform's activation probability (e.g. `0.5` for lighter aug).

**Hard negative mining** (`modules/hard_negative_mining.py`): addresses classes the confusion matrix shows bleeding into each other. `CONFUSABLE_GROUPS` has 3 entries: `{F16, MiG19, MiG21}` (the KCI paper's own confusion matrix — similar fighter silhouettes) and, added after the 2026-07-04 Kaggle GPU run reproduced it on the full 43-class taxonomy, `{BMP2, Bradley, K21}` (IFV/APC bleed) and `{T72, T90, Leopard2}` (MBT bleed) — together these drove GROUND-domain accuracy to 85.2% vs 95.7%/99.75% for AIR/NAVAL that run. `mine_hard_negatives(model, dataset, device)` flags samples that are misclassified or have a low top1/top2 softmax margin, restricted to those groups. When the dataset exposes labels without image loading (ImageFolder `.targets`/`.samples`), mining only decodes and forward-passes the confusable-class subset — ~93% of inference skipped for 3 confusable classes out of 43 (now proportionally more with 9 confusable classes across 3 groups). `HardNegativeDataset` oversamples the flagged indices; `finetune_on_hard_negatives(...)` runs a short low-LR pass on top of an already-trained checkpoint — it is a post-hoc addition to the normal 300-epoch schedule, not a replacement for it.

**Threshold-operational policy** (`modules/threat_policy.py`): `map_confidence_to_policy(confidence, threat_level)` maps a detection to one of `NONE / WARNING / TRACK / ENGAGEMENT`. Thresholds and the per-threat-level ceiling live in `config/targets.yaml`'s `operational_policy` section (loaded as `config.OPERATIONAL_POLICY`) — not hardcoded in the module. Engagement authority is capped to RED-threat classes regardless of confidence.

**FAR/MR battlefield threat metrics** (`modules/threat_metrics.py`): `compute_far_mr(labels, preds)` returns per-class FAR (`FP/(FP+TP)` = 1-Precision) and MR (`FN/(FN+TP)` = 1-Recall), plus `red_threat_FAR`/`red_threat_MR` aggregates restricted to `RED_THREATS` classes — the highest-consequence figure, since a missed RED target never raises an alarm. Wired into `metrics_report.py` and the dashboard's Calibration tab.

**Module C**: `GradCAMExplainer` requires `pytorch-grad-cam`; the inline `_eigen_cam` helper in Module D is gradient-free and ~5–10× faster — prefer it for real-time use. `MCDropoutWrapper.forward()` returns `{mean_probs, uncertainty, all_probs}`.

**Module D Grad-CAM performance**: `module_d_dashboard._grad_cam_batch()` computes Grad-CAM for a whole `run_pipeline` chunk with ONE forward pass + ONE backward pass total (not a forward+backward pair per detection) — the N per-sample target logits are summed before a single `.backward()`; since Conv2d/Linear/eval-mode-BatchNorm are all row-independent, this gives each sample's gradient with zero cross-sample leakage, not an approximation (verified bit-identical to the original per-detection algorithm). A naive `.backward(retain_graph=True)` loop — one call per sample — was tried first and measured **2.7× slower** than the code it was meant to replace: retaining the graph still backprops through the *full* batch on every call, so it does N backward-of-N passes instead of N backward-of-1. Only the single-summed-backward form is a real win (measured 1.58×). `_find_target_layer()` caches the last-Conv2d-layer lookup per model in a `WeakKeyDictionary`, shared with `_eigen_cam`. Grad-CAM always runs against `classifier.models[0]` only — a heterogeneous ensemble has no single meaningful "last conv layer" (ViT_b_16/Swin_T's only Conv2d is their patch-embedding stem, not a semantically deep choice for CAM).

**Module D classification performance**: `run_pipeline()` classifies ROIs in device-aware chunks via `_batch_size_for(classifier)` (CPU: 4, GPU: 32) rather than one ensemble forward per detection — measured on a 4-thread CPU, batch 4 runs ~49 ms/img vs ~75 ms/img at batch 32, so bigger isn't always better without a GPU to actually parallelize across. Chunking (not one unbounded batch) matters because an untrained/misconfigured detector can emit thousands of boxes; stacking all of them into a single tensor is not memory-safe. The iPhone Live Feed tab caches its `GradCAMExplainer` instance in `st.session_state` across Streamlit reruns (rebuilding one every frame at ~15 FPS would repeatedly re-scan the model's modules for hook registration) — invalidated only when the loaded model object changes.

**Config / taxonomy**: `REATS/config/__init__.py` loads `targets.yaml` and exports `CLASSES, NUM_CLASSES, TARGET_META, THREAT_COLOR_BGR, RED_THREATS, ORANGE_THREATS, YELLOW_THREATS, OPERATIONAL_POLICY`. All modules import from here — never hardcode class names, counts, or policy thresholds.

### Ingestion pipeline

`REATS/ingestion/` handles raw dataset → REATS split conversion:
- `formats.py` — parsers for COCO JSON, YOLO txt, Pascal VOC XML, CSV, folder-per-class, and video (frame-sampled). `parse_xml` handles both VOC `<bndbox>` and HRSC coords stored directly on the object element; it uses explicit `is not None` element checks (never `find(a) or find(b)` — a childless ET element is falsy, which silently zeroed VOC/HRSC parsing). `parse_folder`/`parse_video_folder` descend past media-less wrapper directories via `_iter_class_leaf_dirs` — a Kaggle mirror sometimes adds one or more extra directories (a dataset slug, a version string, an internal tooling folder) above the real per-class folders, and a naive "root's immediate children are the classes" reading picks up the *wrapper's* name as the label instead (the 2026-07-04 Kaggle run found raw labels `'swim_dataset_1.0.0'`, `'ships-aerial-images'`, `'hrsc2016.part02'`/`'dev-tools'` this way — 100% of those datasets' annotations went unmapped). A directory counts as a real class leaf once it contains at least one direct image/video file; a directory with none is assumed to be a wrapper and searched one level deeper (capped at 4 levels). Already-flat datasets (`Vehicle_Dataset`, `Dataset2_Folders`) are unaffected — their class folders have direct media at depth 0, same as before.
- `label_maps.yaml` — maps every source dataset's raw labels to REATS class IDs (supports `__size_rule__` for area-based disambiguation). Lookup is normalised (lowercase, `-`/space → `_`), so `"Other Vehicle"` matches `other_vehicle`.
- `pipeline.py` — orchestrates parsers + label resolution + `preprocessor.py` patch extraction + stratified train/val/test split. `_resolve_label` returns `(class_id, matched)`; the run prints an **UNMAPPED** report per dataset listing raw labels with no map entry (the fix-list for `label_maps.yaml`). Writes `data/provenance.json`.

**Data provenance (`data/provenance.json`)**: every generated image is tagged `real` (genuine annotated IR pixels, written by the pipeline), `remapped` (real FLIR ROI intensity-remapped, from `generate_flir_fallback.py --mode crop`), or `synthetic` (procedural target). The notebook's provenance cell splits test accuracy by bucket: `real_backed` classes are field-relevant, `synthetic_only` classes are **architecture validation only**. Most classes are currently synthetic-only, so the headline accuracy is an architecture-validation number until label-map coverage grows.

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

`REATS/.gitignore` excludes `data/**`, `checkpoints/**`, `runs/**` (large files). Placeholder `.gitkeep` files track directory structure:
```
data/**
!data/
!data/**/
!data/**/.gitkeep
```
Force-adding `.gitkeep` files requires `git add -f`; ordinary `git add` will ignore them.

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

Single ConvNeXt_tiny achieves ~90.25%; the 6-model softmax ensemble pushes to ~92%. Latency and FPS targets require GPU — CPU numbers are for architecture validation only.

The paper gives no FAR/MR target — these are the professor's additional battlefield-threat-analysis requirement (see `modules/threat_metrics.py`), reported alongside the paper's metrics but not scored against a pass/fail threshold.

---

## Docker deployment

```bash
docker compose up --build
# Dashboard: http://localhost:8501   Streamer (Module E): http://localhost:7860
```

Two services share one image (`Dockerfile` builds from `REATS/requirements.txt` + `REATS/`):
- `dashboard` — Streamlit (Modules A–D), reaches the streamer via `REATS_STREAMER_URL=http://streamer:7860`
- `streamer` — Module E (`module_e_streamer.py --host 0.0.0.0 --port 7860`)

`docker-compose.yml` bind-mounts `REATS/checkpoints`, `REATS/data`, `REATS/runs` so they persist across container restarts. GPU support requires uncommenting the `deploy:` block (needs `nvidia-container-toolkit`) — otherwise both services run on CPU.

---

## Kaggle notebook workflow (`notebooks/01_kaggle_full_pipeline.ipynb`)

Runs natively as a Kaggle Notebook — `/kaggle/working` for writable output, `/kaggle/input` for read-only mounted datasets. (It briefly targeted Google Colab instead, 2026-07-05 to 2026-07-06, via a `kagglehub`-download cell standing in for Kaggle's own mount panel; reverted back to native Kaggle since Colab has no equivalent of Kaggle's **+ Add Input** panel and the download step just duplicated data Kaggle already serves for free via a mount.)

Run cells in order: `c-gpu` → `c-install` → `c-clone` → `c-config` → `c-ingest` → `c-module-b` (or `c-train-ensemble`) → `c-faithfulness` → `c-dashboard`.

**Dataset access — Kaggle "+ Add Input":** attach each dataset in the table below via the right panel's **+ Add Input** search before running `c-config`. Kaggle mounts a regular user's dataset read-only at `/kaggle/input/datasets/<owner>/<slug>/`; an **organization**-owned dataset (e.g. Airbus) mounts one level deeper, at `/kaggle/input/datasets/organizations/<org>/<slug>/` — this asymmetry isn't documented anywhere obvious on Kaggle's side and will silently make an `.exists()` check fail if you assume the regular-user path for an org account. A previous run's checkpoints, attached the same way (**+ Add Input** → Notebook Output), mount at `/kaggle/input/notebooks/<owner>/<kernel-slug>/`. `c-config` builds `DATASET_INPUTS` directly from these fixed paths (no download step, no credentials needed at runtime) — a key whose dataset isn't attached yet just gets a path that doesn't exist, and `c-ingest` already treats that as "skip, fall back to synthetic," so nothing breaks if you haven't attached everything yet.

**Dataset keys (21 total; 5 new relative to the original Kaggle-mounted 15, plus `Airbus_Aircraft` restored as its own key):**

| Key | Kaggle handle | Domain |
|---|---|---|
| `FLIR_Thermal` | `deepnewbie/flir-thermal-images-dataset` | thermal IR |
| `FLIR_ADAS_v2` | `samdazel/teledyne-flir-adas-thermal-dataset-v2` → fallback `rajababuadigarla/teledyne-flir-free-adas-thermal-dataset-v2` | thermal IR |
| `HIT_UAV` | `pandrii000/hituav-a-highaltitude-infrared-thermal-dataset` | thermal aerial |
| `HIT_UAV_v2` | `trnhvtunt/dataset1` | thermal aerial |
| `Dataset2_Folders` | `trnhvtunt/dataset2` | air (video) |
| `HRSC2016` | `weiming97/hrsc2016-ms-dataset` → fallback `guofeng/hrsc2016` | naval |
| `Ships_Aerial` | `andrewmvd/ship-detection` | naval |
| `Ships_Google_Earth` | `tomluther/ships-in-google-earth` | naval |
| `Ships_Vessels_Aerial` | `siddharthkumarsah/ships-in-aerial-images` | naval |
| `Ships_Satellite` *(new)* | `rhammell/ships-in-satellite-imagery` | naval |
| `SWIM` | `lilitopia/swimship-wake-imagery-mass` | naval |
| `SARScope_Maritime` *(new)* | `alibidaran/sarscope` (notebook output) | naval |
| `Thermal_Ships` *(new)* | `houssemhammami525/thermal-ships` | naval (genuinely IR, unlike most "aerial" sets above) |
| `CGI_Planes` | `aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes` | air |
| `Airbus_Aircraft` *(restored)* | `airbusgeo/airbus-aircrafts-sample-dataset` — **organization** account, mounts under `datasets/organizations/` | air |
| `SwimmingPool_Car` | `kbhartiya83/swimming-pool-and-car-detection` | ground |
| `Vehicle_Dataset` | `alpereniek/vehicle-detection-from-satellite-images-data-set` | ground |
| `Aerial_Vehicle_Detection` *(new)* | `llpukojluct/aerial-vehicle-detection-dataset` | ground |
| `Battle_Tank_UAV` *(new)* | `awaisalisaduzai/tank-detection-vit` (notebook output) | ground — targets the T72/Abrams/Leopard2/BMP2/Bradley/K21 confusion (see `hard_negative_mining.CONFUSABLE_GROUPS`) |
| `Aerial_Segmentation` | `humansintheloop/semantic-segmentation-of-aerial-imagery` | mixed |
| `Aerial_Roof_Seg` | `atilol/aerialimageryforroofsegmentation` | (null labels — contributes 0 annotations) |
| notebook output `trnhvtunt/real-time-ex-03` | warm-start checkpoints | — |

**`CGI_Planes` / `Airbus_Aircraft` split (found 2026-07-06):** the Colab port had collapsed these into a single `CGI_Planes` key pointing at the `airbusgeo` handle, silently dropping the original `aceofspades914` CGI_Planes dataset. Both already had complete `ingestion/label_maps.yaml` entries from before the merge, so restoring the second key was a pure notebook/doc fix — no new label-mapping work needed.

**The 5 new keys have no `ingestion/label_maps.yaml` entry yet** — inventing one without inspecting each dataset's actual raw label strings would risk silently mis-mapping classes (exactly the failure mode `_resolve_label`'s **UNMAPPED** report exists to catch). Run `c-ingest`, read its UNMAPPED report, and add real entries from there.

### Known pitfalls

**Stale bytecode (most common issue):** the Kaggle kernel caches `.pyc` files across cell re-runs. After a `git pull` the old compiled bytecode runs, not the new source. The `c-clone` cell clears this automatically:
```python
for _cache in ROOT.rglob('__pycache__'):
    shutil.rmtree(_cache, ignore_errors=True)
for _mod in [k for k in sys.modules if k.split('.')[0] in ('config', 'ingestion', 'modules')]:
    del sys.modules[_mod]
importlib.invalidate_caches()
```
If an error shows a line number that doesn't match the current source, stale bytecode is the culprit — re-run `c-clone`.

**IRDetector kwarg asymmetry:**
- `IRDetector.__init__()` takes `conf=0.25, iou=0.45`
- `IRDetector.detect()` takes `conf_thresh=0.25, iou_thresh=0.45`
These are different parameter names — never mix them.

**Checkpoint loading:** Training saves a wrapper dict `{state_dict, ema_state_dict, epoch, best_val_acc}`. Dashboard `load_pipeline()` unwraps with priority: `ema_state_dict` → `model_state_dict` → `state_dict` → raw dict.

**GradCAM inside `torch.no_grad()`:** `GradCAMExplainer.explain()` uses `torch.enable_grad()` internally and calls `model.parameters().requires_grad_(True)` in `__init__`. Do not wrap XAI calls in `torch.no_grad()`.

**NumPy 2.0:** `np.trapz` was renamed to `np.trapezoid`. The codebase uses `_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))` for backward compat.

**Negative-stride numpy:** `argsort(...)[::-1]` creates a negative-stride view that PyTorch rejects. Always `.copy()` after reversing: `order = np.argsort(sal.flatten())[::-1].copy()`.

**`device` variable scope:** `device` is set in `c-gpu`. If that cell was skipped, `c-clone` defines a fallback:
```python
try:
    device
except NameError:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## Ingestion pipeline (`REATS/ingestion/`)

### Dataset keys and formats

The pipeline maps `DATASET_INPUTS` dict keys to `label_maps.yaml` entries. Kaggle handles/mount paths for each key (see "Kaggle notebook workflow" above for the full list, including the 5 newer keys and the restored `Airbus_Aircraft` key) live in the notebook's `c-config` cell, not here, so this table doesn't drift out of sync with it.

| Key | Format |
|-----|--------|
| `FLIR_Thermal` | coco |
| `FLIR_ADAS_v2` | coco |
| `HIT_UAV` | yolo |
| `HIT_UAV_v2` | coco |
| `Dataset2_Folders` | video_folder |
| `HRSC2016` | xml |
| `Ships_Aerial` | yolo |
| `Ships_Google_Earth` | folder |
| `Ships_Vessels_Aerial` | csv |
| `SWIM` | folder |
| `CGI_Planes` | folder |
| `Airbus_Aircraft` | csv |
| `SwimmingPool_Car` | folder |
| `Vehicle_Dataset` | folder |
| `Aerial_Segmentation` | folder |
| `Aerial_Roof_Seg` | folder |

The 5 datasets added 2026-07-05 (`Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV`) have no format assigned here yet either — same reason as their missing `label_maps.yaml` entries: inspect first via the ingestion pipeline's own error output, don't guess.

### Video dataset support

`parse_video_folder()` samples `frames_per_video=8` evenly-spaced frames from each `.mp4/.avi/.mov` file. Each annotation dict carries a `_frame_idx` field. `process_annotation()` detects this field and uses `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, frame_idx)` instead of `cv2.imread`.

### COCO JSON quirks

Some COCO JSONs use `filename` (no underscore) or `path` instead of the standard `file_name`. `parse_coco()` tries all three. Some rotated-box datasets add a 5th angle value to bbox — `parse_coco()` takes only `bbox[:4]`.

### Dataset-level error isolation

`_collect_by_class()` wraps each dataset's `load_dataset_annotations()` in try/except and prints a warning then continues — a single broken dataset does not abort the whole ingestion run.

---

## Dashboard deployment (Kaggle → browser/mobile)

```python
# In a Kaggle cell — start Streamlit + ngrok tunnel
import subprocess, time, pyngrok.ngrok as ngrok

proc = subprocess.Popen(["streamlit", "run", str(REATS/"modules/module_d_dashboard.py"),
                         "--server.port=8501", "--server.headless=true"])
# Poll until port is ready (ngrok errors if connected before Streamlit binds)
import socket
for _ in range(30):
    try:
        socket.create_connection(("localhost", 8501), timeout=1).close(); break
    except OSError:
        time.sleep(1)

tunnel = ngrok.connect(8501)
print("Dashboard:", tunnel.public_url)
```

**iPhone Live without same WiFi:** Use Cloudflare Tunnel for the mobile MJPEG streamer (free, no account needed):
```bash
# On the Kaggle GPU runtime
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:5000 --no-autoupdate
```
Paste the `*.trycloudflare.com` URL into Dashboard → iPhone Live URL field.

**Security:** Never commit an ngrok token. Use **Kaggle Secrets** (Add-ons → Secrets) instead:
```python
from kaggle_secrets import UserSecretsClient
NGROK_TOKEN = UserSecretsClient().get_secret('NGROK_AUTHTOKEN')
```
No Kaggle API token is needed at runtime — datasets are mounted via **+ Add Input**, not downloaded. Reset any exposed ngrok token immediately at dashboard.ngrok.com.
