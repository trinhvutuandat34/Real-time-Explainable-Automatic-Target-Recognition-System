# MEMORY.md — REATS Session State

Last updated: 2026-07-09
Active branch: `claude/ponytail-codebase-review-ubhpw2`

---

## What this file is for

Running log of architectural decisions, bug fixes, and session context so that future Claude Code sessions can resume without re-deriving the same information.

---

## Current project state

`notebooks/01_kaggle_full_pipeline.ipynb` is the primary execution environment — it runs natively on Kaggle. All REATS modules (A–E) are implemented and the dashboard is functional. **A real GPU training run completed on Kaggle 2026-07-04** — see "Kaggle run results" below; it's the ground truth for what still needs work, superseding guesses made from synthetic-data smoke tests alone. That run has not yet been repeated since.

### What is working
- Module A: `IRDetector` YOLOv4 pure-PyTorch, forward pass + NMS (bootstrapped from COCO darknet weights). As of 2026-07-08 the notebook can actually fine-tune it — see "Module A detection training pipeline added" below — but that fine-tuning has not yet been run on Kaggle, so until it has, heads are still untrained on IR and detection stays 0 on real input by design.
- Module B: heterogeneous 6-architecture ensemble (ConvNeXt_tiny/ResNeXt50/ViT_b_16/Swin_T/VGG16/ResNet18), AMP + EMA, TemperatureScaler calibration, hard-negative mining (3 confusable groups); optional fast-training mode (`CONFIG['enable_fast_train']` / `--fast` / `make_fast_config()`, 75 epochs instead of 300, ~1.5-2h/model vs ~6h) for quota-constrained runs — see `FAST_TRAINING_GUIDE.md`
- Module C: GradCAM / GradCAM++ / EigenCAM, SHAP, LIME, MCDropout, faithfulness AUC — Grad-CAM batched per-chunk in Module D
- Module D: Streamlit 5-tab dashboard (Live Analysis, Batch, Calibration, About, iPhone Live) — FAR/MR + Warning/Track/Engagement policy wired in; ROI classification batched device-aware (see below)
- Module E: FastAPI/WebSocket phone-camera streamer
- Ingestion pipeline: 21 dataset keys as of 2026-07-06 (was 16 pre-2026-07-05; `Airbus_Aircraft` restored as its own key, see "Kaggle revert" below); wrapper-directory descent fixed 2026-07-04
- Notebook: runs natively on Kaggle, datasets mounted via **+ Add Input** (see "Kaggle revert" below)

### What is pending
- **Re-run the full pipeline on Kaggle with a fresh `c-clone`** — two real crashes hit and fixed so far (see "First real Kaggle run of the detection pipeline" below): (1) `c-train-single`'s `FileNotFoundError` from a pre-fix `main` clone — delete stale `data/{train,val,test}/{images,labels}/` dirs first, they're gitignored so a git-pull won't remove them; (2) `c-train-detector`'s `cv2.error: !ssize.empty()` from a genuine `MosaicDataset._mosaic()` slicing bug, now fixed. Neither fix has been re-verified on Kaggle yet — this is the first full run to actually reach Module A training, so treat the next run as the real test of both.
- Fix the 7 zero-mapped-label datasets flagged by the 2026-07-04 run (partially addressed; HRSC2016 may need dataset-content verification, not just a code fix)
- **New**: add `ingestion/label_maps.yaml` entries for the 5 datasets added 2026-07-05 (`Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV`) — none have a mapping yet; inspect via the ingestion pipeline's own UNMAPPED report first, don't guess
- ~~Fine-tune Module A on labeled IR detection data~~ — superseded by the pending item above; the data pipeline and training cell now exist, only the actual Kaggle run is outstanding
- **New (2026-07-08):** actually train the 6 heterogeneous architectures to convergence on GPU — the code (`ARCHITECTURES`, `build_model`, `train_ensemble`) is complete and unit-tested, but the one real GPU run to date (2026-07-04) only trained a single ConvNeXt_tiny, which already cleared 92% (93.12%), so the notebook's `TRAIN_ENSEMBLE = best_val_acc < 0.92` auto-skip never triggered the ensemble path. No heterogeneous-ensemble accuracy number exists yet to compare against the paper's reported 0.92.
- **New (2026-07-08):** tune `operational_policy` confidence thresholds (0.50/0.75/0.90 in `targets.yaml`) against real measured FAR/MR data — current values are a reasonable starting point, not empirically fit to any trained model's actual confidence distribution.
- ~~`docs/gap_analysis_report.md`'s Gap 3 write-up describes `CONFUSABLE_GROUPS` as a single group~~ — fixed 2026-07-08: now documents all 3 groups (`{F16,MiG19,MiG21}`, `{BMP2,Bradley,K21}`, `{T72,T90,Leopard2}`) and the GROUND-domain accuracy finding that motivated the extra two. `docs/gap_analysis_slides.md` still has single-example illustrative mentions (not false claims, just not exhaustive) — left as-is, lower value to touch a presentation deck's talking points than a factual doc.
- Investigate faithfulness AUC failure (0.49 deletion, target ≥0.80) — likely tied to the 81%-synthetic corpus, needs real-data share to grow before re-testing
- Re-run hard-negative fine-tune (`hard_negative_mining.py`) against a real trained checkpoint to confirm it reduces the fighter-jet and armored-vehicle confusion the Kaggle run's confusion matrix showed
- **New**: re-run the full pipeline natively on Kaggle (datasets mounted via + Add Input, including the 5 new ones + `Battle_Tank_UAV` specifically targeting the GROUND-domain confusion, plus the restored `Airbus_Aircraft` key) to get a post-fix accuracy/FAR-MR baseline — the 93.12%/85.2%-GROUND numbers below are all pre-new-dataset
- **Partly done (2026-07-06 ingest run, see "Kaggle path sync" below):** of the 9 previously-unverified `c-config` paths, 4 now confirmed resolving (`Ships_Satellite`, `Thermal_Ships`, and `SARScope_Maritime`/`Battle_Tank_UAV` after the user repointed the latter two at notebook-output substitutes). Still unresolved: `Aerial_Vehicle_Detection` (user pasted a malformed path — see below), `HIT_UAV_v2`, `Dataset2_Folders` (neither attached that run); the 2 fallback mirrors stayed untested because both primaries resolved
- **Largely resolved 2026-07-06 (see "Ingestion parser fixes" below):** the "UNMAPPED = wrapper-directory names" symptom (`images`, `jpegimages`, `masks`, …) was NOT a wrapper-dir-descent issue — it was `_find_image` over-stripping Roboflow `.rf.HASH` stems, making YOLO parsers return 0 and folder-fall-back. Fixed. `SARScope_Maritime`, `Thermal_Ships`, `Ships_Vessels_Aerial` now map real naval data. Still open: `HRSC2016` (path/mirror), `SWIM` (rotated-box XML), `Ships_Satellite` (filename-prefix classification format), `Aerial_Segmentation` (land-cover only — dead end), and the `Battle_Tank_UAV`/`Aerial_Vehicle_Detection` junk substitute paths
- iPhone Live tab: requires second tunnel (Cloudflare) when phone is not on same WiFi as the notebook's GPU runtime

---

## First real Kaggle run of the detection pipeline — 2 crashes + 1 warning, fixed (2026-07-09)

The user ran the notebook end-to-end on Kaggle (Tesla T4, `real-time-ex-05`) and hit two real, sequential crashes, then a training-loop warning — the first real execution of this session's detection-ingestion and Module A training code, so each thing found here was previously-dormant code, not a regression.

**Crash 1 — the ImageFolder bug, but from a pre-fix clone.** The log showed the *exact* `FileNotFoundError: Found no valid file for the classes labels` already fixed at commit `9012a9f` (see "run_detection() broke Module B training" below), with `dataset_validator.py` reporting `train/images`, `train/labels`, etc. sitting directly under `data/{split}/` — the smoking gun: the fixed code prints an `Output : ...` line that this run's log didn't have, proving `c-clone` fetched a `main` commit that predated the fix. Confirmed `origin/main` now has all three of this session's `pipeline.py`/`module_b_classifier.py` fixes (subtree relocation, `_REATSImageFolder`, `_by_class_cache`) and that no other writer in the codebase ever puts a non-class-named directory under `data/{split}/` (`run()`, `generate_flir_fallback.py`, and `dataset_validator.py --organize` all write through a real `cls` variable). Remediation given to the user: delete the stale `data/{train,val,test}/{images,labels}/` dirs (gitignored, so a git-pull alone won't remove them) and re-run.

**Crash 2 — a genuine pre-existing bug in `MosaicDataset._mosaic()`, surfaced by this being its first real execution.** After crash 1 was worked around, `c-train-detector` got further and hit `cv2.error: ... !ssize.empty() in function 'resize'` inside the DataLoader worker, at `mosaic = cv2.resize(mosaic[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2], (s, s))`. Root cause: `cx`/`cy` are drawn from `random.randint(s//4, 3*s//4)`; whenever `cy < s//2` (~50% of draws), `cy - s//2` is negative, and NumPy interprets a negative slice **start** as "count from the end" rather than clamping to 0 — `mosaic[-160:480]` on a `(2s, 2s)` canvas becomes `mosaic[1120:480]`, start > stop, a silently empty (not erroring) slice that only blows up downstream when `cv2.resize` rejects it. This method predates this session entirely (`IRDetector.train()` existed already, just never run for real — Module A training was always the flagged-pending item), but it's squarely this session's responsibility to fix since `c-train-detector` is what finally exercised it for the first time. Fixed by clamping both bounds (`max(0, cy - s//2)`, `min(2*s, cy + s//2)`, same for `cx`) — the same pattern `IRDetector.crop_roi()` already uses elsewhere in the same file. Verified arithmetically (no numpy/cv2 in this dev sandbox): of the 321 possible `cy` draws at `imgsz=640`, 160 (~50%) would have hit the old empty-slice bug; the clamped version produces zero empty slices across that full range and three other tested `imgsz` values (320/416/960), with crop width always ≥ 3s/4.

**Warning 3 — redundant `torch.tensor(existing_tensor)` in `YOLOv4Loss._build_targets()`.** Training got past both crashes and started actually running, surfacing a `UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()...` at `module_a_detector.py:440-444`. Not a crash, but repeats per target box × per scale × per training step — left running, it would balloon a full multi-epoch Kaggle log. `bw_p`/`bh_p` (pixel-space target width/height) are computed as `bw * W * stride` where `bw` is a tensor slice (`t[4]` from the `targets` tensor) — already a tensor, so wrapping them again in `torch.tensor(bw_p)` before passing to `torch.min(anc_t[:, 0], torch.tensor(bw_p))` was pure redundant re-copying; `torch.min` already broadcasts a `(A,)` tensor against a 0-dim scalar tensor directly. Fixed by removing the redundant wrap. Checked every other `torch.tensor(...)` call in the file (grep) — all others wrap a plain Python float/list (anchors, zero-loss accumulators), which is the legitimate use case and doesn't trigger this warning; this was the only instance of the tensor-wrapping-a-tensor pattern.

**Also confirmed from this run's logs (not yet acted on, left for the user to decide)**: detection ingestion actually found solid real data once the location bug is out of the way — 2,838 images, 7,549 boxes. Several datasets still leave real annotations unmapped: `Battle_Tank_UAV` has no `label_maps.yaml` entry at all; `SARScope_Maritime`'s 19,431 annotations are all labelled the bare numeral `'1'` (needs the dataset's own class schema to map correctly, don't guess); `Thermal_Ships` has 14 unmapped real vessel labels, some possibly mappable (`warship`) but most civilian types with no REATS equivalent; `HRSC2016` is still reading wrapper/tooling directory names instead of real ship classes; `Aerial_Segmentation`/`Aerial_Roof_Seg` are pixel-mask datasets structurally incompatible with this folder-per-class/bbox pipeline (dead ends, not fixable without a different extraction path).

---

## 8-angle code review found + fixed 5 more real bugs (2026-07-08)

A full recall-biased code review (8 parallel finder agents: line-by-line, removed-behavior, cross-file, language-pitfalls, reuse, simplification, efficiency, altitude, conventions) of this session's cumulative diff. Three agents independently converged on the same set of correctness bugs from different angles — strong triangulation. All fixed and pushed.

**Grad-CAM crashed silently on GPU** (`module_d_dashboard.py`, `_grad_cam_batch`): `idx`/`tgt` index tensors used to fancy-index `logits[idx, tgt]` were built with no `device=` argument, so they stayed on CPU even after the earlier session's fix moved the classifier (and hence `logits`) onto `cuda`. PyTorch fancy-indexing requires matching devices; this raised inside the function's own broad `except Exception`, so enabling the Grad-CAM/XAI checkbox on a GPU-loaded classifier silently returned `heatmap=None` for every detection with no visible error. Fixed by placing `idx`/`tgt` on `batch_in.device`.

**`run_detection()` stem collisions** (`ingestion/pipeline.py`, `_stem_for`): two different source images could sanitize to the identical output filename — same basename in different folders (e.g. two per-scene dumps both containing `0001.jpg`), or two paths whose differing characters (`:`, space, etc.) both collapse to `_`. The second image's `save_frame()`/`write_yolo_labels()` call silently overwrote the first's on disk, while the run's printed summary still counted both. Fixed by appending a short deterministic hash of the full, un-sanitised key — collision-proof while staying reproducible across calls (required for resume-safety).

**`run_detection()` orphaned images on interrupted writes**: resume-safety's "already ingested" check only looked at whether the `.jpg` existed, not the paired `.txt` label. `save_frame()` and `write_yolo_labels()` are two separate, non-atomic writes; a Kaggle timeout/OOM between them left an image with no label, and the next call would see the `.jpg` and skip that stem forever — `MosaicDataset` treats a missing label file as "zero objects," so this would silently mistrain Module A on a false-negative frame. Fixed to require both files present before treating a stem as done.

**`ImageFolder` hardened against non-class directories** (`module_b_classifier.py`, new `_REATSImageFolder`): the subtree-relocation fix (previous section below) stopped `run_detection()` specifically from writing into `data/{split}/` directly, but `build_loaders()`'s plain `ImageFolder` still had no defense against *any other* future writer doing the same and reproducing the identical crash — it treats every subdirectory as a class with no filter. Added a small `ImageFolder` subclass overriding `find_classes()` to only accept directories whose name is an actual REATS class (`config.CLASSES`); a stray non-class directory is now ignored rather than crashing or silently becoming a bogus 44th class. This is the deeper fix the altitude-angle reviewer asked for — defense in depth, not reliant on every future ingestion path remembering the convention.

**Redundant dataset re-parse eliminated**: `run_detection()` called `self._collect_by_class()` independently of `run()`, so the notebook's back-to-back `c-ingest` + `c-ingest-detection` cells (same `pipe` object) re-parsed every attached dataset's COCO/XML/YOLO annotations and re-ran label resolution twice for identical output — roughly doubling ingestion wall time with 21 datasets attached. Fixed by caching `_collect_by_class()`'s result on the `IngestPipeline` instance (safe: `self.datasets` is fixed for the instance's lifetime, no setter exists to invalidate it).

Verified: `py_compile` on all touched files, plus standalone pure-Python reimplementations (no cv2/numpy/torch/torchvision in this dev sandbox) of the stem-collision fix, the orphan self-healing, the `find_classes()` filtering, and the caching behavior — all pass. **Still not verified on a real Kaggle GPU run** — same caveat as everything else in this session's ingestion/training work.

Findings surfaced but not acted on (lower priority — cosmetic/maintainability, not correctness): `run_detection()`'s unused `detection_root` parameter; duplicated print-banner formatting between `run()` and `run_detection()`; `_classifier_device()` re-deriving a value already available as `detector.device` at each of its 3 call sites; Module A's fast-mode epoch count (`c-train-detector`) as a standalone hardcoded ternary instead of going through `make_fast_config()` like Module B's does.

---

## run_detection() broke Module B training — fixed (2026-07-08)

The first real Kaggle run of the newly-added detection pipeline crashed `c-train-single` (Module B classification training, which runs *after* `c-ingest-detection` in cell order) with:

```
FileNotFoundError: Found no valid file for the classes labels. Supported extensions are: .jpg, .jpeg, ...
```

Root cause: `run_detection()` wrote `data/{split}/images/*.jpg` + `data/{split}/labels/*.txt` directly under the *same* `data/{split}/` root that `build_loaders()`'s `torchvision.datasets.ImageFolder` scans for Module B. `ImageFolder` treats every subdirectory of `data/{split}/` as a class name — it doesn't distinguish "real class folder" from "some other directory someone left here." So `images/` and `labels/` got picked up as two bogus classes, and `labels/` has zero `.jpg`/`.png`/etc. files in it (only `.txt`), which is exactly what `ImageFolder.make_dataset()`'s `FileNotFoundError` is checking for — "labels" in that error message is a literal (bogus) class name, not a description. This was introduced when `run_detection()` was added earlier the same day and not caught here because `numpy`/`cv2` aren't installed in the dev sandbox this was written in — the pure-Python logic tests run at the time covered the grouping/split-ratio/box-math correctness but had no way to exercise `ImageFolder`'s actual directory-scanning behavior. First real Kaggle GPU run caught it immediately.

Fix: `run_detection()` now writes to `out_root/"detection"/{split}/images|labels/` — a separate subtree — instead of `out_root/{split}/images|labels/`. Added a `detection_root` parameter (defaults to the safe location) in case a caller wants a different path. Updated the notebook's `c-ingest-detection` (existing-file scan + zero-count check) and `c-train-detector` (`data_root=` passed to `IRDetector.train()`, and where it reads back `best.pt`) to point at `data/detection/` accordingly.

**Remediation for a Kaggle session that already hit this**: the crash means `data/{train,val,test}/images/` and `.../labels/` already exist on disk from the broken run — these are gitignored (`data/**`), so re-running `c-clone` (git pull) does **not** remove them; only the code that *writes* to a new location changed, nothing cleans up the old one. Before re-running, delete the 6 stale directories:
```python
import shutil
for s in ('train', 'val', 'test'):
    for d in ('images', 'labels'):
        shutil.rmtree(DATA_DIR / s / d, ignore_errors=True)
```
Then re-run `c-ingest-detection` (writes fresh under `data/detection/` this time) and `c-train-single` should no longer see the bogus classes.

---

## MEMORY.md itself got emptied by a bad merge, then reconciled (2026-07-08)

A `Merge branch 'main' into claude/ponytail-codebase-review-ubhpw2` resolved a MEMORY.md conflict by deleting the file entirely (293 lines → 0) — this branch's "Dashboard GPU device bug" section and `claude/init-gba5ns`'s "Fast-training mode + EMA bug"/"Gap-analysis re-verification" sections had each been added independently, and whatever resolved the conflict picked neither. PR #70 then merged that empty state into `main`. Separately, a direct (non-PR) commit to `main` (`b2f3447`) restored *some* version of the file — but it turned out to be byte-identical (modulo whitespace) to an intermediate commit on this branch, missing the two sections that weren't part of that snapshot.

Reconstructed properly via a 3-way merge (`git merge-file`, merge-base = the last commit both lineages shared) keeping every unique section from both sides — nothing dropped. Confirmed `main`'s direct-commit content was a strict subset of the reconciled version before resolving the second conflict (merging `main` back into this branch) in this branch's favor.

**Takeaway for future sessions**: if a merge into/from a branch touches MEMORY.md and the result looks suspiciously short, check `git show <merge-commit> --stat` for a `NNN -------` line with zero insertions — that's the signature of a conflict resolved by deletion, not a real edit. Diff file sizes across the merge parents before trusting the result.

---

## Fast-training mode added + critical EMA bug fixed (2026-07-08)

User has limited Kaggle GPU quota and can't afford the full ~18-24h heterogeneous-ensemble
training run (~6h for a single model). Added an opt-in fast mode to
`modules/module_b_classifier.py` rather than permanently shrinking the paper's schedule:

- `CONFIG['enable_fast_train']` / CLI `--fast` / `make_fast_config(cfg)` → 75 epochs instead
  of 300, validation/checkpointing from epoch 10 instead of 225, 3-epoch warmup instead of 10,
  lightweight 5-transform `KorniaAugmentPipeline(full=False)` instead of the full
  `MultiViewpointAugmentor` + 10-transform pipeline. ~1.5-2h/model, ~12h for the 6-model
  ensemble (vs ~6h/~24h full). Expected trade: ~1-2% lower final accuracy.
- Full details, expected-accuracy tables, and a recommended fast-then-full two-phase workflow
  are in the new `FAST_TRAINING_GUIDE.md` (repo root). Kaggle notebook cell snippet in the new
  `KAGGLE_FAST_TRAINING_CELLS.md`.

**Critical bug caught on recheck, before any GPU time was spent:** the first draft cut epochs
but left `ema_decay=0.9999` unchanged. EMA has a time constant of `1/(1-decay)` steps — 0.9999
= 10,000 steps. A 75-epoch run is only ~4,300 steps, so the EMA weights would have stayed
**~65% initialization** by the end of training. This matters because `evaluate()` validates
against `ema.module` (not the raw model) and checkpoints save `ema_state_dict`, which
`load_ensemble()`/the dashboard load *first* — so fast mode would have silently produced a
near-random checkpoint reporting a plausible-looking but meaningless val accuracy. Fixed:
`make_fast_config()` now also sets `ema_decay=0.999` (1,000-step constant → ~1.4% init weight
remaining at 75 epochs). Full-mode training (300 epochs, 0.9999) is untouched — the bug only
existed in the interaction between short schedules and the paper's EMA decay constant. **If a
future session shortens `epochs` again for any reason, re-check this ratio — the EMA constant
does not automatically scale with schedule length.**

Also fixed: `--fast` cut epochs but never set `enable_fast_train=True`, so the CLI path
silently trained with the FULL augmentation pipeline despite the notebook path (which sets the
flag directly) using the lightweight one — the two entry points diverged. `make_fast_config`
now sets the flag itself, and `main()` just flips it and lets `train_full_pipeline` apply the
config once, so CLI and notebook are provably identical.

Two additional speedups added while in the file (help full-mode training too, zero accuracy
cost): `torch.backends.cudnn.benchmark=True` on CUDA (auto-tunes conv kernels for the fixed
224×224 input), `persistent_workers=True` in `build_loaders()` (workers survive across epochs
instead of respawning each one).

Committed as `36263dd` on `claude/init-gba5ns`, pushed. Verified via AST-parse +
`make_fast_config` idempotency/no-mutation/CLI-parity logic checks only — no GPU/torch
available in this container, so the actual training run and its accuracy/ECE numbers are still
future work on Kaggle.

---

## Gap-analysis re-verification (2026-07-08) — all 4 gaps confirmed already closed

User described 4 claimed gaps between the codebase and the KCI (2025) paper / professor's
requirements (heterogeneous ensemble, FAR/MR split, hard-negative mining, Warning/Track/
Engagement policy) for slide/report use, as if currently open. Read each relevant file directly
(not just the docs) to verify — **all 4 were already closed**, from the
`claude/model-gaps-ensemble-metrics-e4xysn` work now merged into this branch:

1. `ARCHITECTURES` (6 distinct builders) + `train_ensemble()` defaulting to one-model-per-arch
   — confirmed in `module_b_classifier.py`.
2. `compute_far_mr()` — confirmed in `threat_metrics.py`.
3. `CONFUSABLE_GROUPS` — confirmed in `hard_negative_mining.py`, and actually **ahead of** the
   report (see the stale-doc pending item above).
4. `map_confidence_to_policy()` — confirmed in `threat_policy.py`, matches the user's
   description exactly (0.50/0.75/0.90 thresholds, RED/ORANGE/YELLOW ceiling).

All 4 have matching `smoke_test.py` checks (`module_b.heterogeneous_ensemble`,
`threat_metrics.far_mr`, `hard_negative_mining.mine`, `threat_policy.map_confidence`). No code
changed in this pass — read-only verification. Real remaining work is execution, not
implementation — see the 3 new 2026-07-08 pending-list items above (train the ensemble on GPU;
tune thresholds against real data; fix the stale doc).

---

## Dashboard GPU device bug + Module A detection training pipeline added (2026-07-08)

A user reported the dashboard's Live Analysis tab showing 84.6ms latency (target 40ms) and "No targets detected" on an uploaded image, on what was presumably a Kaggle GPU session. Root-caused two independent issues:

- **Classifier silently ran on CPU regardless of GPU availability.** `load_pipeline()` built the detector via `IRDetector(weights=...)`, which auto-selects `cuda`, but never moved the classifier there — every input tensor built by `run_pipeline()`, the Calibration tab, and `preprocess_roi()` (iPhone Live Feed) was left on CPU too. Fixed by adding a `_classifier_device()` helper and moving the classifier + input tensors to it consistently across all 4 inference call sites. This also uncovered two dormant `.numpy()`-on-GPU-tensor crashes in `_grad_cam_batch`/`_eigen_cam` that had only "worked" because the classifier was always on CPU — fixed alongside.
- **"No targets detected" was expected, not a bug**: `detector_bootstrap.pt` only carries COCO backbone/neck weights (`bootstrap_detector_weights.py` explicitly strips detection heads on class-count mismatch) — heads were randomly initialized for the 43-class IR taxonomy, and nothing in the ingestion pipeline wrote data in the format `IRDetector.train()`/`MosaicDataset` expects (`data/{split}/images/*.jpg` + `labels/*.txt`, YOLO `class cx cy w h`) — `IngestPipeline.run()` only ever wrote classification crops for Module B.

Closed the second gap:
- `ingestion/preprocessor.py`: split `process_annotation`'s image-loading branch into a reusable `load_frame()`; added `to_ir_look()` (class-agnostic thermal-look conversion — no per-class intensity remap, since a full multi-object frame has no single target class) and `save_frame()`/`write_yolo_labels()`.
- `ingestion/pipeline.py`: added `IngestPipeline.run_detection()` — groups the same label-mapped annotations by *source image* instead of by class, assigns each image to one split, writes full frame + YOLO box labels. Annotations with `bbox=None` (folder/video-folder datasets — class label only, no localisation) are skipped rather than turned into a whole-frame box. **Resume-safe**: computes each image's output stem deterministically up front and skips anything already written by a prior call, so re-running after attaching a new Kaggle dataset mid-session doesn't reshuffle `self.rng` (shared, stateful, with `run()`) across already-written images — an earlier version of this fix did reshuffle on every call, which could silently move an image from train to val/test on a second run and leak it across both. Added `--detection` CLI flag.
- Notebook: `c-ingest-detection` (after `c-ingest`) runs the new pass; `c-train-detector` (Section 5, after the bootstrap cell) actually trains Module A, warm-started from the bootstrap checkpoint, epoch count tied to `enable_fast_train` (20 fast / 60 full), guards on ≥4 train images since `MosaicDataset`'s mosaic augmentation needs at least that many to sample from. Saves `checkpoints/detector_trained.pt`. The dashboard sidebar's "Detector weights path" default now prefers `detector_trained.pt` over `detector_bootstrap.pt` once it exists — previously the default was hardcoded to bootstrap, so even a fully trained detector would never load without the user manually retyping the path.
- **Not yet verified**: none of this has run on a real Kaggle GPU session yet. `numpy`/`cv2` aren't installed in the dev sandbox this was written in, so verification was py_compile + notebook-JSON validity + an independent pure-Python reimplementation of the grouping/split/box-normalisation/resume-safety math (all passed) — not an actual execution. First real Kaggle run should confirm: (a) `c-ingest-detection` finds a nonzero image/box count from at least one bbox-carrying dataset, (b) `c-train-detector` actually saves a checkpoint (mAP@0.5 > 0 at some point in training).

Same session also: wired `enable_fast_train`/`make_fast_config` (75-epoch fast mode) into the notebook itself (previously only existed in `module_b_classifier.py`, unreachable from the Kaggle notebook since `c-clone` pulled `main`, which didn't have it yet — see PR #65), and fixed a `persistent_workers` shared-memory accumulation issue between ensemble models in `train_full_pipeline`.

---

## Notebook crash bugs fixed (2026-07-05)

The gap-analysis work (heterogeneous ensemble, FAR/MR, hard-negative mining, threat policy — see `docs/gap_analysis_report.md`) changed `module_b_classifier.py`'s API, but the notebook that actually produces GPU runs was never updated to match. Found and fixed:

- `c-train-ensemble` called `train_ensemble(CONFIG, n_models=6, ckpt_dir=...)` — `n_models` no longer exists (replaced by `architectures`). Would have raised `TypeError` the moment a single model scored below 0.92.
- `c-eval-metrics`'s resume-safe check globbed for `convnext_[0-5].pth` to detect a saved ensemble on disk; new checkpoints are named `{arch}_{i}.pth` (e.g. `convnext_tiny_0.pth`, `resnext50_1.pth`). The glob would never match, so a resumed session would silently fail to find a previously-trained heterogeneous ensemble and retrain from scratch.
- Added a FAR/MR reporting cell right after `c-eval-metrics` (reuses `all_preds`/`all_labels` already collected there — zero extra inference) and an optional hard-negative-mining cell after the confusion matrix (off by default via a flag) — both exist in the codebase but were never wired into the notebook that produces the actual runs.
- `c-pipeline`'s `reats_pipeline()` now runs one untimed warm-up call before timing, then averages 5 warm reps — the 2026-07-04 run's 111.9ms latency figure included one-time GradCAM hook/module-scan construction cost, which the run's own report flagged as "not representative of steady-state."

---

## Kaggle revert (2026-07-06)

Switched `01_kaggle_full_pipeline.ipynb` back to native Kaggle **+ Add Input** dataset mounting at the user's request, replacing a credential-requiring download cell (`c-kaggle-data`, deleted outright) that re-downloaded data Kaggle otherwise serves for free via a mount. `c-config`'s warm-start step changed from a subprocess-based kernel-output download to reading directly from a mounted `/kaggle/input/notebooks/<owner>/<slug>/` path, since the user attaches previous runs as inputs instead of downloading their output at runtime.

**`CGI_Planes` / `Airbus_Aircraft` are two separate dataset keys, not one:** `CGI_Planes` maps to `aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes`; `Airbus_Aircraft` maps to `airbusgeo/airbus-aircrafts-sample-dataset`. Both have complete `ingestion/label_maps.yaml` entries (lines 195 and 219) — don't collapse them into a single key. Caught by reconciling the user's actual attached Kaggle inputs against `KAGGLE_DATASET_HANDLES`; a reminder that `KAGGLE_DATASET_HANDLES`-style dicts should be spot-checked against real attached inputs occasionally, not assumed correct just because the code runs.

**Kaggle mount-path convention (new finding):** regular-user datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`; **organization**-owned datasets (Airbus is a Kaggle organization account, not a user) mount one level deeper, at `/kaggle/input/datasets/organizations/<org>/<slug>/`. This asymmetry isn't obviously documented by Kaggle and will silently fail an `.exists()` check if you guess the regular-user path for an org-owned dataset. Confirmed directly from the user's own attached-input paths, not guessed.

**Dataset path confidence:** 12 of 21 keys' mount paths are confirmed from the user's actual attached Kaggle inputs (`FLIR_Thermal`, `FLIR_ADAS_v2` primary, `HIT_UAV`, `HRSC2016` primary, `Ships_Aerial`, `Ships_Google_Earth`, `Ships_Vessels_Aerial`, `SWIM`, `SwimmingPool_Car`, `Vehicle_Dataset`, `Aerial_Segmentation`, `Aerial_Roof_Seg`) plus the 2 restored/corrected keys (`CGI_Planes`, `Airbus_Aircraft`). The remaining 9 (`FLIR_ADAS_v2`'s and `HRSC2016`'s fallback mirrors, `HIT_UAV_v2`, `Dataset2_Folders`, `Ships_Satellite`, `SARScope_Maritime`, `Thermal_Ships`, `Aerial_Vehicle_Detection`, `Battle_Tank_UAV`) use the same `/kaggle/input/datasets/<owner>/<slug>/` pattern as a best-guess — per user decision, this is safe because `c-ingest` already treats a nonexistent mount path as "skip, fall back to synthetic," so an unattached/wrong guess degrades gracefully instead of crashing. Worth confirming once attached. **(Partly resolved next day — see "Kaggle path sync (2026-07-06)" below; 2 of these were repointed at notebook-output mounts, so the "`/datasets/<owner>/<slug>/`-pattern for all 9" statement no longer holds for `SARScope_Maritime`/`Battle_Tank_UAV`.)**

**Warm-start switched to `real-time-ex-03`** (was `real-time-ex-01`) — per user decision, since `real-time-ex-03` is the run that actually produced the documented 93.12%-accuracy checkpoint (see "Kaggle run results" below), making it the more useful checkpoint to resume from. The user has 6 previous-notebook-output inputs attached (`reats-1`, `real-time-explainable-automatic-target-recognition`, `real-time-ex-01` through `04`) — only `real-time-ex-03` is wired into `WARM_START_KERNEL`.

**Verification used, no Kaggle/GPU account available in this environment:** JSON round-trip validity + unchanged cell IDs/order + `ast.parse()` on every edited code cell + a 4-scenario mock-exec of the new `c-config` cell (nothing attached / all 21 attached / only fallback mirrors attached / warm-start mount with a real checkpoint file) with `Path.exists`/`Path.rglob`/`shutil.copy2` stubbed. All passed, including confirming `Airbus_Aircraft` and `CGI_Planes` resolve to distinct paths and the org-account path is used for `Airbus_Aircraft`.

Files touched: `REATS/notebooks/01_kaggle_full_pipeline.ipynb`, `CLAUDE.md` (Kaggle notebook workflow section, ingestion format table, dashboard deployment section), `MEMORY.md` (this entry), `README.md` (Kaggle workflow section reconciled with the notebook's actual final cell list — that section had drifted stale even before this revert).

### Kaggle path sync (2026-07-06, driven by the user's first native-Kaggle ingest run)

The user ran `c-ingest` on Kaggle and pasted back their live `DATASET_INPUTS` dict plus the run log. Synced the repo notebook to the paths that actually resolved:
- `SARScope_Maritime`: `kailaspsudheer/sarscope-unveiling-the-maritime-landscape` (dataset) → **`/kaggle/input/notebooks/alibidaran/sarscope`** (a notebook-output mount the user substituted) — confirmed resolving (in the run's 18 loaded datasets).
- `Battle_Tank_UAV`: `simuletic/uav-and-aerial-view-battle-tank-detection-dataset` (dataset) → **`/kaggle/input/notebooks/awaisalisaduzai/tank-detection-vit`** (notebook-output mount) — confirmed resolving.
- **`Aerial_Vehicle_Detection`: NOT synced.** The user's pasted path was malformed — `/kaggle/input/datasets/kaggle/input/datasets/rhammell/ships-in-satellite-imagery` (doubled `/kaggle/input/datasets/` prefix, and it points at `Ships_Satellite`'s dataset, not a ground-vehicle one). It correctly did not resolve (absent from the run's 18). Left the repo at the well-formed `llpukojluct/aerial-vehicle-detection-dataset` and flagged it back to the user for the intended source — did not propagate a known-broken path into the repo.
- `Dataset2_Folders` (`trnhvtunt/dataset2`) wasn't visible in the user's pasted dict and wasn't among the 18 loaded, but left it in the repo (non-destructive — an unattached key just gets skipped); flagged for confirmation.

These 2 substitutes are notebook-output mounts (`/kaggle/input/notebooks/...`), so their on-disk folder structure may differ from the original dataset handles' — a concern only once they get `label_maps.yaml` stubs and the pipeline actually parses them (all 4 resolving new keys are still `skipping`-for-no-map, so 0 patches so far regardless).

### Ingestion parser fixes (2026-07-06, from the user's per-dataset structure dumps)

The user pasted the real on-disk tree of every UNMAPPED dataset. Two problems, both now fixed, plus a triage of what's worth using.

**Root-cause bug — `_find_image` double-stemmed Roboflow names (confirmed by local repro, `formats.py`).** `_find_image(root, fname, cache)` did `stem = Path(fname).stem`, but YOLO/XML callers already pass a stem (`txt_path.stem`). Roboflow exports name files `foo_jpg.rf.<HASH>.jpg`, so the true stem `foo_jpg.rf.<HASH>` (what the image cache is keyed by) got re-stripped to `foo_jpg.rf` → cache miss on **every** image → parser returns 0 → pipeline folder-falls-back and reads the *directory* name (`images`) as the label. That's exactly why `Ships_Vessels_Aerial` reported `13,435 anns → 0 mapped, UNMAPPED 'images'`. COCO datasets (FLIR) pass full filenames (`x.jpg`), so `Path().stem` was correct there — which is why only the Roboflow YOLO sets broke. Fix: `_find_image` now tries the name **as-given** first, then its stem. Regression test: `smoke_test.py::ingestion.roboflow_stem_lookup`. Verified with a Pillow venv repro (dotted-name YOLO tree: 0→7 anns) and a full `IngestPipeline` dry-run (Ships_Vessels_Aerial 9/9 mapped).

**Enhancement — YOLO now reads each dataset's own `data.yaml` names (`pipeline.py` `_yolo_names_near`).** The rglob-`labels` fallback + autodetect step 4 now (a) pair each `labels/` dir with its sibling `images/` dir, and (b) read class names from the nearest `data.yaml`/`data.yml`/`dataset.yaml` instead of a single hard-coded `yolo_classes`. This makes `Thermal_Ships` work even though it bundles **three** YOLO sub-datasets with *different class orders* (`massmind_yolo` = `[vessel,person]`, `IR boats.yolov11` = `[person,boat,vessel]`, …) — each resolves its own index→name. Verified: index 0 in one and index 2 in another both map to `vessel`.

**Added, working (config in `_KNOWN_DATASETS` + maps in `label_maps.yaml`):**
- `SARScope_Maritime` — Roboflow YOLO under `SaRscope/{split}/{images,labels}`, `data.yaml` `['background','ship']`; ship→naval size-rule, background→null. thermal:false (SAR, not IR).
- `Thermal_Ships` — 3 IR/thermal YOLO sub-datasets; vessel/ship/boat(+plurals)→naval size-rule, person→null. thermal:true (one of the few *genuine-IR* naval sources — high value).
- `Ships_Vessels_Aerial` — no config change needed; the `_find_image` fix alone unblocked it (9,697+2,165+1,573 imgs now reachable).

**Second pass (2026-07-06, same day — user asked to finish SWIM + Ships_Satellite and switch the HRSC mirror):**
- `SWIM` — **fixed.** VOC layout with **rotated boxes** (`<robndbox><cx><cy><w><h><angle>`). `parse_xml` now takes the axis-aligned envelope of the rotated rect (half_w = |w/2·cosθ| + |h/2·sinθ|, etc.; angle in radians). `_KNOWN_DATASETS["SWIM"]` switched yolo→xml with explicit `Annotations`/`JPEGImages` paths so the same-shaped `Landmarks/` XML dir is *not* picked up (verified: a decoy Landmarks label doesn't leak). wake/ship → naval size-rule. Regression: `smoke_test.py::ingestion.robndbox_xml`.
- `Ships_Satellite` — **fixed.** Not detection: 80×80 tiles named `<label>__<scene>__<coords>.png` (1=ship / 0=no-ship). New `parse_filename_prefix()` parser + `format: filename_prefix` config (`img_root: shipsnet` so the unlabeled `scenes/` folder is skipped). `"1"`→Frigate (generic naval — no size/type signal on an 80×80 tile), `"0"`→null. Low value (tiny optical, civilian). Regression: `smoke_test.py::ingestion.filename_prefix`.
- `HRSC2016` — **mirror swapped, parser still pending a re-probe.** Made `weiming97/hrsc2016-ms-dataset` the PRIMARY in the notebook (guofeng demoted to fallback) because the guofeng mirror is a messy multi-part archive (`hrsc2016_dataset/` + `.part01..05/`, real XML buried below the depth-3 probe). Still need the user to re-run the diagnostic on the weiming97 path to confirm its `Annotations`/`AllImages` layout and wire `_KNOWN_DATASETS["HRSC2016"]` — the current config still assumes `Train/Annotations`+`Test/AllImages`. HRSC XML shape (`<HRSC_Image>`/`<HRSC_Object><Class_ID>`, `.bmp`) is already handled by `parse_xml`.

**Still deferred / dead-ends:**
- `Aerial_Segmentation` — semantic-seg land-cover only; `classes.json` = Water/Land/Road/Building/Vegetation — **no military/vehicle/ship classes at all**, so the `vehicle/airplane/ship` map entry can never match. Dead end; leave at 0 (harmless).
- `Battle_Tank_UAV` substitute (`/kaggle/input/notebooks/awaisalisaduzai/tank-detection-vit`) — **not a dataset**: the mount holds only `__results___files/*.png` (rendered notebook figures) and `__output__.json` (a kernel stderr log). Per the user, keep it as a **visual comparison reference only**, not a training source — so no parser/label-map for it; it stays `skipping` (0 patches). Its sibling error `Aerial_Vehicle_Detection` (malformed doubled-prefix path) is still awaiting a real source from the user.

Naval reality check: even with SARScope + Thermal_Ships + Ships_Vessels_Aerial + SWIM + Ships_Satellite, real data still only feeds the naval size-rule classes (Destroyer/Frigate/Corvette/FastAttack) and a couple GROUND ones — the specific jets/tanks remain synthetic-only.

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

This section used to list literal `/kaggle/input/...` mount paths (as of 2026-06-28, snapshot below). Superseded by the 2026-07-06 switch to native Kaggle mounts, which also changed the *path convention itself* (old snapshot below used `/kaggle/input/<slug>/`; the current code uses `/kaggle/input/datasets/<owner>/<slug>/`, confirmed from the user's actual attached inputs — Kaggle's mount path apparently depends on how/when a dataset was attached, so don't assume either form without checking). Current source of truth is CLAUDE.md's "Kaggle notebook workflow" dataset table (21 keys, includes 2 fallback mirrors, the 5 datasets added 2026-07-05, and the restored `Airbus_Aircraft` key) — don't hand-maintain a second copy of that list here.

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
- **Log file:** `/kaggle/working/streamlit.log`
- **Security:** the ngrok token goes in **Kaggle Secrets** (Add-ons → Secrets), not notebook cells — no Kaggle API token needed at runtime since datasets are mounted, not downloaded

---

## File structure (key files only)

```
REATS/
  config/
    targets.yaml                    43-class taxonomy + operational_policy
  ingestion/
    formats.py                      parse_coco / parse_yolo / parse_xml / parse_csv / parse_folder / parse_video_folder
                                     + _iter_class_leaf_dirs (wrapper-dir descent)
    preprocessor.py                 process_annotation() — handles _frame_idx for video;
                                     load_frame/to_ir_look/save_frame/write_yolo_labels for
                                     the YOLO-format detection writer (2026-07-08)
    pipeline.py                     IngestPipeline / load_dataset_annotations / _collect_by_class
                                     / run_detection() (YOLO-format split for Module A, resume-safe)
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
    00_baseline.ipynb               local training walkthrough (native Kaggle)
    01_kaggle_full_pipeline.ipynb   Kaggle GPU full pipeline
  requirements.txt
  verify_env.py
  smoke_test.py                     25 checks, no GPU/data required
  metrics_report.py                 accuracy/ECE/FAR-MR/mAP/latency vs paper targets
```
