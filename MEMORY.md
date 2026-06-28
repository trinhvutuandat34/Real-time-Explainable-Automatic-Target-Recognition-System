# MEMORY.md — REATS Session State

Last updated: 2026-06-28
Active branch: `claude/clever-goodall-fsmqnr`

---

## What this file is for

Running log of architectural decisions, bug fixes, and session context so that future Claude Code sessions can resume without re-deriving the same information.

---

## Current project state

The Kaggle notebook (`notebooks/01_kaggle_full_pipeline.ipynb`) is the primary execution environment. All REATS modules (A–D) are implemented and the dashboard is functional. Training (Module B, 6 × ConvNeXt-tiny ensemble) still requires a GPU run on Kaggle.

### What is working
- Module A: `IRDetector` YOLOv4 pure-PyTorch, forward pass + NMS
- Module B: ConvNeXt-tiny × 6 ensemble, AMP + EMA, TemperatureScaler calibration
- Module C: GradCAM / GradCAM++ / EigenCAM, SHAP, LIME, MCDropout, faithfulness AUC
- Module D: Streamlit dashboard — Upload and Batch Processing tabs fully functional
- Ingestion pipeline: 16 dataset keys, all formats (COCO JSON, YOLO txt, XML, CSV, Folder, Video folder)

### What is pending
- GPU training run on Kaggle (requires T4/P100 session, ~18–24 hours for 6 models)
- iPhone Live tab: requires second tunnel (Cloudflare) when phone is not on same WiFi as Kaggle

---

## Bug fixes applied this session (in order)

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

## Kaggle dataset paths (as of 2026-06-28)

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

---

## REATS 43-class taxonomy (from `config/targets.yaml`)

| Domain | Classes |
|--------|---------|
| AIR (25) | F16, F18, F22, F35, Su27, Su35, MiG21, MiG29, B52, B2, AH64, UH60, CH47, MQ9, RQ4, Reaper, GlobalHawk, Predator, Eurofighter, Rafale, Tornado, Gripen, LYNX, MiG19, PKG |
| GROUND (13) | T72, Abrams, Leopard2, Challenger2, BMP2, BTR80, M109, M270, Patriot, HIMARs, Humvee, JLTV, ZSU23 |
| NAVAL (6) | Destroyer, Frigate, Corvette, FastAttack, PTG, LittoralCombat |

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
- **Security:** ngrok tokens go in Kaggle Secrets, not notebook cells

---

## File structure (key files only)

```
REATS/
  config/
    targets.yaml                    43-class taxonomy with domain tags
  ingestion/
    formats.py                      parse_coco / parse_yolo / parse_xml / parse_csv / parse_folder / parse_video_folder
    preprocessor.py                 process_annotation() — handles _frame_idx for video
    pipeline.py                     IngestPipeline / load_dataset_annotations / _collect_by_class
    label_maps.yaml                 label → class mapping per dataset key
  modules/
    module_a_detector.py            IRDetector (YOLOv4, pure PyTorch)
    module_b_classifier.py          ConvNeXtEnsemble, TemperatureScaler
    module_c_xai.py                 GradCAMExplainer*, SHAPExplainer, LIMEExplainer, MCDropoutWrapper
    module_d_dashboard.py           Streamlit 4-tab UI
  notebooks/
    00_baseline.ipynb               local training walkthrough
    01_kaggle_full_pipeline.ipynb   Kaggle A100/T4 full pipeline
  requirements.txt
  verify_env.py
```
