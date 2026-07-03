"""
Battlefield threat analysis metrics — False Alarm Rate (FAR) and Miss Rate (MR).

Standard precision/recall averages hide the two failure modes that matter operationally:

  FAR (False Alarm Rate) = FP / (FP + TP) = 1 - Precision
      The rate at which the dashboard raises a threat alarm on a target that is not
      actually that class — e.g. a civilian aircraft or friendly vessel misclassified
      as an enemy platform. Costly because it erodes operator trust and can trigger an
      unnecessary track/engagement recommendation.

  MR (Miss Rate / Omission Rate) = FN / (FN + TP) = 1 - Recall
      The rate at which an actual instance of the class is missed entirely — the
      critical failure mode for RED-threat classes (e.g. an infiltrating enemy fighter
      that never raises an alarm).

RED-threat classes (fighters, bombers, attack helicopters, armed UAVs/vessels) are
weighted separately from the macro average because a missed RED target is the highest-
consequence failure the system can make.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_reats_root = str(Path(__file__).parent.parent)
if _reats_root not in sys.path:
    sys.path.insert(0, _reats_root)

from config import CLASSES, RED_THREATS, ORANGE_THREATS, YELLOW_THREATS


def compute_far_mr(
    all_labels: List[int],
    all_preds: List[int],
    classes: List[str] = CLASSES,
) -> Dict:
    """Per-class + aggregate FAR/MR from label/prediction index lists.

    Returns a dict with:
      per_class[cls] = {FAR, MR, TP, FP, FN}
      macro_FAR, macro_MR                — mean over all classes
      red_threat_FAR, red_threat_MR      — mean restricted to RED_THREATS classes
                                            (the highest-consequence targets)
    """
    from sklearn.metrics import confusion_matrix

    n = len(classes)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n))).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    far = np.divide(fp, fp + tp, out=np.zeros_like(fp), where=(fp + tp) > 0)
    mr  = np.divide(fn, fn + tp, out=np.zeros_like(fn), where=(fn + tp) > 0)

    per_class: Dict[str, Dict] = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "FAR": float(far[i]),
            "MR":  float(mr[i]),
            "TP":  int(tp[i]),
            "FP":  int(fp[i]),
            "FN":  int(fn[i]),
        }

    red_idx = [i for i, c in enumerate(classes) if c in RED_THREATS]

    return {
        "per_class":      per_class,
        "macro_FAR":      float(far.mean()) if n else float("nan"),
        "macro_MR":       float(mr.mean()) if n else float("nan"),
        "red_threat_FAR": float(far[red_idx].mean()) if red_idx else float("nan"),
        "red_threat_MR":  float(mr[red_idx].mean()) if red_idx else float("nan"),
        "n_red_threats":  len(red_idx),
    }


def far_mr_from_model(model, loader, device: str) -> Dict:
    """Run `model` over `loader` and compute FAR/MR. Works for both raw classifiers
    (logits) and EnsembleClassifier (probabilities) — argmax is softmax-invariant,
    so no normalisation is needed either way."""
    import torch

    model.eval()
    all_labels: List[int] = []
    all_preds:  List[int] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return compute_far_mr(all_labels, all_preds)


def format_report(report: Dict, top_n: Optional[int] = None) -> str:
    """Human-readable FAR/MR table, RED-threat classes flagged, worst offenders first."""
    lines = [
        f"{'Class':<14}{'Threat':<8}{'FAR':>8}{'MR':>8}{'TP':>6}{'FP':>6}{'FN':>6}",
        "-" * 56,
    ]
    rows = list(report["per_class"].items())
    rows.sort(key=lambda kv: kv[1]["MR"], reverse=True)
    if top_n:
        rows = rows[:top_n]
    for cls, m in rows:
        lvl = "RED" if cls in RED_THREATS else "ORANGE" if cls in ORANGE_THREATS else "YELLOW"
        lines.append(
            f"{cls:<14}{lvl:<8}{m['FAR']:>8.3f}{m['MR']:>8.3f}{m['TP']:>6}{m['FP']:>6}{m['FN']:>6}"
        )
    lines.append("-" * 56)
    lines.append(f"{'macro avg':<22}{report['macro_FAR']:>8.3f}{report['macro_MR']:>8.3f}")
    lines.append(
        f"{'RED-threat avg':<22}{report['red_threat_FAR']:>8.3f}{report['red_threat_MR']:>8.3f}"
        f"   ({report['n_red_threats']} RED classes — missed-fighter risk)"
    )
    return "\n".join(lines)
