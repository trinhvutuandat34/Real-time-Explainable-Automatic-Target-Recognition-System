"""
REATS target taxonomy — single source of truth loaded from targets.yaml.

Exported symbols:
    CLASSES          list[str]         ordered class IDs (e.g. ["F16", "LYNX", …])
    NUM_CLASSES      int               len(CLASSES)
    TARGET_META      dict[str, dict]   full metadata per class id
    THREAT_COLOR_BGR dict[str, tuple]  BGR colour for bounding-box overlays
    RED_THREATS      set[str]
    ORANGE_THREATS   set[str]
    YELLOW_THREATS   set[str]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_YAML = Path(__file__).parent / "targets.yaml"


def _load() -> tuple:
    with open(_YAML) as f:
        raw = yaml.safe_load(f)
    entries = raw["classes"]
    classes: list[str] = [c["id"] for c in entries]
    meta:    dict       = {c["id"]: c for c in entries}
    colors:  dict       = {c["id"]: tuple(c["color_bgr"]) for c in entries}
    sets:    dict       = {}
    for c in entries:
        lvl = c.get("threat_level", "YELLOW")
        sets.setdefault(lvl, set()).add(c["id"])
    return classes, meta, colors, sets


CLASSES, TARGET_META, THREAT_COLOR_BGR, _THREAT_SETS = _load()

NUM_CLASSES:    int       = len(CLASSES)
RED_THREATS:    set[str] = _THREAT_SETS.get("RED", set())
ORANGE_THREATS: set[str] = _THREAT_SETS.get("ORANGE", set())
YELLOW_THREATS: set[str] = _THREAT_SETS.get("YELLOW", set())


def _ensure_reats_on_path() -> None:
    """Add REATS root to sys.path so submodules can import this package."""
    reats_root = str(Path(__file__).parent.parent)
    if reats_root not in sys.path:
        sys.path.insert(0, reats_root)
