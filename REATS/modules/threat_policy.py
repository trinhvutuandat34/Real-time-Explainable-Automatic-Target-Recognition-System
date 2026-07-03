"""
Confidence-threshold → CMS operational policy mapping (Warning / Track / Engagement).

Until now the dashboard only showed a RED/ORANGE/YELLOW color derived from a class's
static metadata in targets.yaml. That tells an operator what kind of target it might
be, but not what to *do* about it. This module adds the tactical mapping the professor
asked for: classifier confidence + threat_level → one of the CMS code-of-conduct tiers
that actually appear in a combat management system's rules of engagement:

  NONE       — below the detection confidence floor; no operator action
  WARNING    — operator alert only: log + display, no sensor/weapon tasking
  TRACK      — cue sensors and maintain track; no engagement authority
  ENGAGEMENT — engagement-eligible; requires operator confirmation per ROE

Engagement authority is reserved for RED-threat classes: a false engagement
recommendation on a misclassified civilian/friendly target is the costliest failure
mode, so ORANGE/YELLOW targets are capped at TRACK/WARNING respectively regardless of
confidence. Thresholds live in config/targets.yaml (`operational_policy`), not here —
single source of truth, same as the class taxonomy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_reats_root = str(Path(__file__).parent.parent)
if _reats_root not in sys.path:
    sys.path.insert(0, _reats_root)

from config import OPERATIONAL_POLICY

POLICY_ORDER = ["NONE", "WARNING", "TRACK", "ENGAGEMENT"]

_DEFAULT_THRESHOLDS = {"WARNING": 0.50, "TRACK": 0.75, "ENGAGEMENT": 0.90}
_DEFAULT_CEILING = {"RED": "ENGAGEMENT", "ORANGE": "TRACK", "YELLOW": "WARNING"}

CONFIDENCE_THRESHOLDS: dict = OPERATIONAL_POLICY.get("confidence_thresholds", _DEFAULT_THRESHOLDS)
THREAT_LEVEL_CEILING: dict = OPERATIONAL_POLICY.get("threat_level_ceiling", _DEFAULT_CEILING)

POLICY_DESCRIPTION = {
    "NONE":       "Below detection confidence floor — no operator action.",
    "WARNING":    "Operator alert only — log and display, no sensor/weapon tasking.",
    "TRACK":      "Cue sensors and maintain track — no engagement authority.",
    "ENGAGEMENT": "Engagement-eligible — requires operator confirmation per ROE.",
}


def map_confidence_to_policy(confidence: float, threat_level: str) -> str:
    """Map a classifier confidence + target threat_level to a CMS policy tier.

    The confidence thresholds set the tier a detection would earn on its own; the
    threat_level ceiling then caps it, so an ORANGE/YELLOW target can never reach
    ENGAGEMENT no matter how confident the classifier is.
    """
    tier = "NONE"
    for name in ("WARNING", "TRACK", "ENGAGEMENT"):
        if confidence >= CONFIDENCE_THRESHOLDS.get(name, _DEFAULT_THRESHOLDS[name]):
            tier = name

    ceiling = THREAT_LEVEL_CEILING.get(threat_level, "WARNING")
    if POLICY_ORDER.index(tier) > POLICY_ORDER.index(ceiling):
        tier = ceiling
    return tier
