"""Versioned vocabularies (taxonomy 1.0.0). Changing any mapping here requires a version bump."""

from __future__ import annotations

from coordinator.config import TAXONOMY_VERSION

__all__ = [
    "TAXONOMY_VERSION",
    "ACTIONS",
    "COVERAGE_CLASSES",
    "COVERAGE_SOURCE_MAP",
    "MAN_ZONE_CLASSES",
    "ROUTE_MAP",
    "MISSING_REASONS",
    "BOX_BUCKETS",
    "RUSHER_BUCKETS",
    "distance_band",
    "field_zone",
    "box_bucket",
    "rusher_bucket",
]

ACTIONS = ("designed_rush", "dropback")
ACTION_LABELS = {"designed_rush": "Designed rush", "dropback": "Dropback"}

# Coverage is modeled on the FTN-era (2023+) participation vocabulary only. The 2022 NGS-era
# vocabulary differs (PREVENT present; COVER_9/COMBO/BLOWN absent) and its class mix is not
# comparable, so 2022 coverage rows are excluded from the coverage cohort (decision D-004).
COVERAGE_CLASSES = (
    "COVER_0", "COVER_1", "COVER_2", "2_MAN", "COVER_3", "COVER_4", "COVER_6", "OTHER_OBSERVED",
)
# Explicit pooling rule: COVER_9, COMBO, BLOWN and PREVENT are observed values pooled into
# OTHER_OBSERVED for the model vocabulary. Reports keep the raw value.
COVERAGE_SOURCE_MAP = {
    "COVER_0": "COVER_0",
    "COVER_1": "COVER_1",
    "COVER_2": "COVER_2",
    "2_MAN": "2_MAN",
    "COVER_3": "COVER_3",
    "COVER_4": "COVER_4",
    "COVER_6": "COVER_6",
    "COVER_9": "OTHER_OBSERVED",
    "COMBO": "OTHER_OBSERVED",
    "BLOWN": "OTHER_OBSERVED",
    "PREVENT": "OTHER_OBSERVED",
}
MAN_ZONE_CLASSES = ("MAN_COVERAGE", "ZONE_COVERAGE")

ROUTE_MAP = {
    "CORNER": "corner",
    "DEEP OUT": "deep_out",
    "GO": "go",
    "HITCH/CURL": "hitch_curl",
    "IN/DIG": "in_dig",
    "POST": "post",
    "QUICK OUT": "quick_out",
    "SCREEN": "screen",
    "SHALLOW CROSS/DRAG": "shallow_cross_drag",
    "SLANT": "slant",
    "SWING": "swing",
    "TEXAS/ANGLE": "texas_angle",
    "WHEEL": "wheel",
}

MISSING_REASONS = (
    "not_collected", "not_applicable", "not_yet_available", "unmapped", "source_conflict", "unknown",
)

# Bounded count distributions with documented overflow buckets.
BOX_BUCKETS = ("<=4", "5", "6", "7", "8", ">=9")
RUSHER_BUCKETS = ("<=3", "4", "5", "6", ">=7")


def distance_band(ydstogo: int) -> str:
    if ydstogo <= 3:
        return "short"
    if ydstogo <= 6:
        return "medium"
    if ydstogo <= 10:
        return "long"
    return "very_long"


def field_zone(yardline_100: int) -> str:
    if yardline_100 <= 5:
        return "goal_line"
    if yardline_100 <= 20:
        return "red_zone"
    if yardline_100 <= 80:
        return "open_field"
    return "backed_up"


def box_bucket(n: int) -> int:
    return 0 if n <= 4 else min(n - 4, 5)


def rusher_bucket(n: int) -> int:
    return 0 if n <= 3 else min(n - 3, 4)


DISTANCE_BANDS = ("short", "medium", "long", "very_long")
FIELD_ZONES = ("goal_line", "red_zone", "open_field", "backed_up")
