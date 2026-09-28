"""As-of eligibility: which record versions a prediction context may see."""

from __future__ import annotations

from datetime import datetime

import polars as pl

TEMPORAL_MODES = ("archived_as_of", "reconstructed_conservative_delay")


def known_at_col(mode: str) -> str:
    if mode == "archived_as_of":
        return "known_at_archived"
    if mode == "reconstructed_conservative_delay":
        return "known_at_reconstructed"
    raise ValueError(f"unknown temporal mode {mode!r}")


def eligible_versions(
    df: pl.DataFrame | pl.LazyFrame,
    cutoff_at: datetime,
    mode: str,
    key: list[str],
    game_completed_col: str | None = "completed_at_est",
) -> pl.LazyFrame:
    """Latest eligible version per key.

    A row is eligible when its known-at time is at or before the cutoff and, when the row belongs
    to a game, that game completed before the cutoff. Ties break on the stable version ID.
    """
    lf = df.lazy()
    cond = pl.col(known_at_col(mode)).is_not_null() & (pl.col(known_at_col(mode)) <= cutoff_at)
    if game_completed_col is not None:
        cond = cond & pl.col(game_completed_col).is_not_null() & (pl.col(game_completed_col) < cutoff_at)
    return (
        lf.filter(cond)
        .sort([known_at_col(mode), "record_version_id"])
        .unique(subset=key, keep="last", maintain_order=True)
    )
