"""Broad-action classifier, cohort flags and outcome labels (taxonomy 1.0.0).

Null flags are never treated as false: a missing critical flag yields ``ambiguous_intent``.
"""

from __future__ import annotations

import polars as pl

EXCLUDED_PLAY_TYPES = ("kickoff", "punt", "field_goal", "extra_point", "qb_kneel", "qb_spike", "no_play")

ACTION_RULE_ID = "broad_action_v1"


def _flag(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64)


def action_expr() -> tuple[pl.Expr, pl.Expr]:
    """Return (action, action_reason) expressions over raw PBP columns."""
    pt = pl.col("play_type")
    excluded = (
        pt.is_null()
        | pt.is_in(EXCLUDED_PLAY_TYPES)
        | (_flag("two_point_attempt") == 1)
        | (_flag("special") == 1)
        | pl.col("down").is_null()
    )
    critical_missing = (
        _flag("qb_dropback").is_null()
        | _flag("rush_attempt").is_null()
        | _flag("qb_scramble").is_null()
        | _flag("sack").is_null()
    )
    aborted = _flag("aborted_play") == 1
    dropback_ev = (_flag("qb_dropback") == 1) | (_flag("sack") == 1) | (_flag("qb_scramble") == 1)
    rush_ev = (_flag("rush_attempt") == 1) & (_flag("qb_dropback") == 0) & (_flag("qb_scramble") == 0)
    contradictory = (_flag("rush_attempt") == 1) & (_flag("qb_dropback") == 1) & (_flag("qb_scramble") == 0)

    action = (
        pl.when(excluded).then(None)
        .when(critical_missing | aborted | contradictory).then(None)
        .when(dropback_ev).then(pl.lit("dropback"))
        .when(rush_ev).then(pl.lit("designed_rush"))
        .otherwise(None)
    )
    reason = (
        pl.when(excluded).then(pl.lit("excluded_play_type"))
        .when(critical_missing | aborted | contradictory).then(pl.lit("ambiguous_intent"))
        .when(dropback_ev | rush_ev).then(None)
        .otherwise(pl.lit("unsupported_intent"))
    )
    return action.cast(pl.Utf8), reason.cast(pl.Utf8)


def outcome_exprs() -> list[pl.Expr]:
    """Canonical outcome labels. Unconditional within action; masks keep missing as null."""
    action = pl.col("action")
    epa = pl.col("epa").cast(pl.Float64)
    yards = pl.col("yards_gained").cast(pl.Float64)
    is_rush = action == "designed_rush"
    return [
        epa.alias("y_epa"),
        pl.when(epa.is_null()).then(None).otherwise(epa > 0).alias("y_success"),
        yards.alias("y_yards"),
        pl.when(action.is_null() | yards.is_null()).then(None)
        .when(is_rush).then(yards >= 10)
        .otherwise(yards >= 20).alias("y_explosive"),
        # Structural zero for a verified designed rush; missing stays missing.
        pl.when(action.is_null()).then(None)
        .when(is_rush).then(pl.lit(False))
        .otherwise(_flag("sack") == 1).alias("y_sack"),
        pl.when(_flag("interception").is_null() | _flag("fumble_lost").is_null()).then(None)
        .otherwise((_flag("interception") == 1) | (_flag("fumble_lost") == 1)).alias("y_turnover"),
        pl.when(_flag("first_down").is_null()).then(None)
        .otherwise(
            (_flag("first_down") == 1)
            | ((_flag("touchdown") == 1) & (pl.col("td_team") == pl.col("posteam")))
        ).alias("y_first_down_or_td"),
        pl.when(_flag("fourth_down_failed").is_null()).then(None)
        .otherwise(_flag("fourth_down_failed") == 1).alias("y_failed_fourth_down"),
    ]


def cohort_exprs() -> list[pl.Expr]:
    """Scrimmage eligibility and the primary penalty-free outcome cohort flag."""
    base = (
        (pl.col("season_type") == "REG")
        & pl.col("action").is_not_null()
        & pl.col("down").is_not_null()
        & pl.col("offense_id").is_not_null()
        & pl.col("defense_id").is_not_null()
    )
    penalty_free = base & (_flag("penalty") == 0) & pl.col("y_epa").is_not_null()
    return [
        base.alias("is_scrimmage_eligible"),
        penalty_free.alias("in_reg_scrimmage_penalty_free_v1"),
    ]
