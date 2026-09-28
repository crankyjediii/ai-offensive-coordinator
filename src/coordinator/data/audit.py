"""Source audit: coverage, uniqueness, joins, null proportions and quarantine counts.

Every proportion is published with its denominator.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from coordinator.data.ingest import existing_assets

LABEL_FIELDS = {
    "charting_observation": ["box_count", "pass_rusher_count", "blitzer_count", "hash", "qb_alignment", "motion_observed"],
    "participation_observation": ["coverage", "man_zone", "pressure_observed", "off_te"],
}


def _prop(num: int, den: int) -> dict:
    return {"numerator": num, "denominator": den, "proportion": round(num / den, 4) if den else None}


def run_audit(
    games: pl.DataFrame,
    plays: pl.DataFrame,
    charting: pl.DataFrame,
    participation: pl.DataFrame,
    requested_seasons: list[int],
    cutoff=None,
    charting_latency_hours: float = 48.0,
) -> dict:
    """Charting completeness is measured on games completed at least the documented charting latency
    before the cutoff; plays from more recent games are reported as pending, not as missing."""
    report: dict = {"requested_seasons": requested_seasons, "datasets": {}, "seasons": {}, "gates": {}}
    for ds in ("pbp", "ftn_charting", "participation", "schedules"):
        present = sorted({a.season for a in existing_assets(ds) if a.season is not None})
        report["datasets"][ds] = {
            "returned_seasons": present,
            "missing_seasons": [s for s in requested_seasons if ds != "schedules" and s not in present],
        }

    dup = {
        "play": plays.select("game_id", "play_id", "record_version_id").is_duplicated().sum(),
        "charting_observation": charting.select("game_id", "play_id", "record_version_id").is_duplicated().sum(),
        "participation_observation": participation.select("game_id", "play_id", "record_version_id").is_duplicated().sum(),
        "game": games.select("game_id", "record_version_id").is_duplicated().sum(),
    }
    report["duplicate_keys"] = {k: int(v) for k, v in dup.items()}

    elig = plays.filter(pl.col("is_scrimmage_eligible"))
    ch = elig.join(charting.select("game_id", "play_id", *LABEL_FIELDS["charting_observation"]),
                   on=["game_id", "play_id"], how="left")
    pa = elig.join(participation.select("game_id", "play_id", "participation_era", *LABEL_FIELDS["participation_observation"]),
                   on=["game_id", "play_id"], how="left")
    impossible = plays.filter(
        pl.col("is_scrimmage_eligible")
        & (~pl.col("down").is_between(1, 4) | (pl.col("distance") <= 0) | ~pl.col("yards_to_goal").is_between(1, 99)
           | (pl.col("y_epa").is_not_null() & ~pl.col("y_epa").is_finite()))
    )
    report["impossible_state_rows"] = impossible.height
    report["quarantined_rusher_counts"] = int(charting["rusher_count_quarantined"].sum())
    report["unmapped_coverage_values"] = int((participation["coverage"] == "UNMAPPED").sum())

    for season in requested_seasons:
        g = games.filter((pl.col("season") == season) & pl.col("completed"))
        pbp_games = plays.filter(pl.col("season") == season)["game_id"].n_unique()
        e = elig.filter(pl.col("season") == season)
        c = ch.filter(pl.col("season") == season)
        p = pa.filter(pl.col("season") == season)
        n = e.height
        dropbacks = e.filter(pl.col("action") == "dropback").height
        entry = {
            "completed_games_expected": g.height,
            "games_with_pbp": pbp_games,
            "game_coverage": _prop(min(pbp_games, g.height), g.height),
            "eligible_scrimmage_plays": n,
            "action_counts": dict(e.group_by("action").len().iter_rows()),
            "penalty_free_outcome_cohort": _prop(int(e["in_reg_scrimmage_penalty_free_v1"].sum()), n),
            "field_completeness": {},
        }
        if cutoff is not None:
            settled = pl.col("completed_at_est") <= cutoff - timedelta(hours=charting_latency_hours)
            entry["charting_pending_recent_plays"] = c.filter(~settled).height
            c = c.filter(settled)
        c_db = c.filter(pl.col("action") == "dropback").height
        for f in LABEL_FIELDS["charting_observation"]:
            den = c_db if f in ("pass_rusher_count", "blitzer_count") else c.height
            entry["field_completeness"][f] = _prop(int(c[f].is_not_null().sum()), den)
        for f in LABEL_FIELDS["participation_observation"]:
            den = dropbacks if f in ("coverage", "man_zone", "pressure_observed") else n
            entry["field_completeness"][f] = _prop(int(p[f].is_not_null().sum()), den)
        era = p["participation_era"].drop_nulls().unique().to_list()
        entry["participation_era"] = era
        report["seasons"][str(season)] = entry

    report["gates"] = {
        "unique_keys": all(v == 0 for v in report["duplicate_keys"].values()),
        "no_impossible_states": report["impossible_state_rows"] == 0,
        "no_unmapped_categories": report["unmapped_coverage_values"] == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report
