"""Analyst-mode defensive profile: observed conditional tendencies with denominators, game-clustered
intervals and smoothed prior components, all as-of a cutoff."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from coordinator.features.builder import FeatureBuilder
from coordinator.features.context import make_contexts
from coordinator.features.frame import SITUATION_CELLS
from coordinator.taxonomy.vocab import BOX_BUCKETS, COVERAGE_CLASSES
from coordinator.training.metrics import cluster_bootstrap_mean

MIN_DENOMINATOR_FOR_RATE = 10
MIN_GAMES_FOR_INTERVAL = 5


def _known(frame: pl.DataFrame, source: str, cutoff: datetime) -> pl.Expr:
    return pl.col(f"known_at_{source}").is_not_null() & (pl.col(f"known_at_{source}") <= cutoff)


def _filter_situation(df: pl.DataFrame, down: int | None, distance_band: str | None, field_zone: str | None,
                      half: int | None) -> pl.DataFrame:
    bands = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
    zones = {"goal_line": 0, "red_zone": 1, "open_field": 2, "backed_up": 3}
    if down is not None:
        df = df.filter(pl.col("down") == down)
    if distance_band is not None:
        df = df.filter(pl.col("band_idx") == bands[distance_band])
    if field_zone is not None:
        df = df.filter(pl.col("zone_idx") == zones[field_zone])
    if half is not None:
        df = df.filter(pl.col("quarter").is_in([1, 2] if half == 1 else [3, 4, 5]))
    return df


def _rate_row(label: str, df: pl.DataFrame, value_col: str, source: str, denominator_desc: str,
              kind: str = "rate", reps: int = 400) -> dict:
    labeled = df.filter(pl.col(value_col).is_not_null())
    den = labeled.height
    row = {"label": label, "kind": kind, "source": source, "provenance": "observed",
           "denominator": den, "denominator_description": denominator_desc,
           "missing_label_count": df.height - den, "numerator": None, "value": None,
           "interval": None, "interval_method": None, "suppressed": den < MIN_DENOMINATOR_FOR_RATE}
    if den == 0:
        return row
    vals = labeled[value_col].cast(pl.Float64).to_numpy()
    if kind == "rate":
        row["numerator"] = int(vals.sum())
    row["distinct_games"] = labeled["game_id"].n_unique()
    if not row["suppressed"]:
        row["value"] = float(vals.mean())
        if row["distinct_games"] >= MIN_GAMES_FOR_INTERVAL:
            ci = cluster_bootstrap_mean(vals, labeled["game_id"].to_numpy(), reps=reps)
            row["interval"] = [ci["ci_low"], ci["ci_high"]]
            row["interval_method"] = "game_clustered_bootstrap_95"
        else:
            row["interval_method"] = f"not_estimated_fewer_than_{MIN_GAMES_FOR_INTERVAL}_games"
    return row


def observed_tendencies(frame: pl.DataFrame, defense: str, season: int, cutoff: datetime,
                        down=None, distance_band=None, field_zone=None, half=None) -> dict:
    base = frame.filter((pl.col("season") == season) & (pl.col("completed_at_est") < cutoff))
    base = _filter_situation(base, down, distance_band, field_zone, half)
    pbp = base.filter(_known(base, "pbp", cutoff))
    d = pbp.filter(pl.col("defense_id") == defense)
    league = pbp
    ftn_d = d.filter(_known(d, "ftn", cutoff))
    part_d = d.filter(_known(d, "part", cutoff) & (pl.col("participation_era") == "ftn"))
    cohort = d.filter(pl.col("in_reg_scrimmage_penalty_free_v1"))
    lcohort = league.filter(pl.col("in_reg_scrimmage_penalty_free_v1"))
    d.filter(pl.col("is_dropback"))

    rows = [
        _rate_row("Opponent dropback rate", d.with_columns(pl.col("is_dropback").alias("v")), "v", "pbp", "eligible scrimmage plays faced"),
    ]
    for act, name in (("dropback", "Dropback"), ("designed_rush", "Designed rush")):
        c = cohort.filter(pl.col("action") == act)
        lc = lcohort.filter(pl.col("action") == act)
        rows.append(_rate_row(f"EPA/play allowed: {name}", c, "y_epa", "pbp", f"penalty-free {name.lower()} plays", kind="mean"))
        rows.append(_rate_row(f"League EPA/play: {name}", lc, "y_epa", "pbp", f"league penalty-free {name.lower()} plays", kind="mean"))
        rows.append(_rate_row(f"Success allowed: {name}", c, "y_success", "pbp", f"penalty-free {name.lower()} plays"))
        rows.append(_rate_row(f"Explosive allowed: {name}", c, "y_explosive", "pbp", f"penalty-free {name.lower()} plays"))
    rows.append(_rate_row("Sack rate", cohort.filter(pl.col("action") == "dropback"), "y_sack", "pbp", "penalty-free dropbacks"))
    rows.append(_rate_row("Takeaway rate", cohort, "y_turnover", "pbp", "penalty-free plays"))
    rows.append(_rate_row("Mean box count", ftn_d, "box_count", "ftn_charting", "charted eligible plays", kind="mean"))
    rows.append(_rate_row("Heavy box (8+)", ftn_d.with_columns((pl.col("box_count") >= 8).alias("v")), "v", "ftn_charting", "charted eligible plays"))
    rows.append(_rate_row("Five or more pass rushers", ftn_d.filter(pl.col("is_dropback")), "t_five_plus", "ftn_charting", "charted dropbacks"))
    rows.append(_rate_row("Blitz (provider blitzer count >= 1)", ftn_d.filter(pl.col("is_dropback")), "t_blitz", "ftn_charting", "charted dropbacks"))
    rows.append(_rate_row("Man coverage", part_d.filter(pl.col("is_dropback")), "t_man", "participation", "charted dropbacks"))
    rows.append(_rate_row("Pressure", part_d.filter(pl.col("is_dropback")), "t_pressure", "participation", "charted dropbacks"))

    cov = part_d.filter(pl.col("t_coverage").is_not_null())
    cov_rows = []
    for i, c in enumerate(COVERAGE_CLASSES):
        cov_rows.append({"coverage": c, "count": int((cov["t_coverage"] == i).sum())})
    unavailable = []
    if ftn_d.height == 0 and d.height > 0:
        unavailable.append({"field": "ftn_charting", "reason": "not_yet_available"})
    if part_d.height == 0:
        unavailable.append({"field": "coverage", "reason": "COVERAGE_LABELS_UNAVAILABLE",
                            "detail": "Participation labels are published after the season; none are available for this season at this cutoff."})
    dates = d["completed_at_est"]
    return {
        "season": season, "defense": defense,
        "filters": {"down": down, "distance_band": distance_band, "field_zone": field_zone, "half": half},
        "eligible_plays": d.height, "distinct_games": d["game_id"].n_unique(),
        "date_range": [dates.min().isoformat() if d.height else None, dates.max().isoformat() if d.height else None],
        "tendencies": rows,
        "coverage_distribution": {"denominator": cov.height, "rows": cov_rows if cov.height else [],
                                  "missing_label_count": part_d.filter(pl.col("is_dropback")).height - cov.height},
        "unavailable": unavailable,
        "min_denominator_for_rate": MIN_DENOMINATOR_FOR_RATE,
    }


def smoothed_estimates(builder: FeatureBuilder, defense: str, season: int, cutoff: datetime) -> dict:
    ctx = make_contexts([(defense, defense, season, cutoff)])
    cf = builder.context_features(ctx)
    out = {}
    names = {"box": list(BOX_BUCKETS), "five_plus": ["no", "yes"], "blitz": ["no", "yes"],
             "coverage": list(COVERAGE_CLASSES), "faced_dropback": ["rush", "dropback"], "pressure": ["no", "yes"]}
    for h, labels in names.items():
        comp = cf.components[h]
        sup = cf.support[h][0]
        out[h] = {
            "classes": labels,
            "estimate": cf.priors[h][0, SITUATION_CELLS].round(4).tolist(),
            "league_component": comp["league"][0].round(4).tolist(),
            "team_history_component": comp["team"][0].round(4).tolist(),
            "prior": comp["prior"][0].round(4).tolist(),
            "current_season_counts": comp["current_counts"][0].round(2).tolist(),
            "support": {"raw_current_count_N": float(sup[0]), "weighted_mass_W": float(sup[1]),
                        "kish_n_eff": float(sup[2]), "weighted_history_team": float(sup[3])},
            "provenance": "predicted",
        }
    return {
        "method": builder.prior_params.to_dict(),
        "coordinator_component": "unavailable",
        "playcaller_attribution_status": "unknown",
        "statement": "Estimate combines league and team history with current-season observations; "
                     "no verified playcaller registry is loaded, so no coordinator component is used.",
        "heads": out,
    }
