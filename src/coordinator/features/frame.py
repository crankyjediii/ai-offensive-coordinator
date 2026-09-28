"""The modeling play frame: eligible regular-season scrimmage plays with label masks and
per-source known-at times, plus team-game aggregation units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from coordinator.data.snapshot import Snapshot
from coordinator.taxonomy.vocab import COVERAGE_CLASSES, DISTANCE_BANDS, FIELD_ZONES

SITUATION_CELLS = 16  # down (4) x distance band (4)


def _band_idx() -> pl.Expr:
    d = pl.col("distance")
    return pl.when(d <= 3).then(0).when(d <= 6).then(1).when(d <= 10).then(2).otherwise(3)


def _zone_idx() -> pl.Expr:
    y = pl.col("yards_to_goal")
    return pl.when(y <= 5).then(0).when(y <= 20).then(1).when(y <= 80).then(2).otherwise(3)


def load_play_frame(snapshot: Snapshot) -> pl.DataFrame:
    """One row per eligible REG scrimmage play, sorted by (completed_at_est, game_id, play_order)."""
    ka = snapshot.manifest["known_at_column"]
    plays = snapshot.table("play").filter(pl.col("is_scrimmage_eligible"))
    ch = snapshot.table("charting_observation").select(
        "game_id", "play_id", "box_count", "pass_rusher_count", "blitzer_count", "qb_alignment",
        "motion_observed", "play_action_observed", "rpo_observed", pl.col(ka).alias("known_at_ftn"),
    )
    pa = snapshot.table("participation_observation").select(
        "game_id", "play_id", "coverage", "man_zone", "pressure_observed", "off_te", "participation_era",
        pl.col(ka).alias("known_at_part"),
    )
    df = (
        plays.join(ch, on=["game_id", "play_id"], how="left")
        .join(pa, on=["game_id", "play_id"], how="left")
        .with_columns(pl.col(ka).alias("known_at_pbp"))
    )
    ngs = pl.col("participation_era") == "ngs"
    is_db = pl.col("action") == "dropback"
    cov_idx = {c: i for i, c in enumerate(COVERAGE_CLASSES)}
    df = df.with_columns(
        (pl.col("down").cast(pl.Int32) - 1).alias("down_idx"),
        _band_idx().alias("band_idx"),
        _zone_idx().alias("zone_idx"),
        is_db.alias("is_dropback"),
        # Label targets (null = masked).
        pl.when(pl.col("box_count").is_null()).then(None)
        .otherwise(pl.min_horizontal(pl.max_horizontal(pl.col("box_count").cast(pl.Int32) - 4, 0), 5)).alias("t_box"),
        pl.when(pl.col("pass_rusher_count").is_null() | ~is_db).then(None)
        .otherwise(pl.min_horizontal(pl.max_horizontal(pl.col("pass_rusher_count").cast(pl.Int32) - 3, 0), 4)).alias("t_rush"),
        pl.when(pl.col("pass_rusher_count").is_null() | ~is_db).then(None)
        .otherwise(pl.col("pass_rusher_count") >= 5).alias("t_five_plus"),
        pl.when(pl.col("blitzer_count").is_null() | ~is_db).then(None)
        .otherwise(pl.col("blitzer_count") >= 1).alias("t_blitz"),
        pl.when(ngs | pl.col("coverage").is_null() | ~is_db).then(None)
        .otherwise(pl.col("coverage").replace_strict(cov_idx, default=None)).cast(pl.Int32).alias("t_coverage"),
        pl.when(ngs | pl.col("man_zone").is_null() | ~is_db).then(None)
        .otherwise(pl.col("man_zone") == "MAN_COVERAGE").alias("t_man"),
        pl.when(ngs | pl.col("pressure_observed").is_null() | ~is_db).then(None)
        .otherwise(pl.col("pressure_observed")).alias("t_pressure"),
    ).with_columns(
        (pl.col("down_idx") * 4 + pl.col("band_idx")).alias("cell_idx"),
    )
    return df.filter(pl.col("season_type") == "REG").sort(["completed_at_est", "game_id", "play_order"])


@dataclass
class TeamGames:
    """Aggregation units: one row per (game, team) from both the defense and offense perspective."""

    table: pl.DataFrame  # game_id, team, season, completed_at_est, known_at_pbp/ftn/part
    defense: dict[str, np.ndarray]  # arrays shaped (n_units, ...) of defense-perspective sums
    offense: dict[str, np.ndarray]


def _onehot_counts(df: pl.DataFrame, unit_index: np.ndarray, n_units: int, cell: np.ndarray,
                   col: str, n_classes: int) -> np.ndarray:
    """Counts shaped (units, cells+1, classes); cell index SITUATION_CELLS holds the overall total."""
    vals = df[col].cast(pl.Float64).to_numpy()
    mask = ~np.isnan(vals)
    out = np.zeros((n_units, SITUATION_CELLS + 1, n_classes), dtype=np.float64)
    u, c, k = unit_index[mask], cell[mask], vals[mask].astype(int)
    np.add.at(out, (u, c, k), 1.0)
    np.add.at(out, (u, np.full_like(c, SITUATION_CELLS), k), 1.0)
    return out


OUTCOME_COLS = ("y_epa", "y_success", "y_explosive", "y_sack", "y_turnover", "y_first_down_or_td")


def _outcome_sums(df: pl.DataFrame, unit_index: np.ndarray, n_units: int) -> np.ndarray:
    """Penalty-free outcome sums by action: (units, 2 actions, 1 + len(OUTCOME_COLS)) with count first."""
    cohort = df["in_reg_scrimmage_penalty_free_v1"].to_numpy()
    act = (df["action"] == "dropback").to_numpy().astype(int)
    out = np.zeros((n_units, 2, 1 + len(OUTCOME_COLS)), dtype=np.float64)
    np.add.at(out, (unit_index[cohort], act[cohort], 0), 1.0)
    for j, c in enumerate(OUTCOME_COLS):
        v = df[c].cast(pl.Float64).fill_null(0.0).to_numpy()
        np.add.at(out, (unit_index[cohort], act[cohort], 1 + j), v[cohort])
    return out


def build_team_games(frame: pl.DataFrame) -> TeamGames:
    games = (
        frame.group_by("game_id")
        .agg(pl.col("season").first(), pl.col("completed_at_est").first(),
             pl.col("known_at_pbp").max(), pl.col("known_at_ftn").max(), pl.col("known_at_part").max(),
             pl.col("offense_id").unique().alias("teams"))
        .explode("teams").rename({"teams": "team"})
        .sort(["completed_at_est", "game_id", "team"])
        .with_row_index("unit")
    )
    n = games.height
    unit_of = {(g, t): u for u, g, t in games.select("unit", "game_id", "team").iter_rows()}

    def units(team_col: str) -> np.ndarray:
        return np.array([unit_of[(g, t)] for g, t in frame.select("game_id", team_col).iter_rows()], dtype=np.int64)

    du, ou = units("defense_id"), units("offense_id")
    cell = frame["cell_idx"].to_numpy()
    defense = {
        "box": _onehot_counts(frame, du, n, cell, "t_box", 6),
        "rush": _onehot_counts(frame, du, n, cell, "t_rush", 5),
        "five_plus": _onehot_counts(frame, du, n, cell, "t_five_plus", 2),
        "blitz": _onehot_counts(frame, du, n, cell, "t_blitz", 2),
        "coverage": _onehot_counts(frame, du, n, cell, "t_coverage", len(COVERAGE_CLASSES)),
        "man": _onehot_counts(frame, du, n, cell, "t_man", 2),
        "pressure": _onehot_counts(frame, du, n, cell, "t_pressure", 2),
        "faced_dropback": _onehot_counts(frame, du, n, cell, "is_dropback", 2),
        "outcomes": _outcome_sums(frame, du, n),
    }
    offense = {
        "outcomes": _outcome_sums(frame, ou, n),
        "dropback": _onehot_counts(frame, ou, n, cell, "is_dropback", 2),
    }
    return TeamGames(games, defense, offense)


__all__ = ["load_play_frame", "build_team_games", "TeamGames", "SITUATION_CELLS", "OUTCOME_COLS",
           "DISTANCE_BANDS", "FIELD_ZONES"]
