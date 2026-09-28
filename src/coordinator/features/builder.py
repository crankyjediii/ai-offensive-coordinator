"""Shared feature builder used by training, backtests and the API (training-serving parity).

Only fields present in the recommendation request contract enter per-play features:
quarter, seconds remaining in quarter, down, yards to go, yardline_100, score differential,
goal-to-go. Context features come from as-of aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl

from coordinator.data.snapshot import Snapshot
from coordinator.features.context import (
    ContextFeatures,
    PriorParams,
    StrengthParams,
    compute_context_features,
)
from coordinator.features.frame import SITUATION_CELLS, TeamGames, build_team_games, load_play_frame
from coordinator.features.sequence import HistoryBatch, build_histories

FEATURE_CONTRACTS = ("pbp", "pbp_ftn", "charted")
PRIOR_HEADS_BY_CONTRACT = {
    "pbp": ["faced_dropback"],
    "pbp_ftn": ["faced_dropback", "box", "five_plus", "blitz"],
    "charted": ["faced_dropback", "box", "five_plus", "blitz", "coverage", "man", "pressure"],
}
SITUATION_NUM = ["distance", "log_distance", "yards_to_goal", "goal_to_go", "margin", "half_time",
                 "quarter_time", "is_second_half"]
SITUATION_REQUIRED = ["quarter", "seconds_remaining", "down", "distance", "yards_to_goal",
                      "offense_score_margin", "goal_to_go"]


def situation_arrays(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """rows need SITUATION_REQUIRED columns. Returns (cat[N,4], num[N,8])."""
    q = rows["quarter"].cast(pl.Int64).to_numpy()
    sec = rows["seconds_remaining"].cast(pl.Float64).to_numpy()
    dist = rows["distance"].cast(pl.Float64).to_numpy()
    ytg = rows["yards_to_goal"].cast(pl.Float64).to_numpy()
    down = rows["down"].cast(pl.Int64).to_numpy()
    band = np.select([dist <= 3, dist <= 6, dist <= 10], [0, 1, 2], 3)
    zone = np.select([ytg <= 5, ytg <= 20, ytg <= 80], [0, 1, 2], 3)
    half_sec = np.where(np.isin(q, (1, 3)), sec + 900, sec)
    cat = np.stack([down - 1, band, zone, np.minimum(q, 5) - 1], axis=1).astype(np.int64)
    num = np.stack([
        dist / 10.0, np.log1p(dist), ytg / 100.0, rows["goal_to_go"].cast(pl.Float64).to_numpy(),
        np.clip(rows["offense_score_margin"].cast(pl.Float64).to_numpy(), -28, 28) / 14.0,
        half_sec / 1800.0, sec / 900.0, (q >= 3).astype(float),
    ], axis=1).astype(np.float32)
    return cat, num


def prior_arrays(cf: ContextFeatures, ctx_idx: np.ndarray, cell_idx: np.ndarray, heads: list[str]) -> tuple[np.ndarray, list[str]]:
    feats, names = [], []
    for h in heads:
        p = cf.priors[h]
        k = p.shape[-1]
        cell_p = p[ctx_idx, cell_idx]
        over_p = p[ctx_idx, SITUATION_CELLS]
        cols = range(k) if k > 2 else [1]
        for j in cols:
            feats += [cell_p[:, j], over_p[:, j]]
            names += [f"prior_{h}_{j}_cell", f"prior_{h}_{j}_all"]
        sup = cf.support[h][ctx_idx]
        feats += [np.log1p(sup[:, 0]), np.log1p(sup[:, 3])]
        names += [f"support_{h}_log_n_cur", f"support_{h}_log_w_hist"]
    return np.stack(feats, axis=1).astype(np.float32), names


@dataclass
class ModelInputs:
    rows: pl.DataFrame
    sit_cat: np.ndarray
    sit_num: np.ndarray
    ctx_num: np.ndarray  # priors + strength
    ctx_names: list[str]
    ctx_idx: np.ndarray


class FeatureBuilder:
    """Holds the as-of play frame for one snapshot and builds features for any contexts."""

    def __init__(self, snapshot: Snapshot, prior_params: PriorParams | None = None,
                 strength_params: StrengthParams | None = None):
        self.snapshot = snapshot
        self.prior_params = prior_params or PriorParams()
        self.strength_params = strength_params or StrengthParams()

    @cached_property
    def frame(self) -> pl.DataFrame:
        return load_play_frame(self.snapshot)

    @cached_property
    def team_games(self) -> TeamGames:
        return build_team_games(self.frame)

    def context_features(self, contexts: pl.DataFrame) -> ContextFeatures:
        return compute_context_features(self.team_games, contexts, self.prior_params, self.strength_params)

    def histories(self, contexts: pl.DataFrame, length: int) -> HistoryBatch:
        return build_histories(self.frame, contexts, length)

    def inputs(self, rows: pl.DataFrame, cf: ContextFeatures, contract: str) -> ModelInputs:
        """rows: SITUATION_REQUIRED columns plus context_idx."""
        sit_cat, sit_num = situation_arrays(rows)
        ctx_idx = rows["context_idx"].cast(pl.Int64).to_numpy()
        cell = sit_cat[:, 0] * 4 + sit_cat[:, 1]
        pri, pnames = prior_arrays(cf, ctx_idx, cell, PRIOR_HEADS_BY_CONTRACT[contract])
        ctx_num = np.concatenate([pri, cf.strength[ctx_idx]], axis=1)
        return ModelInputs(rows, sit_cat, sit_num, ctx_num, pnames + cf.strength_names, ctx_idx)


def target_rows(frame: pl.DataFrame, contexts: pl.DataFrame) -> pl.DataFrame:
    """Attach each eligible play to its pregame (game, defense) context."""
    return frame.join(
        contexts.select("context_idx", "game_id", "defense_id"), on=["game_id", "defense_id"], how="inner"
    )
