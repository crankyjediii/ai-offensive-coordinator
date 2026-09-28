"""Chronological splits and array assembly for Model A and Model B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from coordinator.features.builder import FeatureBuilder, ModelInputs, target_rows
from coordinator.features.context import ContextFeatures, game_contexts
from coordinator.features.sequence import HistoryBatch
from coordinator.models.nets import A_HEADS, B_EVENTS

# Development layout (06-evaluation §3). 2022 is burn-in history only.
SPLITS: dict[str, pl.Expr] = {
    "select_train": (pl.col("season") == 2023) & (pl.col("week") <= 14),
    "select_early_stop": (pl.col("season") == 2023) & (pl.col("week") >= 15),
    "select_val": (pl.col("season") == 2024) & (pl.col("week") <= 12),
    "final_train": ((pl.col("season") == 2023) | ((pl.col("season") == 2024) & (pl.col("week") <= 8))),
    "final_early_stop": (pl.col("season") == 2024) & pl.col("week").is_between(9, 12),
    "calibration": (pl.col("season") == 2024) & (pl.col("week") >= 13),
    "test": pl.col("season") == 2025,
    "forward_2026": pl.col("season") == 2026,
}

A_TARGET_COLS = {"box": "t_box", "rush": "t_rush", "blitz": "t_blitz", "coverage": "t_coverage",
                 "man": "t_man", "pressure": "t_pressure"}
B_EVENT_COLS = {"success": "y_success", "explosive": "y_explosive", "sack": "y_sack",
                "turnover": "y_turnover", "first_down_or_td": "y_first_down_or_td"}


class Standardizer:
    """Mean/scale fitted on training rows only; stored in the model bundle."""

    MIN_STD = 1e-3
    CLIP = 5.0

    def __init__(self, mean: np.ndarray | None = None, scale: np.ndarray | None = None,
                 keep: np.ndarray | None = None, clip: float | None = None):
        self.mean, self.scale, self.keep, self.clip = mean, scale, keep, clip

    def fit(self, x: np.ndarray) -> Standardizer:
        """Features constant on the training rows carry no learnable signal and are zeroed, so a
        later distribution shift in them cannot push the network far outside its training range."""
        std = x.std(0)
        self.mean = x.mean(0)
        self.keep = std > self.MIN_STD
        self.scale = np.where(self.keep, std, 1.0)
        self.clip = self.CLIP
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        if self.keep is not None:
            z = z * self.keep
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)

    def to_dict(self) -> dict:
        d = {"mean": self.mean.tolist(), "scale": self.scale.tolist()}
        if self.keep is not None:
            d["keep"] = self.keep.tolist()
            d["clip"] = self.clip
        return d

    @staticmethod
    def from_dict(d: dict) -> Standardizer:
        keep = np.asarray(d["keep"], dtype=bool) if "keep" in d else None
        return Standardizer(np.asarray(d["mean"]), np.asarray(d["scale"]), keep, d.get("clip"))


@dataclass
class Workspace:
    """Everything derived once from a snapshot for all splits."""

    builder: FeatureBuilder
    contexts: pl.DataFrame
    cf: ContextFeatures
    hist: HistoryBatch
    rows: pl.DataFrame  # all target plays with context_idx


def make_workspace(builder: FeatureBuilder, history_length: int = 128, seasons=(2023, 2024, 2025, 2026)) -> Workspace:
    frame = builder.frame
    contexts = game_contexts(frame.filter(pl.col("season").is_in(list(seasons))))
    cf = builder.context_features(contexts)
    hist = builder.histories(contexts, history_length)
    rows = target_rows(frame.filter(pl.col("season").is_in(list(seasons))), contexts).sort(
        ["kickoff_at", "game_id", "play_order"])
    return Workspace(builder, contexts, cf, hist, rows)


def a_targets(rows: pl.DataFrame) -> dict[str, np.ndarray]:
    return {h: rows[c].cast(pl.Float64).fill_null(-1).to_numpy().astype(np.int64) for h, c in A_TARGET_COLS.items()}


def b_targets(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(epa, events[N,E] as float with nan=masked, action_idx)."""
    epa = rows["y_epa"].cast(pl.Float64).to_numpy().astype(np.float32)
    ev = np.stack([rows[B_EVENT_COLS[e]].cast(pl.Float64).to_numpy() for e in B_EVENTS], axis=1).astype(np.float32)
    act = (rows["action"] == "dropback").to_numpy().astype(np.int64)
    ev[act == 0, B_EVENTS.index("sack")] = np.nan  # structural zero: not a training target
    return epa, ev, act


def split_rows(ws: Workspace, split: str, outcome_cohort: bool = False) -> pl.DataFrame:
    r = ws.rows.filter(SPLITS[split])
    if outcome_cohort:
        r = r.filter(pl.col("in_reg_scrimmage_penalty_free_v1"))
    return r


def inputs_for(ws: Workspace, rows: pl.DataFrame, contract: str) -> ModelInputs:
    return ws.builder.inputs(rows, ws.cf, contract)


def a_heads_present() -> list[str]:
    return list(A_HEADS)
