"""Defensive history sequences for Model A.

Each context receives the latest ``L`` eligible snaps of its defense, ordered oldest -> newest and
left-padded. Charted fields whose source was not yet available at the context cutoff are replaced by
the reserved NOT_YET_AVAILABLE token; they are never shown as observed zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

PAD, UNKNOWN, NOT_APPLICABLE, NOT_YET_AVAILABLE = 0, 1, 2, 3
OFFSET = 4

# (name, source, n_values). Source decides the availability mask.
CAT_FIELDS: list[tuple[str, str, int]] = [
    ("down", "pbp", 4),
    ("band", "pbp", 4),
    ("zone", "pbp", 4),
    ("quarter", "pbp", 5),
    ("action", "pbp", 2),
    ("success", "pbp", 2),
    ("box", "ftn", 6),
    ("rush", "ftn", 5),
    ("blitz", "ftn", 2),
    ("qb_alignment", "ftn", 3),
    ("motion", "ftn", 2),
    ("play_action", "ftn", 2),
    ("coverage", "part", 8),
    ("man", "part", 2),
    ("pressure", "part", 2),
    ("off_te", "part", 4),
]
NUM_FIELDS = ["distance", "yards_to_goal", "margin", "half_time", "epa", "yards",
              "days_before_cutoff", "same_season", "game_boundary", "games_back"]
CAT_CARDINALITY = [OFFSET + n for _, _, n in CAT_FIELDS]


def _cat_values(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Static per-play categorical IDs, plus a per-field source index (0 pbp, 1 ftn, 2 part)."""
    is_db = frame["is_dropback"].to_numpy()

    def enc(values: np.ndarray, na_mask: np.ndarray | None = None) -> np.ndarray:
        out = np.where(np.isnan(values), UNKNOWN, values + OFFSET).astype(np.int64)
        if na_mask is not None:
            out = np.where(na_mask, NOT_APPLICABLE, out)
        return out

    def f(col: str) -> np.ndarray:
        return frame[col].cast(pl.Float64).to_numpy()

    qb = frame["qb_alignment"].replace_strict({"shotgun": 0, "under_center": 1, "pistol": 2}, default=None).cast(pl.Float64).to_numpy()
    te = np.minimum(f("off_te"), 3)
    cols = [
        enc(f("down_idx")), enc(f("band_idx")), enc(f("zone_idx")), enc(np.minimum(f("quarter"), 5) - 1),
        enc(is_db.astype(float)), enc(f("y_success")),
        enc(f("t_box")), enc(f("t_rush"), ~is_db), enc(f("t_blitz"), ~is_db), enc(qb),
        enc(f("motion_observed")), enc(f("play_action_observed")),
        enc(f("t_coverage"), ~is_db), enc(f("t_man"), ~is_db), enc(f("t_pressure"), ~is_db), enc(te),
    ]
    src = np.array([{"pbp": 0, "ftn": 1, "part": 2}[s] for _, s, _ in CAT_FIELDS])
    return np.stack(cols, axis=1), src


@dataclass
class HistoryBatch:
    cat: np.ndarray  # (C, L, Fcat) int64
    num: np.ndarray  # (C, L, Fnum) float32
    num_missing: np.ndarray  # (C, L, Fnum) bool
    padding_mask: np.ndarray  # (C, L) True where padded
    length: np.ndarray  # (C,) number of real tokens


def _days(col: pl.Series) -> np.ndarray:
    arr = col.dt.replace_time_zone(None).to_numpy().astype("datetime64[s]")
    out = arr.astype(np.float64) / 86400.0
    return np.where(np.isnat(arr), np.inf, out)


def build_histories(frame: pl.DataFrame, contexts: pl.DataFrame, length: int = 128) -> HistoryBatch:
    cat_static, src = _cat_values(frame)
    known = np.stack([_days(frame["known_at_pbp"]), _days(frame["known_at_ftn"]), _days(frame["known_at_part"])], axis=1)
    done = _days(frame["completed_at_est"])
    season = frame["season"].to_numpy()
    game = frame["game_id"].to_numpy()
    num_static = np.stack([
        frame["distance"].cast(pl.Float64).to_numpy() / 10.0,
        frame["yards_to_goal"].cast(pl.Float64).to_numpy() / 100.0,
        frame["offense_score_margin"].cast(pl.Float64).to_numpy() / 14.0,
        frame["half_seconds_remaining"].cast(pl.Float64).to_numpy() / 1800.0,
        frame["y_epa"].cast(pl.Float64).to_numpy() / 2.0,
        frame["y_yards"].cast(pl.Float64).to_numpy() / 10.0,
    ], axis=1)
    by_defense: dict[str, np.ndarray] = {}
    for d, idx in frame.with_row_index("i").group_by("defense_id").agg(pl.col("i")).iter_rows():
        by_defense[d] = np.asarray(idx, dtype=np.int64)  # frame is time-sorted

    c = contexts.height
    fc, fn = len(CAT_FIELDS), len(NUM_FIELDS)
    cat = np.zeros((c, length, fc), dtype=np.int64)
    num = np.zeros((c, length, fn), dtype=np.float32)
    miss = np.zeros((c, length, fn), dtype=bool)
    pad = np.ones((c, length), dtype=bool)
    lens = np.zeros(c, dtype=np.int64)
    cut_days = _days(contexts["cutoff_at"])
    c_season = contexts["season"].to_numpy()
    for i, d in enumerate(contexts["defense_id"].to_list()):
        idx = by_defense.get(d)
        if idx is None:
            continue
        cut = cut_days[i]
        elig = idx[(known[idx, 0] <= cut) & (done[idx] < cut)]
        sel = elig[-length:]
        n = len(sel)
        lens[i] = n
        if n == 0:
            continue
        s = slice(length - n, length)
        pad[i, s] = False
        cv = cat_static[sel].copy()
        not_yet = known[sel][:, src] > cut  # (n, Fcat)
        cv = np.where(not_yet & (cv != NOT_APPLICABLE), NOT_YET_AVAILABLE, cv)
        cat[i, s] = cv
        nv = num_static[sel]
        g = game[sel]
        boundary = np.r_[True, g[1:] != g[:-1]]
        games_back = np.cumsum(boundary[::-1])[::-1] - 1
        extra = np.stack([
            (cut - done[sel]) / 365.0,
            (season[sel] == c_season[i]).astype(float),
            boundary.astype(float),
            games_back / 17.0,
        ], axis=1)
        full = np.concatenate([nv, extra], axis=1)
        m = np.isnan(full)
        num[i, s] = np.where(m, 0.0, full)
        miss[i, s] = m
    return HistoryBatch(cat, num, miss, pad, lens)
