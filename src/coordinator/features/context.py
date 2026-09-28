"""Context-level features computed as-of a cutoff: defensive priors and lagged strengths.

A context is (defense, offense, season, cutoff). Every aggregate is a weighted sum over team-game
units whose source-specific known-at time is at or before the cutoff. Weights never include the
context's own game because its completion is after a pregame cutoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from coordinator.config import LEAGUE_AVERAGE
from coordinator.features.frame import OUTCOME_COLS, SITUATION_CELLS, TeamGames

PRIOR_HEADS = {
    # head: (team-game array, known-at source)
    "box": ("box", "ftn"),
    "rush": ("rush", "ftn"),
    "five_plus": ("five_plus", "ftn"),
    "blitz": ("blitz", "ftn"),
    "coverage": ("coverage", "part"),
    "man": ("man", "part"),
    "pressure": ("pressure", "part"),
    "faced_dropback": ("faced_dropback", "pbp"),
}


@dataclass(frozen=True)
class PriorParams:
    """Empirical-shrinkage prior. Values are selected on earlier folds (experiment R01)."""

    tau_hist_days: float = 365.0
    tau_cur_days: float = 10_000.0
    eta_cell: float = 30.0
    eta_team: float = 60.0
    kappa: float = 50.0
    a_team: float = 0.5
    league_pseudocount: float = 1.0
    method_version: str = "prior_v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StrengthParams:
    tau_days: float = 300.0
    shrink_plays: float = 150.0


@dataclass
class ContextFeatures:
    contexts: pl.DataFrame
    priors: dict[str, np.ndarray]  # head -> (C, cells+1, K) smoothed distributions
    components: dict[str, dict[str, np.ndarray]]  # head -> league/team/current pieces (overall)
    support: dict[str, np.ndarray]  # head -> (C, 4): N_current, W_current, n_eff, W_hist_team
    strength: np.ndarray  # (C, F_strength)
    strength_names: list[str] = field(default_factory=list)
    params: PriorParams = field(default_factory=PriorParams)


def _to_days(dt: np.ndarray) -> np.ndarray:
    return dt.astype("datetime64[s]").astype(np.float64) / 86400.0


def _as_np_dt(col: pl.Series) -> np.ndarray:
    return col.dt.replace_time_zone(None).to_numpy().astype("datetime64[s]")


def _weights(tg: TeamGames, contexts: pl.DataFrame, source: str, team_col: str | None,
             season_rel: str, tau_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights C x U, eligibility C x U). season_rel in {'hist', 'cur', 'all'}."""
    t = tg.table
    known = _as_np_dt(t[f"known_at_{source}"])
    known_days = np.where(np.isnat(known), np.inf, _to_days(np.where(np.isnat(known), np.datetime64(0, "s"), known)))
    done_days = _to_days(_as_np_dt(t["completed_at_est"]))
    cut_days = _to_days(_as_np_dt(contexts["cutoff_at"]))[:, None]
    elig = (known_days[None, :] <= cut_days) & (done_days[None, :] < cut_days)
    u_season = t["season"].to_numpy()[None, :]
    c_season = contexts["season"].to_numpy()[:, None]
    if season_rel == "hist":
        elig &= u_season < c_season
    elif season_rel == "cur":
        elig &= u_season == c_season
    if team_col is not None:
        teams = contexts[team_col].to_numpy()[:, None]
        elig &= t["team"].to_numpy()[None, :] == teams
    age = np.clip(cut_days - done_days[None, :], 0, None)
    w = np.where(elig, np.exp(-age / tau_days), 0.0)
    return w, elig


def _normalize(x: np.ndarray) -> np.ndarray:
    tot = x.sum(-1, keepdims=True)
    return np.divide(x, tot, out=np.zeros_like(x), where=tot > 0)


def compute_priors(tg: TeamGames, contexts: pl.DataFrame, params: PriorParams) -> tuple[dict, dict, dict]:
    priors: dict[str, np.ndarray] = {}
    comps: dict[str, dict[str, np.ndarray]] = {}
    support: dict[str, np.ndarray] = {}
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for head, (arr_name, source) in PRIOR_HEADS.items():
        x = tg.defense[arr_name]  # (U, cells+1, K)
        k = x.shape[-1]
        key_l, key_t, key_c = (source, "hist", None), (source, "hist", "team"), (source, "cur", "team")
        for key, (sr, tc, tau) in {
            key_l: ("hist", None, params.tau_hist_days),
            key_t: ("hist", "defense_id", params.tau_hist_days),
            key_c: ("cur", "defense_id", params.tau_cur_days),
        }.items():
            if key not in cache:
                cache[key] = _weights(tg, contexts, source, tc, sr, tau)
        wl, _ = cache[key_l]
        wt, _ = cache[key_t]
        wc, ec = cache[key_c]
        hl = np.einsum("cu,ujk->cjk", wl, x)
        ht = np.einsum("cu,ujk->cjk", wt, x)
        cc = np.einsum("cu,ujk->cjk", wc, x)

        uniform = np.full(k, 1.0 / k)
        o = SITUATION_CELLS
        pl_o = (hl[:, o] + params.league_pseudocount * uniform) / (hl[:, o].sum(-1, keepdims=True) + params.league_pseudocount)
        pl_cell = (hl + params.eta_cell * pl_o[:, None, :]) / (hl.sum(-1, keepdims=True) + params.eta_cell)
        pl_cell[:, o] = pl_o
        pt_cell = (ht + params.eta_team * pl_cell) / (ht.sum(-1, keepdims=True) + params.eta_team)
        p0 = (1.0 - params.a_team) * pl_cell + params.a_team * pt_cell
        denom = cc.sum(-1, keepdims=True) + params.kappa
        p_hat = np.where(denom > 0, (cc + params.kappa * p0) / np.where(denom > 0, denom, 1.0), p0)
        priors[head] = p_hat
        comps[head] = {"league": pl_cell[:, o], "team": pt_cell[:, o], "prior": p0[:, o],
                       "current_counts": cc[:, o]}

        n_per_unit = x[:, o].sum(-1)
        n_raw = (ec.astype(np.float64) * n_per_unit[None, :]).sum(1)
        w_cur = (wc * n_per_unit[None, :]).sum(1)
        w_sq = ((wc**2) * n_per_unit[None, :]).sum(1)
        n_eff = np.divide(w_cur**2, w_sq, out=np.zeros_like(w_cur), where=w_sq > 0)
        w_hist_team = (wt * n_per_unit[None, :]).sum(1)
        support[head] = np.stack([n_raw, w_cur, n_eff, w_hist_team], axis=1)
    return priors, comps, support


def compute_strength(tg: TeamGames, contexts: pl.DataFrame, params: StrengthParams) -> tuple[np.ndarray, list[str]]:
    """Lagged, league-shrunk offense and defense-allowed outcome rates by broad action."""
    wl, _ = _weights(tg, contexts, "pbp", None, "all", params.tau_days)
    names: list[str] = []
    feats = []
    league_sum = np.einsum("cu,uaf->caf", wl, tg.defense["outcomes"])
    league_mean = np.divide(league_sum[..., 1:], league_sum[..., :1], out=np.zeros_like(league_sum[..., 1:]),
                            where=league_sum[..., :1] > 0)
    for side, arr, team_col in (("off", tg.offense["outcomes"], "offense_id"), ("def", tg.defense["outcomes"], "defense_id")):
        wt, _ = _weights(tg, contexts, "pbp", team_col, "all", params.tau_days)
        if side == "off":
            wt = np.where((contexts[team_col] == LEAGUE_AVERAGE).to_numpy()[:, None], 0.0, wt)
        s = np.einsum("cu,uaf->caf", wt, arr)
        n = s[..., :1]
        shrunk = (s[..., 1:] + params.shrink_plays * league_mean) / (n + params.shrink_plays)
        for a, aname in enumerate(("rush", "db")):
            for j, c in enumerate(OUTCOME_COLS):
                feats.append(shrunk[:, a, j])
                names.append(f"{side}_{aname}_{c.removeprefix('y_')}")
            feats.append(np.log1p(n[:, a, 0]))
            names.append(f"{side}_{aname}_log_n")
    for a, aname in enumerate(("rush", "db")):
        for j, c in enumerate(OUTCOME_COLS):
            feats.append(league_mean[:, a, j])
            names.append(f"league_{aname}_{c.removeprefix('y_')}")
    return np.stack(feats, axis=1).astype(np.float32), names


def compute_context_features(tg: TeamGames, contexts: pl.DataFrame,
                             prior_params: PriorParams | None = None,
                             strength_params: StrengthParams | None = None) -> ContextFeatures:
    prior_params = prior_params or PriorParams()
    strength_params = strength_params or StrengthParams()
    priors, comps, support = compute_priors(tg, contexts, prior_params)
    strength, names = compute_strength(tg, contexts, strength_params)
    return ContextFeatures(contexts, priors, comps, support, strength, names, prior_params)


def make_contexts(rows: list[tuple[str, str, int, datetime]]) -> pl.DataFrame:
    """rows: (defense_id, offense_id, season, cutoff_at UTC)."""
    return pl.DataFrame(
        rows, schema={"defense_id": pl.Utf8, "offense_id": pl.Utf8, "season": pl.Int16,
                      "cutoff_at": pl.Datetime("us", "UTC")}, orient="row",
    ).with_row_index("context_idx")


def game_contexts(frame: pl.DataFrame, cutoff_offset_minutes: int = 60) -> pl.DataFrame:
    """Pregame contexts for every (game, defense) in the frame; cutoff = kickoff - offset."""
    g = (
        frame.group_by(["game_id", "defense_id", "offense_id"])
        .agg(pl.col("season").first(), pl.col("week").first(), pl.col("kickoff_at").first())
        .with_columns((pl.col("kickoff_at") - pl.duration(minutes=cutoff_offset_minutes)).alias("cutoff_at"))
        .sort(["kickoff_at", "game_id", "defense_id"])
        .with_row_index("context_idx")
    )
    return g
