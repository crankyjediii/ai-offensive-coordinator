from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from coordinator.features.context import PriorParams, compute_priors
from coordinator.features.frame import SITUATION_CELLS, TeamGames
from coordinator.models.calibration import conformal_quantile, fit_temperature, softmax
from coordinator.training.metrics import brier, cluster_bootstrap_mean, log_loss

T0 = datetime(2025, 9, 1, tzinfo=UTC)


def _tg(units):
    """units: list of (team, season, days_offset, positives, total) for a binary head."""
    table = pl.DataFrame({
        "unit": list(range(len(units))), "game_id": [f"g{i}" for i in range(len(units))],
        "team": [u[0] for u in units], "season": [u[1] for u in units],
        "completed_at_est": [T0 + timedelta(days=u[2]) for u in units],
    }).with_columns(pl.col("completed_at_est").cast(pl.Datetime("us", "UTC")))
    for s in ("pbp", "ftn", "part"):
        table = table.with_columns((pl.col("completed_at_est") + timedelta(hours=1)).alias(f"known_at_{s}"))
    arr = np.zeros((len(units), SITUATION_CELLS + 1, 2))
    for i, (_, _, _, pos, tot) in enumerate(units):
        arr[i, SITUATION_CELLS] = [tot - pos, pos]
        arr[i, 0] = [tot - pos, pos]
    heads = {k: arr for k in ("box", "rush", "five_plus", "blitz", "coverage", "man", "pressure", "faced_dropback")}
    return TeamGames(table, heads, {})


def _ctx(cutoff_days, season=2025, team="GB"):
    return pl.DataFrame({"defense_id": [team], "offense_id": ["CHI"], "season": [season],
                         "cutoff_at": [T0 + timedelta(days=cutoff_days)]}).with_columns(
        pl.col("cutoff_at").cast(pl.Datetime("us", "UTC"))).with_row_index("context_idx")


def test_illustrative_update_arithmetic_from_the_spec():
    # Binary prior 0.30 (from league history), kappa=40, 20 current snaps with 8 positives -> 0.333.
    tg = _tg([("DAL", 2024, -300, 300, 1000), ("GB", 2025, 1, 8, 20)])
    p = PriorParams(tau_hist_days=1e9, tau_cur_days=1e9, kappa=40, a_team=0.0, league_pseudocount=1e-9, eta_cell=1e-9)
    priors, comps, support = compute_priors(tg, _ctx(5), p)
    assert comps["blitz"]["prior"][0, 1] == pytest.approx(0.30, abs=1e-6)
    assert priors["blitz"][0, SITUATION_CELLS, 1] == pytest.approx((8 + 40 * 0.30) / (20 + 40), abs=1e-6)
    n_raw, w, n_eff, _ = support["blitz"][0]
    assert n_raw == 20 and w == pytest.approx(20) and n_eff == pytest.approx(20)


def test_zero_history_falls_back_and_distributions_normalize():
    tg = _tg([("DAL", 2025, 1, 3, 10)])
    priors, _, support = compute_priors(tg, _ctx(0), PriorParams())  # nothing completed before the cutoff
    p = priors["blitz"][0]
    assert np.allclose(p.sum(-1), 1.0)
    assert np.allclose(p[SITUATION_CELLS], [0.5, 0.5])
    assert support["blitz"][0].tolist() == [0, 0, 0, 0]


def test_future_game_never_enters_prior():
    tg = _tg([("GB", 2025, 1, 5, 10), ("GB", 2025, 30, 10, 10)])
    a, _, _ = compute_priors(tg, _ctx(10), PriorParams())
    tg2 = _tg([("GB", 2025, 1, 5, 10)])
    b, _, _ = compute_priors(tg2, _ctx(10), PriorParams())
    assert np.allclose(a["blitz"], b["blitz"])


def test_kappa_zero_uses_current_rate_and_prior_when_empty():
    tg = _tg([("DAL", 2024, -300, 30, 100), ("GB", 2025, 1, 8, 20)])
    p = PriorParams(kappa=0.0)
    priors, _, _ = compute_priors(tg, _ctx(5), p)
    assert priors["blitz"][0, SITUATION_CELLS, 1] == pytest.approx(0.4)


def test_brier_normalization_and_log_loss():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    y = np.array([1, 1])
    assert np.allclose(brier(p, y), [0.04, 0.36])
    p3 = np.array([[0.5, 0.25, 0.25]])
    assert np.allclose(brier(p3, np.array([0])), [0.25 + 0.0625 + 0.0625])
    assert np.allclose(log_loss(p, y), -np.log([0.8, 0.4]))


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 3))
    y = np.array([rng.choice(3, p=softmax(z)) for z in true_logits])
    t = fit_temperature(true_logits * 3.0, y)
    assert 2.4 < t < 3.6


def test_conformal_quantile_finite_sample():
    r = np.arange(1, 101, dtype=float)
    assert conformal_quantile(r, 0.9) == 91.0


def test_cluster_bootstrap_counts_clusters():
    out = cluster_bootstrap_mean(np.array([1.0, 1.0, 0.0, 0.0]), np.array(["a", "a", "b", "b"]), reps=200)
    assert out["n_clusters"] == 2 and out["mean"] == 0.5
