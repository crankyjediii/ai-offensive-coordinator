"""Availability and leakage sentinels."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import torch

from coordinator.data.asof import eligible_versions
from coordinator.features.builder import SITUATION_REQUIRED, situation_arrays
from coordinator.features.sequence import NOT_APPLICABLE, NOT_YET_AVAILABLE, PAD, build_histories
from coordinator.models.nets import DefenseSequenceModel

T0 = datetime(2025, 10, 1, tzinfo=UTC)


def _versions():
    return pl.DataFrame({
        "game_id": ["g1", "g1", "g2"],
        "play_id": [1, 1, 1],
        "record_version_id": ["v1", "v2", "v1"],
        "known_at_reconstructed": [T0, T0 + timedelta(days=3), T0 + timedelta(days=10)],
        "completed_at_est": [T0 - timedelta(days=1)] * 2 + [T0 + timedelta(days=8)],
    }).with_columns(pl.col("known_at_reconstructed", "completed_at_est").cast(pl.Datetime("us", "UTC")))


def test_future_record_is_rejected_and_latest_eligible_version_selected():
    df = _versions()
    before = eligible_versions(df, T0 + timedelta(days=1), "reconstructed_conservative_delay", ["game_id", "play_id"]).collect()
    assert before["record_version_id"].to_list() == ["v1"]
    later = eligible_versions(df, T0 + timedelta(days=5), "reconstructed_conservative_delay", ["game_id", "play_id"]).collect()
    assert later["record_version_id"].to_list() == ["v2"]
    # g2 completes after the cutoff: never eligible even if someone injects an early known_at.
    injected = df.with_columns(pl.when(pl.col("game_id") == "g2").then(pl.lit(T0)).otherwise(pl.col("known_at_reconstructed"))
                               .cast(pl.Datetime("us", "UTC")).alias("known_at_reconstructed"))
    out = eligible_versions(injected, T0 + timedelta(days=5), "reconstructed_conservative_delay", ["game_id", "play_id"]).collect()
    assert "g2" not in out["game_id"].to_list()


def _frame(n=6):
    base = T0
    rows = []
    for i in range(n):
        done = base + timedelta(days=7 * (i // 2))
        rows.append({
            "game_id": f"g{i // 2}", "defense_id": "GB", "season": 2025, "is_dropback": i % 2 == 0,
            "down_idx": 0, "band_idx": 1, "zone_idx": 2, "quarter": 1, "y_success": True, "t_box": 2, "t_rush": 1 if i % 2 == 0 else None,
            "t_blitz": 0 if i % 2 == 0 else None, "qb_alignment": "shotgun", "motion_observed": False, "play_action_observed": False,
            "t_coverage": 1 if i % 2 == 0 else None, "t_man": 1 if i % 2 == 0 else None, "t_pressure": 0 if i % 2 == 0 else None,
            "off_te": 1, "distance": 7, "yards_to_goal": 50, "offense_score_margin": 0, "half_seconds_remaining": 1000,
            "y_epa": 0.1, "y_yards": 4.0, "completed_at_est": done, "known_at_pbp": done + timedelta(days=1),
            "known_at_ftn": done + timedelta(days=2), "known_at_part": datetime(2026, 3, 1, tzinfo=UTC),
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("completed_at_est", "known_at_pbp", "known_at_ftn", "known_at_part").cast(pl.Datetime("us", "UTC")))


def _ctx(cutoff):
    return pl.DataFrame({"defense_id": ["GB"], "season": [2025], "cutoff_at": [cutoff]}).with_columns(
        pl.col("cutoff_at").cast(pl.Datetime("us", "UTC")))


def test_history_excludes_unfinished_and_masks_unpublished_labels():
    f = _frame()
    cut = T0 + timedelta(days=7, hours=30)  # g0 (+1d pbp, +2d ftn) visible; g1 pbp visible, g1 ftn not yet
    h = build_histories(f, _ctx(cut), length=8)
    assert h.length[0] == 4
    cat = h.cat[0][~h.padding_mask[0]]
    box_col, cov_col, rush_col = 6, 12, 7
    assert list(cat[:2, box_col]) != [NOT_YET_AVAILABLE] * 2
    assert list(cat[2:, box_col]) == [NOT_YET_AVAILABLE] * 2
    assert set(cat[:, cov_col]) <= {NOT_YET_AVAILABLE, NOT_APPLICABLE}  # participation published after the season
    assert cat[1, rush_col] == NOT_APPLICABLE  # rush counts do not apply to designed rushes
    assert (h.cat[0][h.padding_mask[0]] == PAD).all()


def test_injected_future_row_does_not_change_history():
    f = _frame()
    cut = T0 + timedelta(days=10)
    a = build_histories(f, _ctx(cut), 8)
    future = f.tail(1).with_columns(pl.lit("gX").alias("game_id"),
                                    (pl.col("completed_at_est") + timedelta(days=30)).alias("completed_at_est"),
                                    (pl.col("known_at_pbp") + timedelta(days=30)).alias("known_at_pbp"))
    b = build_histories(pl.concat([f, future]), _ctx(cut), 8)
    assert np.array_equal(a.cat, b.cat) and np.array_equal(a.num, b.num)


def test_situation_features_use_only_contract_fields():
    rows = pl.DataFrame({"quarter": [2], "seconds_remaining": [582], "down": [2], "distance": [7], "yards_to_goal": [38],
                         "offense_score_margin": [0], "goal_to_go": [False]})
    assert set(rows.columns) == set(SITUATION_REQUIRED)
    cat, num = situation_arrays(rows)
    assert cat.tolist() == [[1, 2, 2, 1]]
    assert np.isclose(num[0, 5], 582 / 1800)  # second quarter: half clock equals quarter clock


def test_padding_values_cannot_change_prediction():
    torch.manual_seed(0)
    m = DefenseSequenceModel(n_query_num=5, d=32, n_heads=4, n_layers=1, ff=64, max_len=8).eval()
    b, length = 2, 8
    from coordinator.features.sequence import CAT_CARDINALITY, NUM_FIELDS
    cat = torch.stack([torch.randint(4, c, (b, length)) for c in CAT_CARDINALITY], -1)
    num = torch.randn(b, length, len(NUM_FIELDS))
    miss = torch.zeros_like(num, dtype=torch.bool)
    pad = torch.zeros(b, length, dtype=torch.bool)
    pad[:, :3] = True
    sit = torch.tensor([[1, 2, 2, 1], [0, 0, 3, 0]])
    q = torch.randn(b, 5)
    with torch.inference_mode():
        out1 = m(cat, num, miss, pad, sit, q)
        cat2, num2 = cat.clone(), num.clone()
        cat2[:, :3] = 1
        num2[:, :3] = 99.0
        out2 = m(cat2, num2, miss, pad, sit, q)
        empty = m(cat, num, miss, torch.ones_like(pad), sit, q)  # zero-history rows keep the query token
    for h in out1:
        assert torch.allclose(out1[h], out2[h], atol=1e-5)
        assert torch.isfinite(empty[h]).all()


def test_prior_offset_model_starts_at_the_prior():
    from coordinator.models.nets import A_HEADS
    torch.manual_seed(0)
    m = DefenseSequenceModel(n_query_num=5, d=32, n_heads=4, n_layers=1, ff=64, max_len=8, prior_offset=True).eval()
    from coordinator.features.sequence import CAT_CARDINALITY, NUM_FIELDS
    b, length = 3, 8
    cat = torch.stack([torch.randint(4, c, (b, length)) for c in CAT_CARDINALITY], -1)
    prior = {h: torch.log_softmax(torch.randn(b, k), -1) for h, k in A_HEADS.items()}
    with torch.inference_mode():
        out = m(cat, torch.randn(b, length, len(NUM_FIELDS)), torch.zeros(b, length, len(NUM_FIELDS), dtype=torch.bool),
                torch.zeros(b, length, dtype=torch.bool), torch.zeros(b, 4, dtype=torch.long), torch.randn(b, 5), prior)
    for h in A_HEADS:
        assert torch.allclose(torch.softmax(out[h], -1), prior[h].exp(), atol=1e-6)
