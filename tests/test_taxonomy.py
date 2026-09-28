import polars as pl
import pytest

from coordinator.taxonomy.labels import action_expr, outcome_exprs
from coordinator.taxonomy.vocab import box_bucket, distance_band, field_zone, rusher_bucket

BASE = dict(play_type="run", two_point_attempt=0.0, special=0.0, down=1.0, qb_dropback=0.0, rush_attempt=1.0,
            qb_scramble=0.0, sack=0.0, aborted_play=0.0, epa=0.1, yards_gained=4.0, interception=0.0,
            fumble_lost=0.0, first_down=0.0, touchdown=0.0, td_team=None, posteam="CHI", fourth_down_failed=0.0)


def classify(**overrides):
    row = {**BASE, **overrides}
    df = pl.DataFrame([row], schema_overrides={"td_team": pl.Utf8})
    a, r = action_expr()
    out = df.with_columns(a.alias("action"), r.alias("reason"))
    out = out.with_columns(outcome_exprs())
    return out.row(0, named=True)


@pytest.mark.parametrize("overrides,action,reason", [
    ({}, "designed_rush", None),
    ({"play_type": "pass", "qb_dropback": 1.0, "rush_attempt": 0.0}, "dropback", None),
    ({"play_type": "pass", "qb_dropback": 1.0, "rush_attempt": 0.0, "sack": 1.0}, "dropback", None),
    ({"qb_dropback": 1.0, "qb_scramble": 1.0}, "dropback", None),  # scramble stays in dropback
    ({"play_type": "qb_kneel"}, None, "excluded_play_type"),
    ({"play_type": "qb_spike"}, None, "excluded_play_type"),
    ({"play_type": "no_play"}, None, "excluded_play_type"),  # nullified penalty
    ({"play_type": "punt"}, None, "excluded_play_type"),
    ({"two_point_attempt": 1.0}, None, "excluded_play_type"),
    ({"qb_dropback": None}, None, "ambiguous_intent"),  # null is never false
    ({"qb_dropback": 1.0, "rush_attempt": 1.0, "qb_scramble": 0.0}, None, "ambiguous_intent"),  # contradiction
    ({"aborted_play": 1.0}, None, "ambiguous_intent"),
    ({"play_type": "novel_category", "rush_attempt": 0.0}, None, "unsupported_intent"),
])
def test_broad_action_classifier(overrides, action, reason):
    r = classify(**overrides)
    assert r["action"] == action
    assert r["reason"] == reason


def test_success_requires_positive_epa_and_missing_stays_missing():
    assert classify(epa=0.0)["y_success"] is False
    assert classify(epa=None)["y_success"] is None


def test_explosive_thresholds_depend_on_action():
    assert classify(yards_gained=10.0)["y_explosive"] is True
    db = dict(play_type="pass", qb_dropback=1.0, rush_attempt=0.0)
    assert classify(**db, yards_gained=19.0)["y_explosive"] is False
    assert classify(**db, yards_gained=20.0)["y_explosive"] is True
    # A long scramble uses the dropback threshold.
    assert classify(qb_dropback=1.0, qb_scramble=1.0, yards_gained=15.0)["y_explosive"] is False


def test_sack_structural_zero_for_designed_rush_and_null_when_unclassified():
    assert classify()["y_sack"] is False
    assert classify(play_type="qb_kneel")["y_sack"] is None


def test_defensive_touchdown_is_not_offensive_success():
    r = classify(touchdown=1.0, td_team="GB", interception=1.0, first_down=0.0)
    assert r["y_first_down_or_td"] is False
    assert r["y_turnover"] is True


def test_bins():
    assert [distance_band(x) for x in (1, 3, 4, 6, 7, 10, 11)] == ["short", "short", "medium", "medium", "long", "long", "very_long"]
    assert [field_zone(x) for x in (1, 5, 6, 20, 21, 80, 81)] == ["goal_line", "goal_line", "red_zone", "red_zone", "open_field", "open_field", "backed_up"]
    assert [box_bucket(x) for x in (3, 4, 5, 9, 11)] == [0, 0, 1, 5, 5]
    assert [rusher_bucket(x) for x in (2, 3, 4, 7, 9)] == [0, 0, 1, 4, 4]
