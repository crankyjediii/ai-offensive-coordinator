import numpy as np
import pytest

from coordinator import config
from coordinator.models import bundle as bundle_mod
from coordinator.models.nets import OutcomeModel


@pytest.fixture
def tmp_bundles(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUNDLES_DIR", tmp_path)
    return tmp_path


def _spec(n_num=6):
    return {
        "bundle_name": "unit",
        "model_a": {"kind": "prior_mixture", "arch": None, "history_length": 8, "n_query_num": 3, "member_files": [],
                    "temperatures": {}, "standardizer": None},
        "model_b": {"kind": "mlp", "n_num": n_num, "hidden": 16, "member_files": ["model_b_seed0.pt"],
                    "standardizer": {"mean": [0.0] * n_num, "scale": [1.0] * n_num},
                    "platt": {e: [1.0, 0.0] for e in ("success", "explosive", "sack", "turnover", "first_down_or_td")},
                    "conformal_half_width": {"designed_rush": 1.0, "dropback": 2.0}, "conformal_nominal_coverage": 0.9},
    }


def test_bundle_round_trip_and_tamper_detection(tmp_bundles):
    m = OutcomeModel(6, hidden=16)
    bid = bundle_mod.write_bundle(_spec(), {"model_b_seed0.pt": m.state_dict()})
    assert bid.startswith("bundle_unit_")
    # Identical content yields the same ID (idempotent publish).
    assert bundle_mod.write_bundle(_spec(), {"model_b_seed0.pt": m.state_dict()}) == bid
    loaded = bundle_mod.load_bundle(bid)
    from coordinator.training.fit import BData
    d = BData(np.zeros((2, 4), dtype=np.int64), np.zeros((2, 6), dtype=np.float32), np.array([0, 1]),
              np.zeros(2, dtype=np.float32), np.zeros((2, 5), dtype=np.float32))
    out = loaded.predict_outcomes(d, np.array([0, 1]))
    assert out["event_probs"][0, 2] == 0.0  # structural zero sack for designed rush
    assert np.isfinite(out["epa_members"]).all()

    path = tmp_bundles / bid / "model_b_seed0.pt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(bundle_mod.BundleIntegrityError):
        bundle_mod.load_bundle(bid)


def test_missing_bundle_raises(tmp_bundles):
    with pytest.raises(FileNotFoundError):
        bundle_mod.load_bundle("bundle_missing")
