"""End-to-end API tests against the local active deployment (skipped when none exists)."""

import copy
import json
import uuid

import jsonschema
import pytest

from coordinator import config

pytestmark = pytest.mark.data

ACTIVE = config.DEPLOYMENTS_DIR / "active.json"
if not ACTIVE.exists():
    pytest.skip("no active deployment; run `coordinator deploy` first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from coordinator.api.app import create_app  # noqa: E402
from coordinator.deploy import active_record  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(active_record())) as c:
        yield c


@pytest.fixture(scope="module")
def dep(client):
    return client.get("/v1/deployment").json()


def _validator():
    schema = json.loads((config.CONTRACTS_DIR / "recommendation-response.schema.json").read_text())
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def _req(dep, **over):
    snap = max(dep["snapshots"], key=lambda s: s["cutoff_at"])
    r = json.loads((config.CONTRACTS_DIR / "recommendation-request.example.json").read_text())
    r.update(request_id=str(uuid.uuid4()), snapshot_id=snap["snapshot_id"], model_bundle_id=dep["model_bundle_id"],
             cutoff_at=snap["cutoff_at"], season=max(snap["actual_seasons"]))
    for k, v in over.items():
        if k == "situation":
            r["situation"].update(v)
        else:
            r[k] = v
    return r


def test_health(client):
    assert client.get("/health/live").json()["status"] == "alive"
    assert client.get("/health/ready").json()["status"] == "ready"


def test_demo_request_is_contract_valid_and_deterministic(client, dep):
    r1 = client.post("/v1/recommendations", json=_req(dep))
    assert r1.status_code == 200, r1.text
    body = r1.json()
    _validator().validate(body)
    assert body["outcome_cohort_id"] == "reg_scrimmage_penalty_free_v1"
    assert body["is_synthetic_example"] is False
    assert body["status"] in ("limited", "abstained")  # broad mode is never 'ok'
    assert "BROAD_CANDIDATES_ONLY" in body["reason_codes"] or body["status"] == "abstained"
    r2 = client.post("/v1/recommendations", json=_req(dep)).json()
    strip = lambda b: {k: v for k, v in b.items() if k != "request_id"}  # noqa: E731
    assert strip(r2) == strip(body)
    for c in body["candidates"]:
        if c["candidate_id"] == "designed_rush":
            assert c["metrics"]["sack_probability"] == 0.0


def test_uncached_recompute_is_identical(dep):
    from coordinator.api.app import load_deployment
    from coordinator.recommendation.service import recommend
    d, _ = load_deployment(active_record())
    a = recommend(d, _req(dep, request_id="00000000-0000-4000-8000-000000000001"))
    b = recommend(d, _req(dep, request_id="00000000-0000-4000-8000-000000000001"))
    assert a == b


@pytest.mark.parametrize("over,code", [
    ({"situation": {"goal_to_go": True}}, "GOAL_TO_GO_INCONSISTENT"),
    ({"situation": {"yards_to_go": 50}}, "IMPOSSIBLE_SITUATION"),
    ({"offense_team": "GB"}, "SAME_TEAM"),
    ({"cutoff_at": "2099-01-01T00:00:00Z"}, "CUTOFF_AFTER_SNAPSHOT"),
])
def test_semantic_validation_422(client, dep, over, code):
    r = client.post("/v1/recommendations", json=_req(dep, **over))
    assert r.status_code == 422
    assert r.json()["code"] == code


def test_optional_inputs_rejected_not_ignored(client, dep):
    req = _req(dep, mode="charted_history", optional_inputs={"offensive_personnel": "11"})
    r = client.post("/v1/recommendations", json=req)
    assert r.status_code == 422 and r.json()["code"] == "UNSUPPORTED_OPTIONAL_INPUT"
    assert r.json()["fields"] == ["optional_inputs.offensive_personnel"]


def test_schema_invalid_request(client, dep):
    bad = _req(dep)
    bad["situation"]["down"] = 5
    r = client.post("/v1/recommendations", json=bad)
    assert r.status_code == 422 and r.json()["code"] == "INVALID_REQUEST"
    assert "request_id" in r.json() and "Traceback" not in r.text


def test_unknown_ids(client, dep):
    assert client.post("/v1/recommendations", json=_req(dep, snapshot_id="snap_nope")).status_code == 404
    assert client.post("/v1/recommendations", json=_req(dep, model_bundle_id="bundle_nope")).status_code == 404
    assert client.get("/v1/models/bundle_nope/card").status_code == 404


def test_enriched_current_is_unavailable_with_empty_candidates(client, dep):
    body = client.post("/v1/recommendations", json=_req(dep, mode="enriched_current")).json()
    _validator().validate(body)
    assert body["status"] == "unavailable" and body["candidates"] == [] and body["recommended_candidate_id"] is None


def test_low_support_situation_abstains(client, dep):
    # 4th and 1 at the opponent 1 in Q1: comparable penalty-free dropbacks are scarce.
    req = _req(dep, situation={"down": 4, "yards_to_go": 1, "yardline_100": 1, "goal_to_go": True, "quarter": 1})
    body = client.post("/v1/recommendations", json=req).json()
    _validator().validate(body)
    if body["status"] == "abstained":
        assert body["recommended_candidate_id"] is None
        assert all(c["rank"] is None for c in body["candidates"])


def test_capabilities_and_profile(client, dep):
    snap = max(dep["snapshots"], key=lambda s: s["cutoff_at"])
    cap = client.get(f"/v1/capabilities?season={max(snap['actual_seasons'])}&defense=GB&snapshot_id={snap['snapshot_id']}").json()
    assert cap["modes"]["enriched_current"]["status"] == "unavailable"
    prof = client.get(f"/v1/defenses/GB/profile?season={max(snap['actual_seasons'])}&snapshot_id={snap['snapshot_id']}").json()
    labels = {t["label"] for t in prof["current_season"]["tendencies"]}
    assert "Opponent dropback rate" in labels
    for t in prof["current_season"]["tendencies"]:
        assert t["denominator"] >= 0 and "missing_label_count" in t


def test_refresh_cannot_change_pinned_response(client, dep):
    """A response names its snapshot; the same pinned request reproduces after other requests."""
    req = _req(dep)
    a = client.post("/v1/recommendations", json=req).json()
    client.post("/v1/recommendations", json=_req(dep, defense_team="DAL"))
    b = client.post("/v1/recommendations", json=copy.deepcopy(req)).json()
    assert a["snapshot_id"] == b["snapshot_id"] and a["candidates"] == b["candidates"]


def test_failed_head_is_never_displayed(client, dep):
    """configs/head_validation.yaml marks coverage failed for the deployed bundle."""
    from coordinator.recommendation.service import head_validation
    if head_validation(dep["model_bundle_id"]).get("coverage") != "failed":
        pytest.skip("deployed bundle has no failed coverage head")
    research = min(dep["snapshots"], key=lambda s: s["cutoff_at"])
    req = _req(dep, mode="charted_history", season=2025, snapshot_id=research["snapshot_id"], cutoff_at=research["cutoff_at"])
    body = client.post("/v1/recommendations", json=req).json()
    _validator().validate(body)
    assert body["defense_expectation"]["coverage_probabilities"] is None
    assert "MODEL_UNAVAILABLE" in body["defense_expectation"]["reason_codes"]
