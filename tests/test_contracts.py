import copy
import json

import jsonschema
import pytest

from coordinator import config
from coordinator.api.schemas import RecommendationRequest


def _validator(name):
    schema = json.loads((config.CONTRACTS_DIR / name).read_text())
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


REQ = json.loads((config.CONTRACTS_DIR / "recommendation-request.example.json").read_text())
RESP = json.loads((config.CONTRACTS_DIR / "recommendation-response.example.json").read_text())


def test_examples_validate():
    _validator("recommendation-request.schema.json").validate(REQ)
    _validator("recommendation-response.schema.json").validate(RESP)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(cutoff_at="2026-09-27T12:00:00+02:00"),
    lambda r: r.update(candidate_ids=["mesh"]),
    lambda r: r.update(optional_inputs={"hash": "left"}),
    lambda r: r["situation"].update(down=5),
    lambda r: r.update(unexpected=True),
    lambda r: r.update(request_id="not-a-uuid"),
    lambda r: r.update(scenario_kind="historical_replay"),  # needs a string game_id
])
def test_invalid_requests_rejected_by_schema_and_pydantic(mutate):
    bad = copy.deepcopy(REQ)
    mutate(bad)
    with pytest.raises(jsonschema.ValidationError):
        _validator("recommendation-request.schema.json").validate(bad)
    if "candidate_ids" not in bad or bad["candidate_ids"] != ["mesh"]:
        with pytest.raises(ValueError):
            RecommendationRequest.model_validate(bad)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="abstained"),  # abstained must not recommend
    lambda r: r["candidates"][0]["metrics"].update(epa_prediction_interval=None, missing_reasons={}),
    lambda r: r["candidates"][0]["metrics"].update(success_probability=1.2),
    lambda r: r.update(is_synthetic_example=False),  # notice must then be null
    lambda r: r["defense_expectation"].update(blitz_probability=0.3),  # unavailable must be null
])
def test_invalid_responses_rejected(mutate):
    bad = copy.deepcopy(RESP)
    mutate(bad)
    with pytest.raises(jsonschema.ValidationError):
        _validator("recommendation-response.schema.json").validate(bad)


def test_pydantic_accepts_the_example_and_round_trips():
    m = RecommendationRequest.model_validate(REQ)
    assert m.as_contract_dict() == REQ


def test_manifest_example_prohibits_training():
    m = json.loads((config.CONTRACTS_DIR / "data-manifest.example.json").read_text())
    assert m["audit_status"] == "not_run" and m["promotion_allowed"] is False
