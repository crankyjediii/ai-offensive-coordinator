"""Strict Pydantic request models mirroring contracts/recommendation-request.schema.json."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Team = Literal[
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
ID_PATTERN = r"^[A-Za-z0-9_.:-]+$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Situation(Strict):
    quarter: int = Field(ge=1, le=4)
    seconds_remaining_in_quarter: int = Field(ge=0, le=900)
    down: int = Field(ge=1, le=4)
    yards_to_go: int = Field(ge=1, le=99)
    yardline_100: int = Field(ge=1, le=99)
    score_differential: int = Field(ge=-100, le=100)
    goal_to_go: bool


class OptionalInputs(Strict):
    hash: Literal["left", "middle", "right"] | None = None
    offensive_personnel: str | None = Field(default=None, pattern=r"^[0-5][0-5]$")
    formation: Literal["under_center", "shotgun", "pistol"] | None = None
    motion: bool | None = None


class Utility(Strict):
    policy: Literal["expected_epa"]


class RecommendationRequest(Strict):
    schema_version: Literal["1.0.0"]
    request_id: UUID
    scenario_kind: Literal["hypothetical", "historical_replay"]
    mode: Literal["pbp_baseline", "charted_history", "enriched_current"]
    season: int = Field(ge=1920, le=2100)
    offense_team: Team | Literal["LEAGUE_AVERAGE"]
    defense_team: Team
    game_id: str | None = Field(max_length=128, pattern=ID_PATTERN)
    cutoff_at: str = Field(pattern=r"Z$")
    snapshot_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    model_bundle_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN)
    situation: Situation
    optional_inputs: OptionalInputs | None = None
    candidate_ids: list[str] = Field(min_length=1, max_length=40)
    utility: Utility

    @field_validator("request_id", mode="before")
    @classmethod
    def _uuid(cls, v: object) -> object:
        return UUID(v) if isinstance(v, str) else v  # JSON strings are the wire form of UUIDs

    @field_validator("candidate_ids")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("candidate_ids must be unique")
        return v

    @model_validator(mode="after")
    def _cross(self) -> RecommendationRequest:
        if self.scenario_kind == "historical_replay":
            if self.game_id is None:
                raise ValueError("historical_replay requires game_id")
            if self.offense_team == "LEAGUE_AVERAGE":
                raise ValueError("historical_replay cannot use LEAGUE_AVERAGE offense")
        if self.mode == "pbp_baseline" and self.optional_inputs is not None:
            raise ValueError("pbp_baseline does not accept optional_inputs")
        return self

    def as_contract_dict(self) -> dict:
        d = self.model_dump(mode="json", exclude_none=False)
        if d.get("optional_inputs") is None:
            d.pop("optional_inputs", None)
        else:
            d["optional_inputs"] = {k: v for k, v in d["optional_inputs"].items() if v is not None}
        return d
