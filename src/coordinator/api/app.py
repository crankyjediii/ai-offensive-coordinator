"""FastAPI service. Read-only public routes; ingestion, training and deployment stay offline."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache

import jsonschema
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from coordinator import config
from coordinator.analysis.profile import observed_tendencies, smoothed_estimates
from coordinator.api.schemas import RecommendationRequest
from coordinator.data.snapshot import load_snapshot, parse_utc
from coordinator.deploy import active_record
from coordinator.models.bundle import load_bundle
from coordinator.recommendation.service import Deployment, DomainError, capabilities, recommend, resolve_snapshot

MAX_BODY_BYTES = 16 * 1024
RATE_LIMIT_PER_MINUTE = int(os.environ.get("COORDINATOR_RATE_LIMIT", "30"))
CACHE_SIZE = 512


@lru_cache(maxsize=1)
def response_validator() -> jsonschema.protocols.Validator:
    schema = json.loads((config.CONTRACTS_DIR / "recommendation-response.schema.json").read_text())
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


class State:
    deployment: Deployment | None = None
    record: dict | None = None
    error: str | None = None
    cache: OrderedDict[str, dict] = OrderedDict()
    hits: dict[str, deque] = defaultdict(deque)
    lock = threading.Lock()


def _error(status: int, code: str, message: str, fields: list[str] | None = None, request_id: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"request_id": request_id or str(uuid.uuid4()), "code": code,
                                                     "message": message, "fields": fields or []})


def load_deployment(record: dict | None = None) -> tuple[Deployment, dict]:
    record = record or active_record()
    bundle = load_bundle(record["model_bundle_id"])
    snaps = [load_snapshot(s) for s in record["snapshot_ids"]]
    dep = Deployment.create(snaps, bundle)
    for b in dep.builders.values():  # warm caches so readiness means ready
        _ = b.team_games
    return dep, record


def create_app(record: dict | None = None) -> FastAPI:
    state = State()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            state.deployment, state.record = load_deployment(record)
        except Exception as exc:  # readiness reports failure; the process stays alive
            state.error = f"{type(exc).__name__}"
        yield

    app = FastAPI(title="NFL AI Offensive Coordinator API", version="1.0.0", lifespan=lifespan)
    app.state.coordinator = state
    app.add_middleware(
        CORSMiddleware, allow_origins=os.environ.get("COORDINATOR_WEB_ORIGIN", "http://localhost:3000").split(","),
        allow_methods=["GET", "POST"], allow_headers=["content-type"],
    )

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        if int(request.headers.get("content-length", "0") or 0) > MAX_BODY_BYTES:
            return _error(413, "BODY_TOO_LARGE", "Request body exceeds 16 KB.")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
        return _error(422, "INVALID_REQUEST", "Request failed validation.", fields)

    @app.exception_handler(DomainError)
    async def domain_handler(request: Request, exc: DomainError):
        return _error(exc.status, exc.code, exc.message, exc.fields)

    @app.exception_handler(Exception)
    async def internal_handler(request: Request, exc: Exception):
        return _error(500, "INTERNAL_ERROR", "Unexpected server error.")

    def dep() -> Deployment:
        if state.deployment is None:
            raise DomainError(503, "SERVICE_UNAVAILABLE", "No validated deployment is loaded.")
        return state.deployment

    def _rate_limited(client: str) -> bool:
        now = time.monotonic()
        with state.lock:
            q = state.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_LIMIT_PER_MINUTE:
                return True
            q.append(now)
        return False

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if state.deployment is None:
            return JSONResponse(status_code=503, content={"status": "not_ready", "error": state.error})
        return {"status": "ready", "deployment_id": state.record["deployment_id"],
                "model_bundle_id": state.deployment.bundle.bundle_id, "snapshot_ids": list(state.deployment.snapshots)}

    @app.get("/v1/deployment")
    def deployment():
        d = dep()
        return {
            "record": state.record,
            "snapshots": [{"snapshot_id": s.snapshot_id, "cutoff_at": s.manifest["cutoff_at"],
                           "temporal_evidence_mode": s.temporal_mode, "actual_seasons": s.manifest["actual_seasons"],
                           "audit_status": s.manifest["audit_status"], "row_counts": s.manifest["row_counts"],
                           "attributions": sorted({a["attribution"] for a in s.manifest["assets"]})}
                          for s in d.snapshots.values()],
            "model_bundle_id": d.bundle.bundle_id, "teams": list(config.TEAMS),
        }

    @app.get("/v1/capabilities")
    def get_capabilities(season: int, defense: str, cutoff: str | None = None, snapshot_id: str | None = None):
        return capabilities(dep(), season, defense, parse_utc(cutoff) if cutoff else None, snapshot_id)

    @app.get("/v1/defenses/{team}/profile")
    def profile(team: str, season: int, cutoff: str | None = None, snapshot_id: str | None = None,
                down: int | None = Query(default=None, ge=1, le=4),
                distance_band: str | None = Query(default=None, pattern="^(short|medium|long|very_long)$"),
                field_zone: str | None = Query(default=None, pattern="^(goal_line|red_zone|open_field|backed_up)$"),
                half: int | None = Query(default=None, ge=1, le=2)):
        d = dep()
        if team not in config.TEAMS:
            raise DomainError(422, "UNKNOWN_TEAM", "Unknown team.", ["team"])
        snap = resolve_snapshot(d, snapshot_id)
        cut = parse_utc(cutoff) if cutoff else snap.cutoff_at
        if cut > snap.cutoff_at:
            raise DomainError(422, "CUTOFF_AFTER_SNAPSHOT", "cutoff is later than the snapshot cutoff.", ["cutoff"])
        b = d.builders[snap.snapshot_id]
        cur = observed_tendencies(b.frame, team, season, cut, down, distance_band, field_zone, half)
        prev = observed_tendencies(b.frame, team, season - 1, cut, down, distance_band, field_zone, half)
        return {"snapshot_id": snap.snapshot_id, "cutoff_at": cut.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "temporal_evidence_mode": snap.temporal_mode, "defense": team, "season": season,
                "current_season": cur, "previous_season": prev,
                "smoothed": smoothed_estimates(b, team, season, cut)}

    @app.post("/v1/recommendations")
    async def recommendations(request: Request):
        client = request.client.host if request.client else "unknown"
        if _rate_limited(client):
            return _error(429, "RATE_LIMITED", f"Limit is {RATE_LIMIT_PER_MINUTE} recommendations per minute.")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(422, "INVALID_JSON", "Body is not valid JSON.")
        try:
            req = RecommendationRequest.model_validate(body)
        except Exception as exc:
            fields = [".".join(str(p) for p in e["loc"]) for e in getattr(exc, "errors", lambda: [])()]
            return _error(422, "INVALID_REQUEST", "Request failed validation.", fields,
                          request_id=body.get("request_id") if isinstance(body, dict) else None)
        d = dep()
        payload = req.as_contract_dict()
        key_src = {k: v for k, v in payload.items() if k != "request_id"}
        key = hashlib.sha256(json.dumps({**key_src, "contract": config.RESPONSE_CONTRACT_VERSION},
                                        sort_keys=True).encode()).hexdigest()
        with state.lock:
            cached = state.cache.get(key)
            if cached is not None:
                state.cache.move_to_end(key)
        if cached is not None:
            result = {**cached, "request_id": payload["request_id"]}
        else:
            result = recommend(d, payload)
            errors = sorted(response_validator().iter_errors(result), key=lambda e: e.path)
            if errors:
                return _error(500, "CONTRACT_VIOLATION", "Response failed contract validation.",
                              [".".join(str(p) for p in errors[0].path)], payload["request_id"])
            with state.lock:
                state.cache[key] = result
                if len(state.cache) > CACHE_SIZE:
                    state.cache.popitem(last=False)
        _log_prediction(state.record, payload, result)
        return result

    @app.get("/v1/models/{bundle_id}/card")
    def card(bundle_id: str):
        from coordinator.reports import model_card
        try:
            return model_card(bundle_id)
        except FileNotFoundError:
            raise DomainError(404, "UNKNOWN_BUNDLE", "Unknown model bundle.", ["bundle_id"]) from None

    return app


def _log_prediction(record: dict | None, payload: dict, result: dict) -> None:
    config.PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = {
        "request_id": payload["request_id"], "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deployment_id": (record or {}).get("deployment_id"), "scenario": {k: v for k, v in payload.items() if k != "request_id"},
        "status": result["status"], "reason_codes": result["reason_codes"],
        "recommended_candidate_id": result["recommended_candidate_id"],
        "scores": {c["candidate_id"]: c["metrics"]["expected_epa"] for c in result["candidates"]},
    }
    with (config.PREDICTION_LOG_DIR / f"{day}.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


app = None if os.environ.get("COORDINATOR_NO_DEFAULT_APP") else create_app()
