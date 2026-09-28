"""Recommendation service: capability checks, candidate scoring, evidence gates and abstention.

The service never downloads data, trains models or silently switches snapshots. Every response
is bound to the exact snapshot and bundle IDs it was computed from.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
import yaml

from coordinator import config
from coordinator.data.snapshot import Snapshot, capability_policy, parse_utc
from coordinator.features.builder import FeatureBuilder, situation_arrays
from coordinator.features.context import PriorParams, StrengthParams, make_contexts
from coordinator.models.bundle import LoadedBundle
from coordinator.models.nets import B_EVENTS
from coordinator.recommendation.explain import explanation_text
from coordinator.taxonomy.vocab import ACTION_LABELS, ACTIONS, COVERAGE_CLASSES
from coordinator.training.fit import AData, BData, HistoryStore

BOX_REPRESENTATIVE = np.array([4, 5, 6, 7, 8, 9], dtype=float)  # overflow buckets use their bound
MIN_ACTION_SHARE = 0.05
MIN_RANK_MARGIN_EPA = 0.02
MAX_SUPPORTED_DISTANCE = 25
UNSUPPORTED_INPUT_FIELDS = ("hash", "offensive_personnel", "formation", "motion")


class DomainError(Exception):
    """Maps to an HTTP error. code is stable; fields lists offending request fields."""

    def __init__(self, status: int, code: str, message: str, fields: list[str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.fields = status, code, message, fields or []


@dataclass
class Deployment:
    snapshots: dict[str, Snapshot]
    bundle: LoadedBundle
    builders: dict[str, FeatureBuilder]
    head_validation: dict[str, str]

    @staticmethod
    def create(snapshots: list[Snapshot], bundle: LoadedBundle) -> Deployment:
        pp = PriorParams(**bundle.spec["prior_params"])
        sp = StrengthParams(**bundle.spec["strength_params"])
        return Deployment({s.snapshot_id: s for s in snapshots}, bundle,
                          {s.snapshot_id: FeatureBuilder(s, pp, sp) for s in snapshots},
                          head_validation(bundle.bundle_id))

    def head_passed(self, head: str) -> bool:
        return self.head_validation.get(head) == "passed"


def head_validation(bundle_id: str) -> dict[str, str]:
    """Heads without a recorded 'passed' decision are never displayed as predictions."""
    path = config.CONFIGS_DIR / "head_validation.yaml"
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return (data.get(bundle_id) or {}).get("heads", {})


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_snapshot(dep: Deployment, snapshot_id: str | None) -> Snapshot:
    if snapshot_id is None:
        return max(dep.snapshots.values(), key=lambda s: s.manifest["cutoff_at"])
    if snapshot_id not in dep.snapshots:
        from coordinator.data.snapshot import list_snapshots
        if snapshot_id in list_snapshots():
            raise DomainError(409, "SNAPSHOT_NOT_DEPLOYED", "Snapshot exists but is not served by this deployment.", ["snapshot_id"])
        raise DomainError(404, "UNKNOWN_SNAPSHOT", "Unknown snapshot_id.", ["snapshot_id"])
    return dep.snapshots[snapshot_id]


def capabilities(dep: Deployment, season: int, defense: str, cutoff: datetime | None, snapshot_id: str | None) -> dict:
    snap = resolve_snapshot(dep, snapshot_id)
    cutoff = cutoff or snap.cutoff_at
    if cutoff > snap.cutoff_at:
        raise DomainError(422, "CUTOFF_AFTER_SNAPSHOT", "cutoff_at is later than the snapshot cutoff.", ["cutoff_at"])
    if defense not in config.TEAMS:
        raise DomainError(422, "UNKNOWN_TEAM", "Unknown defense team.", ["defense"])
    policy = capability_policy()
    cap = snap.capabilities_for(season)
    frame = dep.builders[snap.snapshot_id].frame
    known = frame.filter((pl.col("season") == season) & (pl.col("defense_id") == defense)
                         & (pl.col("known_at_pbp") <= cutoff) & (pl.col("completed_at_est") < cutoff))
    modes = {}
    bundle_modes = set(dep.bundle.spec["modes"])
    for mode, mspec in policy["modes"].items():
        reasons = []
        if mode not in bundle_modes:
            reasons.append("MODEL_UNAVAILABLE")
        if not mspec["candidates"]:
            reasons.append(mspec.get("unavailable_reason", "MISSING_REQUIRED_CAPABILITY"))
        if cap is None:
            reasons.append("SNAPSHOT_UNAVAILABLE")
        heads = {}
        for h in mspec["defense_heads"]:
            hs = (cap or {}).get("heads", {}).get(h, {})
            ok = hs.get("status") == "available"
            if h in ("coverage", "man_zone", "pressure"):
                # Coverage is forecast from labels published by the cutoff (prior seasons only in-season).
                ok = _participation_known(frame, cutoff)
            reason = None if ok else "COVERAGE_LABELS_UNAVAILABLE"
            validated = dep.head_passed(h) if h not in ("five_plus_rushers",) else dep.head_passed("blitz")
            if ok and not validated:
                ok, reason = False, "MODEL_UNAVAILABLE"
            heads[h] = {"status": "available" if ok else "unavailable", "source": hs.get("source"),
                        "season_completeness": hs.get("completeness"), "validation": dep.head_validation.get(h, "not_validated"),
                        "reason": reason}
        modes[mode] = {
            "status": "unavailable" if reasons else "available",
            "reason_codes": sorted(set(reasons)),
            "candidates": mspec["candidates"] if not reasons else [],
            "optional_inputs_supported": [],
            "defense_heads": heads,
        }
    return {
        "snapshot_id": snap.snapshot_id, "cutoff_at": _iso(cutoff), "snapshot_cutoff_at": snap.manifest["cutoff_at"],
        "temporal_evidence_mode": snap.temporal_mode, "season": season, "defense": defense,
        "model_bundle_id": dep.bundle.bundle_id,
        "defense_current_season_eligible_plays": known.height,
        "defense_current_season_games": known["game_id"].n_unique(),
        "latest_eligible_event_at": _iso(known["completed_at_est"].max()) if known.height else None,
        "season_capabilities": cap,
        "modes": modes,
        "unsupported_optional_inputs": list(UNSUPPORTED_INPUT_FIELDS),
    }


def _participation_known(frame: pl.DataFrame, cutoff: datetime) -> bool:
    return frame.filter((pl.col("known_at_part") <= cutoff) & pl.col("t_coverage").is_not_null()).height > 0


def _validate(req: dict, dep: Deployment) -> tuple[Snapshot, datetime]:
    if req["offense_team"] == req["defense_team"]:
        raise DomainError(422, "SAME_TEAM", "Offense and defense must differ.", ["offense_team", "defense_team"])
    if req["model_bundle_id"] != dep.bundle.bundle_id:
        from coordinator.models.bundle import list_bundles
        if req["model_bundle_id"] in list_bundles():
            raise DomainError(409, "INCOMPATIBLE_BUNDLE", "Bundle is not deployed with this snapshot set.", ["model_bundle_id"])
        raise DomainError(404, "UNKNOWN_BUNDLE", "Unknown model_bundle_id.", ["model_bundle_id"])
    snap = resolve_snapshot(dep, req["snapshot_id"])
    cutoff = parse_utc(req["cutoff_at"])
    if cutoff > snap.cutoff_at:
        raise DomainError(422, "CUTOFF_AFTER_SNAPSHOT", "cutoff_at is later than the snapshot cutoff.", ["cutoff_at"])
    if req["season"] not in snap.manifest["actual_seasons"]:
        raise DomainError(422, "SEASON_NOT_IN_SNAPSHOT", "Season is not covered by the snapshot.", ["season"])
    sit = req["situation"]
    if sit["yards_to_go"] > sit["yardline_100"]:
        raise DomainError(422, "IMPOSSIBLE_SITUATION", "yards_to_go cannot exceed yardline_100.", ["situation.yards_to_go"])
    if sit["goal_to_go"] != (sit["yards_to_go"] == sit["yardline_100"]):
        raise DomainError(422, "GOAL_TO_GO_INCONSISTENT",
                          "goal_to_go must be true exactly when yards_to_go equals yardline_100.", ["situation.goal_to_go"])
    opt = req.get("optional_inputs") or {}
    bad = [f"optional_inputs.{k}" for k in opt if k in UNSUPPORTED_INPUT_FIELDS]
    if bad:
        raise DomainError(422, "UNSUPPORTED_OPTIONAL_INPUT",
                          "The selected bundle has no validated feature for these inputs.", bad)
    unknown_cands = [c for c in req["candidate_ids"] if c not in ACTIONS]
    if unknown_cands:
        raise DomainError(422, "UNVALIDATED_CANDIDATE",
                          f"Candidates without a validated action definition: {', '.join(unknown_cands)}.", ["candidate_ids"])
    if req["scenario_kind"] == "historical_replay":
        g = snap.table("game").filter(pl.col("game_id") == req["game_id"])
        if g.height == 0:
            raise DomainError(404, "UNKNOWN_GAME", "game_id is not in the snapshot.", ["game_id"])
        row = g.row(0, named=True)
        if {row["home_team_id"], row["away_team_id"]} != {req["offense_team"], req["defense_team"]} or row["season"] != req["season"]:
            raise DomainError(422, "GAME_MISMATCH", "game_id does not match the season and teams.", ["game_id"])
        if cutoff >= row["kickoff_at"]:
            raise DomainError(422, "CUTOFF_NOT_PREGAME", "Replay cutoff must precede kickoff.", ["cutoff_at"])
    return snap, cutoff


def _metrics_obj(mu: float, probs: np.ndarray, action: str, half_width: float | None, cal_id: str,
                 coverage: float) -> dict:
    missing = {"expected_yards": "METRIC_NOT_ESTIMATED"}
    m = {
        "expected_epa": round(float(mu), 4),
        "expected_yards": None,
        **{f"{e}_probability": round(float(probs[j]), 4) for j, e in enumerate(B_EVENTS)},
    }
    if half_width is None or not np.isfinite(half_width):
        m["epa_prediction_interval"] = None
        missing["epa_prediction_interval"] = "INTERVAL_NOT_CALIBRATED"
    else:
        m["epa_prediction_interval"] = {
            "quantity": "single_play_epa", "lower": round(float(mu - half_width), 4), "upper": round(float(mu + half_width), 4),
            "nominal_coverage": coverage, "method": "split_conformal_residual", "calibration_artifact_id": cal_id,
        }
    m["missing_reasons"] = missing
    return m


def recommend(dep: Deployment, req: dict) -> dict:
    snap, cutoff = _validate(req, dep)
    bundle = dep.bundle
    builder = dep.builders[snap.snapshot_id]
    policy = capability_policy()
    mode = req["mode"]
    mspec = policy["modes"][mode]
    sit = req["situation"]
    offense, defense, season = req["offense_team"], req["defense_team"], req["season"]
    base = {
        "schema_version": "1.0.0", "request_id": req["request_id"], "is_synthetic_example": False,
        "example_notice": None, "mode": mode, "cutoff_at": req["cutoff_at"], "snapshot_id": snap.snapshot_id,
        "model_bundle_id": bundle.bundle_id, "outcome_cohort_id": bundle.spec["outcome_cohort_id"],
    }
    frame = builder.frame
    known = frame.filter((pl.col("known_at_pbp") <= cutoff) & (pl.col("completed_at_est") < cutoff))
    d_season = known.filter((pl.col("season") == season) & (pl.col("defense_id") == defense))
    last_ingested = max(a["retrieved_at"] for a in snap.manifest["assets"])
    availability_basis = snap.temporal_mode
    data_evidence = {
        "eligible_defense_play_count": d_season.height, "effective_sample_size": None,
        "latest_eligible_event_at": _iso(d_season["completed_at_est"].max()) if d_season.height else None,
        "last_ingested_at": last_ingested, "availability_basis": availability_basis,
        "source_ids": sorted({a["source_asset_id"] for a in snap.manifest["assets"] if a["season"] in (season, season - 1, None)}),
        "unavailable_inputs": list(UNSUPPORTED_INPUT_FIELDS), "reason_codes": [],
    }
    model_evidence = {
        "evaluation_report_id": bundle.spec["evaluation_report_id"], "calibration_artifact_id": bundle.spec["calibration_artifact_id"],
        "ranking_policy_id": bundle.spec["ranking_policy_id"],
        "ensemble_members": max(1, len(bundle.b_models)), "ranking_passed_policy": False, "reason_codes": [],
    }

    if not mspec["candidates"] or mode not in bundle.spec["modes"]:
        code = mspec.get("unavailable_reason", "MODEL_UNAVAILABLE")
        return {**base, "status": "unavailable", "reason_codes": [code], "recommended_candidate_id": None, "candidates": [],
                "defense_expectation": {"status": "unavailable", "coverage_probabilities": None, "blitz_probability": None,
                                        "expected_box_count": None, "reason_codes": [code]},
                "data_evidence": {**data_evidence, "reason_codes": [code]},
                "model_evidence": {**model_evidence, "reason_codes": [code]},
                "explanation": {"engine": "template", "text": explanation_text("unavailable", mode=mode, reasons=[code])}}
    if mode == "charted_history" and not _participation_known(frame, cutoff):
        code = "COVERAGE_LABELS_UNAVAILABLE"
        return {**base, "status": "unavailable", "reason_codes": [code, "MISSING_REQUIRED_CAPABILITY"],
                "recommended_candidate_id": None, "candidates": [],
                "defense_expectation": {"status": "unavailable", "coverage_probabilities": None, "blitz_probability": None,
                                        "expected_box_count": None, "reason_codes": [code]},
                "data_evidence": {**data_evidence, "reason_codes": [code]},
                "model_evidence": {**model_evidence, "reason_codes": [code]},
                "explanation": {"engine": "template", "text": explanation_text("unavailable", mode=mode, reasons=[code])}}

    candidates = [c for c in req["candidate_ids"] if c in mspec["candidates"]]
    ctx = make_contexts([(defense, offense, season, cutoff)])
    cf = builder.context_features(ctx)
    rows = pl.DataFrame({
        "context_idx": [0], "quarter": [sit["quarter"]], "seconds_remaining": [sit["seconds_remaining_in_quarter"]],
        "down": [sit["down"]], "distance": [sit["yards_to_go"]], "yards_to_goal": [sit["yardline_100"]],
        "offense_score_margin": [sit["score_differential"]], "goal_to_go": [sit["goal_to_go"]],
    })
    sit_cat, _ = situation_arrays(rows)
    cell = int(sit_cat[0, 0] * 4 + sit_cat[0, 1])
    zone = int(sit_cat[0, 2])

    # ---- defense expectation
    fp = cf.support["faced_dropback"][0]
    data_evidence["effective_sample_size"] = round(float(fp[2]), 2)
    de_reasons: list[str] = []
    probs_a = None
    if bundle.a_models:
        inp = builder.inputs(rows, cf, bundle.spec["model_a"]["feature_contract"])
        q = bundle.a_std.transform(np.concatenate([inp.sit_num, inp.ctx_num], axis=1))
        hist = builder.histories(ctx, bundle.spec["model_a"]["history_length"])
        store = HistoryStore(hist, torch.device("cpu"))
        from coordinator.training.experiment import PRIOR_HEAD_FOR_A
        prior_logp = {h: np.log(np.clip(cf.priors[PRIOR_HEAD_FOR_A[h]][[0], cell], 1e-6, 1.0)).astype(np.float32)
                      for h in PRIOR_HEAD_FOR_A}
        probs_a = bundle.predict_defense(store, AData(np.array([0]), inp.sit_cat, q, {}, prior_logp))
        box = probs_a["box"][0]
        blitz = float(probs_a["blitz"][0, 1])
        cov = probs_a["coverage"][0]
    else:
        box = cf.priors["box"][0, cell]
        blitz = float(cf.priors["blitz"][0, cell, 1])
        cov = cf.priors["coverage"][0, cell]
    ftn_ok = all((snap.capabilities_for(season) or {}).get("heads", {}).get(h, {}).get("status") == "available"
                 and dep.head_passed(h) for h in ("box_count", "blitz"))
    coverage_probs = None
    if mode == "charted_history" and dep.head_passed("coverage"):
        coverage_probs = {c: round(float(p), 4) for c, p in zip(COVERAGE_CLASSES, cov / cov.sum(), strict=True)}
    elif mode == "charted_history":
        de_reasons.append("MODEL_UNAVAILABLE")  # coverage head failed locked-test validation
    else:
        de_reasons.append("COVERAGE_LABELS_UNAVAILABLE")
    if cf.support["box"][0, 1] == 0:
        de_reasons.append("PRIOR_DOMINATED")
    defense_expectation = {
        "status": "predicted" if ftn_ok else "unavailable",
        "coverage_probabilities": coverage_probs if ftn_ok else None,
        "blitz_probability": round(blitz, 4) if ftn_ok else None,
        "expected_box_count": round(float((box * BOX_REPRESENTATIVE).sum()), 3) if ftn_ok else None,
        "reason_codes": sorted(set(de_reasons if ftn_ok else de_reasons + ["MISSING_REQUIRED_CAPABILITY"])),
    }

    # ---- outcome scoring
    inp_b = builder.inputs(rows, cf, bundle.spec["model_b"]["feature_contract"])
    x = bundle.b_std.transform(np.concatenate([inp_b.sit_num, inp_b.ctx_num], axis=1))
    n = len(candidates)
    b_data = BData(np.repeat(inp_b.sit_cat, n, 0), np.repeat(x, n, 0), np.zeros(n, dtype=np.int64),
                   np.zeros(n, dtype=np.float32), np.zeros((n, len(B_EVENTS)), dtype=np.float32))
    act_idx = np.array([ACTIONS.index(c) for c in candidates], dtype=np.int64)
    out = bundle.predict_outcomes(b_data, act_idx)
    mu = out["epa_members"].mean(0)
    sd = out["epa_members"].std(0) if out["epa_members"].shape[0] > 1 else np.full(n, np.nan)
    hw = bundle.spec["model_b"]["conformal_half_width"]
    cov_nom = bundle.spec["model_b"]["conformal_nominal_coverage"]

    # ---- support and overlap (comparable league snaps known by the cutoff)
    comp = known.filter(pl.col("in_reg_scrimmage_penalty_free_v1") & (pl.col("cell_idx") == cell) & (pl.col("zone_idx") == zone))
    cell_total = comp.height
    reasons: set[str] = set()
    cand_out = []
    for i, c in enumerate(candidates):
        cc = comp.filter(pl.col("action") == c)
        share = cc.height / cell_total if cell_total else 0.0
        cr: list[str] = []
        if cc.height < policy["support"]["min_comparable_snaps"] or cc["game_id"].n_unique() < policy["support"]["min_distinct_games"]:
            cr.append("LOW_SUPPORT")
        if share < MIN_ACTION_SHARE:
            cr.append("LOW_OVERLAP")
        if mode == "pbp_baseline":
            cr.append("BROAD_CANDIDATES_ONLY")
        reasons.update(cr)
        dd = d_season.filter((pl.col("action") == c) & (pl.col("cell_idx") == cell))
        cand_out.append({
            "candidate_id": c, "label": ACTION_LABELS[c], "rank": None, "utility_epa_units": round(float(mu[i]), 4),
            "metrics": _metrics_obj(mu[i], out["event_probs"][i], c, hw.get(c), bundle.spec["calibration_artifact_id"], cov_nom),
            "support": {"observed_play_count": cc.height, "effective_sample_size": None,
                        "scope": (f"League penalty-free {ACTION_LABELS[c].lower()} plays in the same down, distance band and "
                                  f"field zone known by the cutoff ({cc['game_id'].n_unique()} games; {share:.0%} of comparable snaps). "
                                  f"This defense faced {dd.height} such plays this season.")},
            "ensemble_epa_standard_deviation": None if np.isnan(sd[i]) else round(float(sd[i]), 4),
            "reason_codes": sorted(cr),
        })
    if sit["yards_to_go"] > MAX_SUPPORTED_DISTANCE or cell_total < policy["support"]["min_comparable_snaps"]:
        reasons.add("OUT_OF_DISTRIBUTION")
    if cf.support["faced_dropback"][0, 1] == 0:
        reasons.add("PRIOR_DOMINATED")
    stale_days = policy["support"]["stale_source_days"]
    if d_season.height and (cutoff - d_season["completed_at_est"].max()).days > stale_days:
        reasons.add("STALE_SOURCE")
    if n >= 2:
        order = np.argsort(-mu, kind="stable")
        margin = float(mu[order[0]] - mu[order[1]])
        spread = float(np.sqrt(sd[order[0]] ** 2 + sd[order[1]] ** 2)) if not np.isnan(sd).any() else 0.0
        if margin < max(MIN_RANK_MARGIN_EPA, spread):
            reasons.add("UNSTABLE_RANKING")
    abstain_codes = {"LOW_SUPPORT", "LOW_OVERLAP", "OUT_OF_DISTRIBUTION", "UNSTABLE_RANKING"}
    if n < 2:
        reasons.add("NO_ELIGIBLE_CANDIDATE" if n == 0 else "LOW_SUPPORT")
    abstained = bool(reasons & (abstain_codes | {"NO_ELIGIBLE_CANDIDATE"}))
    status = "abstained" if abstained else ("limited" if reasons else "ok")
    recommended = None
    if not abstained:
        # Deterministic: expected EPA descending, then stable candidate-ID order.
        order = sorted(range(n), key=lambda i: (-mu[i], candidates[i]))
        for r, i in enumerate(order, 1):
            cand_out[i]["rank"] = r
        cand_out = [cand_out[i] for i in order]
        recommended = cand_out[0]["candidate_id"]
        model_evidence["ranking_passed_policy"] = True
    if not any(np.isfinite([hw.get(c, np.nan) for c in candidates])):
        reasons.add("INTERVAL_NOT_CALIBRATED")
    model_evidence["reason_codes"] = sorted(reasons & (abstain_codes | {"INTERVAL_NOT_CALIBRATED"}))
    data_evidence["reason_codes"] = sorted(reasons & {"BROAD_CANDIDATES_ONLY", "STALE_SOURCE", "PRIOR_DOMINATED"} | set(
        ["COVERAGE_LABELS_UNAVAILABLE"] if mode == "pbp_baseline" else []))
    all_reasons = sorted(reasons | ({"COVERAGE_LABELS_UNAVAILABLE"} if mode == "pbp_baseline" else set()))
    text = explanation_text(status, mode=mode, reasons=all_reasons, candidates=cand_out, recommended=recommended,
                            defense=defense, offense=offense, defense_expectation=defense_expectation,
                            cohort=bundle.spec["outcome_cohort_id"], snapshot=snap.snapshot_id)
    return {**base, "status": status, "reason_codes": all_reasons, "recommended_candidate_id": recommended,
            "candidates": cand_out, "defense_expectation": defense_expectation, "data_evidence": data_evidence,
            "model_evidence": model_evidence, "explanation": {"engine": "template", "text": text}}


__all__ = ["Deployment", "DomainError", "recommend", "capabilities", "resolve_snapshot", "dataclasses"]
