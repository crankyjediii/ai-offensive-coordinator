"""Model card (JSON for the API, Markdown for the repository) generated from recorded evidence."""

from __future__ import annotations

import json

from coordinator import config
from coordinator.data.snapshot import load_snapshot

INTENDED_USE = ("Pregame scouting and hypothetical-situation exploration: describe a defense's supported tendencies "
                "and compare broad designed-rush versus dropback families by model-estimated outcomes on eligible "
                "penalty-free regular-season scrimmage plays.")
UNSUPPORTED_USE = [
    "Live in-game play calling or consumption of a live stadium feed.",
    "Named concept recommendations (mesh, flood, inside zone, etc.); no validated labels exist.",
    "Claims of improved scoring, wins added or causal effects of an unplayed call.",
    "Betting or wagering decisions.",
    "Coach or player skill rankings.",
]


def model_card(bundle_id: str) -> dict:
    path = config.BUNDLES_DIR / bundle_id / "bundle.json"
    if not path.exists():
        raise FileNotFoundError(bundle_id)
    spec = json.loads(path.read_text())
    rep_path = config.EXPERIMENTS_DIR / spec["experiment_id"] / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    snap = load_snapshot(spec["training_snapshot_id"])
    final = rep.get("final", {})
    return {
        "model_bundle_id": bundle_id,
        "created_at": spec["created_at"],
        "intended_use": INTENDED_USE,
        "unsupported_use": UNSUPPORTED_USE,
        "modes": spec["modes"],
        "candidates": spec["candidates"],
        "outcome_cohort_id": spec["outcome_cohort_id"],
        "outcome_estimand": ("Expected EPA conditional on eligible penalty-free regular-season scrimmage plays. Penalty "
                             "occurrence is unknown at decision time, so this restricted estimand is not full-call policy value."),
        "taxonomy_version": spec["taxonomy_version"],
        "feature_schema_version": spec["feature_schema_version"],
        "training_snapshot_id": spec["training_snapshot_id"],
        "temporal_evidence": snap.temporal_mode,
        "temporal_evidence_note": ("Reconstructed historical test with a conservative delay policy. Retrieved assets are "
                                   "current re-uploads; no archived first-publication evidence exists for 2022-2025, so this "
                                   "is not an archived as-of replay."),
        "availability_policy": snap.manifest["availability_policy"],
        "data_sources": sorted({(a["dataset"], a["attribution"], a["license_url"]) for a in snap.manifest["assets"]}),
        "splits": rep.get("splits"),
        "training_boundary": spec["training_boundary"],
        "calibration_boundary": spec["calibration_boundary"],
        "decision_rule": rep.get("decision_rule"),
        "model_a": {"served": spec["model_a"]["kind"], "selection": rep.get("model_A_selection", {}).get("decision"),
                    "temperatures": spec["model_a"].get("temperatures"), "history_length": spec["model_a"]["history_length"]},
        "model_b": {"served": spec["model_b"]["kind"], "feature_contract": spec["model_b"]["feature_contract"],
                    "selection": rep.get("model_B_selection", {}).get("decision"),
                    "interval": {"method": "split_conformal_residual", "nominal_coverage": spec["model_b"]["conformal_nominal_coverage"],
                                 "half_width_by_action": spec["model_b"]["conformal_half_width"]}},
        "prior_method": spec["prior_params"],
        "coordinator_prior": "No verified playcaller registry is loaded; the coordinator component is unavailable and its weight is redistributed to team and league history.",
        "selection": {
            "model_A": {k: {kk: vv for kk, vv in v.items()} for k, v in rep.get("model_A_selection", {}).get("results", {}).items()},
            "model_A_fit": {k: {kk: vv for kk, vv in v.items() if kk != "epochs"} for k, v in rep.get("model_A_selection", {}).get("fit_logs", {}).items()},
            "model_B": rep.get("model_B_selection", {}).get("results", {}),
        },
        "locked_test_2025": {"model_A": final.get("model_A_test"), "model_B": final.get("model_B_test"),
                             "model_A_calibration": final.get("model_A_test_calibration"),
                             "coverage_accuracy": final.get("model_A_test_coverage_accuracy"),
                             "coverage_majority_class_rate": final.get("model_A_test_coverage_majority_class_rate"),
                             "model_A_members": final.get("model_A_members"), "model_B_members": final.get("model_B_members")},
        "R01_priors": {k: v for k, v in rep.get("R01_priors", {}).items() if k != "grid"},
        "known_limitations": [
            "Coverage labels exist only after each season; in-season coverage is never observed, only forecast from prior seasons.",
            "2022 participation labels (NGS era) are excluded from charted cohorts because their vocabulary and class mix differ.",
            "Broad actions bundle many calls; RPO branches are not separated.",
            "Outcome estimates are observational associations for the chosen action; unchosen actions have no factual outcome.",
            "Team, coach and roster effects are entangled; no causal attribution is made.",
        ],
        "hardware": rep.get("hardware"),
        "runtime_seconds": rep.get("runtime_seconds"),
        "experiment_id": spec["experiment_id"],
        "is_quick_smoke_bundle": spec.get("is_quick_smoke_bundle", False),
    }
