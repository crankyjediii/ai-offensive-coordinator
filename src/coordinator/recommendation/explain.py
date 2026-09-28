"""Deterministic explanation templates. Every number comes from the response payload."""

from __future__ import annotations

REASON_TEXT = {
    "BROAD_CANDIDATES_ONLY": "Only the broad designed-rush versus dropback comparison is supported.",
    "COVERAGE_LABELS_UNAVAILABLE": "Coverage-specific information is unavailable for this snapshot.",
    "LOW_SUPPORT": "At least one candidate has too few comparable historical snaps.",
    "LOW_OVERLAP": "At least one candidate is rarely chosen in this situation, so its estimate extrapolates.",
    "UNSTABLE_RANKING": "The expected-EPA margin is smaller than the ranking policy's minimum or the ensemble spread.",
    "OUT_OF_DISTRIBUTION": "The situation is outside the supported training range.",
    "PRIOR_DOMINATED": "No current-season observations of this defense are available; estimates rely on prior history.",
    "STALE_SOURCE": "The latest eligible observation is older than the freshness policy.",
    "MISSING_REQUIRED_CAPABILITY": "The requested mode has no approved data source in this build.",
    "MODEL_UNAVAILABLE": "No compatible model is deployed for this mode.",
    "INTERVAL_NOT_CALIBRATED": "No calibrated EPA prediction interval is available.",
    "NO_ELIGIBLE_CANDIDATE": "No eligible candidate remains after capability checks.",
}


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "unavailable"
    return f"{x:.1%}" if pct else f"{x:+.3f}"


def explanation_text(status: str, mode: str, reasons: list[str], candidates: list[dict] | None = None,
                     recommended: str | None = None, defense: str | None = None, offense: str | None = None,
                     defense_expectation: dict | None = None, cohort: str | None = None, snapshot: str | None = None) -> str:
    parts: list[str] = []
    if status == "unavailable":
        parts.append(f"No recommendation is available in {mode} mode.")
    elif status == "abstained":
        parts.append("Insufficient evidence to rank these candidates reliably.")
    else:
        parts.append("The available data supports a broad run-versus-dropback comparison on eligible penalty-free plays.")
    for c in candidates or []:
        m = c["metrics"]
        parts.append(
            f"{c['label']}: expected EPA on eligible penalty-free plays {_fmt(m['expected_epa'])}, "
            f"success {_fmt(m['success_probability'], True)}, explosive {_fmt(m['explosive_probability'], True)}, "
            f"sack {_fmt(m['sack_probability'], True)}, turnover {_fmt(m['turnover_probability'], True)} "
            f"({c['support']['observed_play_count']} comparable league snaps)."
        )
    if recommended and candidates:
        top = next(c for c in candidates if c["candidate_id"] == recommended)
        parts.append(f"{top['label']} ranks first by model-estimated expected EPA{f' against {defense}' if defense else ''}.")
    if defense_expectation and defense_expectation.get("status") == "predicted":
        parts.append(
            f"Model forecast for this defense: expected box count {defense_expectation['expected_box_count']}, "
            f"blitz probability on a dropback {_fmt(defense_expectation['blitz_probability'], True)}."
        )
    if cohort:
        parts.append("Penalty occurrence is unknown before a call, so this restricted comparison is not full-call policy value.")
    for r in reasons:
        if r in REASON_TEXT:
            parts.append(REASON_TEXT[r])
    parts.append("These are model estimates from observational data, not measured effects of an unplayed call.")
    return " ".join(parts)[:3000]
