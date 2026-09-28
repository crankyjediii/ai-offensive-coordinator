"""Immutable snapshots: a manifest of exact inputs, cutoff, policies and eligible rows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from coordinator import config
from coordinator.data.asof import eligible_versions, known_at_col
from coordinator.data.audit import run_audit
from coordinator.data.canonical import load_policy, read_silver
from coordinator.data.ingest import existing_assets

AUDIT_METHOD_VERSION = "audit_v2_latency_aware"

TABLE_KEYS = {
    "game": ["game_id"],
    "play": ["game_id", "play_id"],
    "charting_observation": ["game_id", "play_id"],
    "participation_observation": ["game_id", "play_id"],
}


def parse_utc(s: str) -> datetime:
    if not s.endswith("Z"):
        raise ValueError("UTC timestamps must end in 'Z'")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def code_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capability_policy() -> dict:
    return yaml.safe_load((config.CONFIGS_DIR / "capability_policy.yaml").read_text())


def season_capabilities(audit: dict) -> list[dict]:
    policy = capability_policy()
    out = []
    for season, entry in audit["seasons"].items():
        fc = entry["field_completeness"]
        heads = {}
        for head, spec in policy["heads"].items():
            field = spec["field"]
            meas = fc.get(field) or {"proportion": None, "numerator": 0, "denominator": 0}
            threshold = policy["min_completeness"].get(field, 1.0)
            ok = meas["proportion"] is not None and meas["proportion"] >= threshold
            heads[head] = {
                "status": "available" if ok else "unavailable",
                "source": spec["source"],
                "timely_in_season": spec["timely"],
                "labeled": meas["numerator"],
                "denominator": meas["denominator"],
                "completeness": meas["proportion"],
                "reason": None if ok else "COVERAGE_LABELS_UNAVAILABLE" if head in ("coverage", "man_zone", "pressure") else "MISSING_REQUIRED_CAPABILITY",
            }
        era = entry.get("participation_era") or []
        if "ngs" in era:
            for h in ("coverage", "man_zone", "pressure"):
                heads[h]["status"] = "unavailable"
                heads[h]["reason"] = "COVERAGE_LABELS_UNAVAILABLE"
                heads[h]["note"] = "NGS-era participation labels excluded from charted cohorts (decision D-004)"
        out.append({
            "season": int(season),
            "broad_actions": "available" if entry["eligible_scrimmage_plays"] > 0 else "unavailable",
            "eligible_scrimmage_plays": entry["eligible_scrimmage_plays"],
            "heads": heads,
        })
    return out


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    path: Path
    manifest: dict

    @property
    def cutoff_at(self) -> datetime:
        return parse_utc(self.manifest["cutoff_at"])

    @property
    def temporal_mode(self) -> str:
        return self.manifest["temporal_evidence_mode"]

    def table(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.path / f"{name}.parquet")

    def capabilities_for(self, season: int) -> dict | None:
        return next((c for c in self.manifest["capabilities"] if c["season"] == season), None)


def build_snapshot(cutoff_at: str, temporal_mode: str, requested_seasons: list[int]) -> tuple[Snapshot, bool]:
    """Build (or reuse) the snapshot for these exact inputs. Returns (snapshot, is_new)."""
    cutoff = parse_utc(cutoff_at)
    assets = [a for a in existing_assets() if a.dataset == "schedules" or a.season in requested_seasons]
    eligible_assets = [a for a in assets if datetime.fromisoformat(a.retrieved_at.replace("Z", "+00:00")) <= cutoff
                       or temporal_mode == "reconstructed_conservative_delay"]
    core = {
        "manifest_schema_version": "1.0.0",
        "cutoff_at": cutoff_at,
        "temporal_evidence_mode": temporal_mode,
        "requested_seasons": requested_seasons,
        "availability_policy": load_policy(),
        "capability_policy": capability_policy(),
        "taxonomy_version": config.TAXONOMY_VERSION,
        "feature_schema_version": config.FEATURE_SCHEMA_VERSION,
        "canonical_schema_version": config.SCHEMA_VERSION,
        "audit_method_version": AUDIT_METHOD_VERSION,
        "assets": sorted(
            ({"source_asset_id": a.source_asset_id, "dataset": a.dataset, "season": a.season, "sha256": a.sha256,
              "url": a.url, "retrieved_at": a.retrieved_at, "source_published_at": a.source_published_at,
              "license_url": a.license_url, "attribution": a.attribution} for a in eligible_assets),
            key=lambda d: d["source_asset_id"],
        ),
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    snapshot_id = f"snap_{cutoff.strftime('%Y%m%dT%H%MZ')}_{digest[:12]}"
    path = config.SNAPSHOTS_DIR / snapshot_id
    if (path / "manifest.json").exists():
        return load_snapshot(snapshot_id), False

    asset_ids = {a.source_asset_id for a in eligible_assets}
    games = read_silver("game").filter(pl.col("record_version_id").is_in(asset_ids))
    games = games.filter(pl.col("season").is_in(requested_seasons))
    tables: dict[str, pl.DataFrame] = {"game": games}
    completed = games.select("game_id", "completed_at_est")
    for name in ("play", "charting_observation", "participation_observation"):
        df = read_silver(name).filter(pl.col("record_version_id").is_in(asset_ids))
        if "completed_at_est" not in df.columns:
            df = df.join(completed, on="game_id", how="inner")
        tables[name] = eligible_versions(df, cutoff, temporal_mode, TABLE_KEYS[name]).collect()

    audit = run_audit(games, tables["play"], tables["charting_observation"],
                      tables["participation_observation"], requested_seasons, cutoff=cutoff,
                      charting_latency_hours=load_policy()["sources"]["ftn_charting"]["delay_after_game_end_hours"])
    manifest = {
        **core,
        "snapshot_id": snapshot_id,
        "artifact_kind": "snapshot",
        "manifest_digest": digest,
        "audit_status": "passed" if audit["passed"] else "failed",
        "code_commit": code_commit(),
        "actual_seasons": sorted({int(s) for s in tables["play"]["season"].unique().to_list()}),
        "row_counts": {k: v.height for k, v in tables.items()},
        "known_at_column": known_at_col(temporal_mode),
        "capabilities": season_capabilities(audit),
        "promotion_allowed": audit["passed"],
        "quality_report_path": "audit.json",
    }
    tmp = path.with_name(path.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(tmp / f"{name}.parquet")
    (tmp / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, path)  # atomic publish: readers never see a partial snapshot
    return load_snapshot(snapshot_id), True


def load_snapshot(snapshot_id: str) -> Snapshot:
    path = config.SNAPSHOTS_DIR / snapshot_id
    manifest_file = path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(snapshot_id)
    return Snapshot(snapshot_id, path, json.loads(manifest_file.read_text()))


def list_snapshots() -> list[str]:
    if not config.SNAPSHOTS_DIR.exists():
        return []
    return sorted(p.name for p in config.SNAPSHOTS_DIR.iterdir() if (p / "manifest.json").exists())
