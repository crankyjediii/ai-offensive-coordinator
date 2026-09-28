"""Immutable deployment records and the atomic ``active`` pointer."""

from __future__ import annotations

import hashlib
import json
import os
import time

from coordinator import config
from coordinator.data.snapshot import load_snapshot
from coordinator.models.bundle import load_bundle


def create_record(bundle_id: str, snapshot_ids: list[str], image_digest: str = "local-dev") -> dict:
    bundle = load_bundle(bundle_id)  # verifies digests
    snaps = [load_snapshot(s) for s in snapshot_ids]
    for s in snaps:
        if not s.manifest["promotion_allowed"]:
            raise RuntimeError(f"snapshot {s.snapshot_id} failed its audit and cannot be deployed")
        if s.manifest["taxonomy_version"] != bundle.spec["taxonomy_version"]:
            raise RuntimeError(f"taxonomy mismatch between {s.snapshot_id} and {bundle_id}")
    cap_digest = hashlib.sha256(json.dumps([s.manifest["capabilities"] for s in snaps], sort_keys=True).encode()).hexdigest()
    rec = {
        "application_image_digest": image_digest,
        "model_bundle_id": bundle_id,
        "snapshot_ids": snapshot_ids,
        "feature_schema_version": bundle.spec["feature_schema_version"],
        "taxonomy_version": bundle.spec["taxonomy_version"],
        "outcome_cohort_id": bundle.spec["outcome_cohort_id"],
        "response_contract_version": bundle.spec["response_contract_version"],
        "capability_manifest_digest": cap_digest,
        "evaluation_report_id": bundle.spec["evaluation_report_id"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rid = "deploy_" + hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:12]
    rec["deployment_id"] = rid
    config.DEPLOYMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DEPLOYMENTS_DIR / f"{rid}.json").write_text(json.dumps(rec, indent=2))
    return rec


def activate(deployment_id: str) -> None:
    path = config.DEPLOYMENTS_DIR / f"{deployment_id}.json"
    if not path.exists():
        raise FileNotFoundError(deployment_id)
    active = config.DEPLOYMENTS_DIR / "active.json"
    prev = json.loads(active.read_text())["deployment_id"] if active.exists() else None
    tmp = active.with_suffix(".tmp")
    tmp.write_text(json.dumps({"deployment_id": deployment_id, "previous": prev,
                               "activated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    os.replace(tmp, active)


def rollback() -> str:
    active = json.loads((config.DEPLOYMENTS_DIR / "active.json").read_text())
    if not active.get("previous"):
        raise RuntimeError("no previous deployment to roll back to")
    activate(active["previous"])
    return active["previous"]


def active_record() -> dict:
    ptr = json.loads((config.DEPLOYMENTS_DIR / "active.json").read_text())
    return json.loads((config.DEPLOYMENTS_DIR / f"{ptr['deployment_id']}.json").read_text())
