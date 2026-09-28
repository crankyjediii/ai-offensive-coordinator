"""Model bundle: an immutable directory binding Model A/B (or baselines), preprocessing,
calibrators, policies and training provenance. Loading verifies every file digest."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from coordinator import config
from coordinator.models.calibration import sigmoid, softmax
from coordinator.models.nets import A_HEADS, B_EVENTS, DefenseSequenceModel, OutcomeModel
from coordinator.training.data import Standardizer


class BundleIntegrityError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_bundle(spec: dict, files: dict[str, object]) -> str:
    """files: name -> torch state_dict (``.pt``) or sklearn estimator (``.joblib``)."""
    staging = config.BUNDLES_DIR / f".staging_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, obj in files.items():
        path = staging / name
        if name.endswith(".pt"):
            torch.save(obj, path)
        else:
            joblib.dump(obj, path)
        digests[name] = sha256_file(path)
    spec = {**spec, "files": digests}
    core = json.dumps(spec, sort_keys=True, default=str).encode()
    bundle_id = f"bundle_{spec['bundle_name']}_{hashlib.sha256(core).hexdigest()[:12]}"
    spec["model_bundle_id"] = bundle_id
    (staging / "bundle.json").write_text(json.dumps(spec, indent=2, default=str))
    final = config.BUNDLES_DIR / bundle_id
    if final.exists():
        for p in staging.iterdir():
            p.unlink()
        staging.rmdir()
    else:
        os.replace(staging, final)
    return bundle_id


@dataclass
class LoadedBundle:
    spec: dict
    a_models: list[DefenseSequenceModel]
    b_models: list[OutcomeModel]
    b_sklearn: object | None
    a_std: Standardizer | None
    b_std: Standardizer

    @property
    def bundle_id(self) -> str:
        return self.spec["model_bundle_id"]

    # ---- Model A
    def predict_defense(self, store, a_data) -> dict[str, np.ndarray] | None:
        """Calibrated head probabilities (ensemble mean), or None when A is prior-based."""
        if not self.a_models:
            return None
        from coordinator.training.fit import predict_a_logits
        temps = self.spec["model_a"]["temperatures"]
        acc = {h: [] for h in A_HEADS}
        for m in self.a_models:
            logits = predict_a_logits(m, store, a_data, torch.device("cpu"))
            for h in A_HEADS:
                acc[h].append(softmax(logits[h] / temps.get(h, 1.0)))
        return {h: np.mean(v, axis=0) for h, v in acc.items()}

    # ---- Model B
    def predict_outcomes(self, b_data, action: np.ndarray) -> dict[str, np.ndarray]:
        """Returns per-member EPA means (M, N) and calibrated event probabilities (N, E)."""
        platt = self.spec["model_b"]["platt"]
        if self.b_models:
            from coordinator.training.fit import predict_b
            members = [predict_b(m, b_data, action) for m in self.b_models]
            epa = np.stack([p["epa_mean"] for p in members])
            logits = np.mean([p["event_logits"] for p in members], axis=0)
        else:
            x = np.concatenate([b_data.num, np.eye(2)[action], b_data.sit_cat], axis=1)
            est = self.b_sklearn
            epa = est["epa"].predict(x)[None, :]
            raw = np.stack([est[e].predict_proba(x)[:, 1] for e in B_EVENTS], axis=1)
            raw = np.clip(raw, 1e-6, 1 - 1e-6)
            logits = np.log(raw) - np.log(1 - raw)
        probs = np.stack([sigmoid(platt[e][0] * logits[:, j] + platt[e][1]) for j, e in enumerate(B_EVENTS)], axis=1)
        probs[action == 0, B_EVENTS.index("sack")] = 0.0  # structural zero for a verified designed rush
        return {"epa_members": epa, "event_probs": probs}


def load_bundle(bundle_id: str) -> LoadedBundle:
    path = config.BUNDLES_DIR / bundle_id
    spec_path = path / "bundle.json"
    if not spec_path.exists():
        raise FileNotFoundError(bundle_id)
    spec = json.loads(spec_path.read_text())
    for name, digest in spec["files"].items():
        if sha256_file(path / name) != digest:
            raise BundleIntegrityError(f"digest mismatch for {name} in {bundle_id}")
    a_models, b_models, b_sklearn = [], [], None
    ma = spec["model_a"]
    for name in ma.get("member_files", []):
        m = DefenseSequenceModel(ma["n_query_num"], arch=ma["arch"], max_len=ma["history_length"],
                                 use_position=ma.get("use_position", True), prior_offset=ma.get("prior_offset", False))
        m.load_state_dict(torch.load(path / name, weights_only=True))
        a_models.append(m.eval())
    mb = spec["model_b"]
    if mb["kind"] == "mlp":
        for name in mb["member_files"]:
            m = OutcomeModel(mb["n_num"], hidden=mb.get("hidden", 128))
            m.load_state_dict(torch.load(path / name, weights_only=True))
            b_models.append(m.eval())
    else:
        b_sklearn = joblib.load(path / mb["member_files"][0])
    a_std = Standardizer.from_dict(ma["standardizer"]) if ma.get("standardizer") else None
    return LoadedBundle(spec, a_models, b_models, b_sklearn, a_std, Standardizer.from_dict(mb["standardizer"]))


def list_bundles() -> list[str]:
    if not config.BUNDLES_DIR.exists():
        return []
    return sorted(p.name for p in config.BUNDLES_DIR.iterdir() if (p / "bundle.json").exists())
