"""Immutable raw-asset retrieval.

Each retrieval stores the exact bytes under ``data/raw/{provider}/{dataset}/{retrieval_id}/`` with a
``retrieval.json`` record. Re-downloading identical bytes records no new logical asset.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from coordinator import config
from coordinator.data.sources import NFLVERSE_API, SOURCES, SourceSpec

USER_AGENT = "ai-offensive-coordinator/0.1 (portfolio research)"


@dataclass(frozen=True)
class SourceAsset:
    source_asset_id: str
    provider: str
    dataset: str
    season: int | None
    url: str
    sha256: str
    size_bytes: int
    retrieved_at: str
    source_published_at: str | None
    source_available_basis: str
    license_url: str
    attribution: str
    path: str

    @staticmethod
    def load(path: Path) -> SourceAsset:
        return SourceAsset(**json.loads(path.read_text()))


def utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_asset_times(spec: SourceSpec) -> dict[str, str]:
    """Upload time of each current release asset, from the GitHub release API."""
    resp = requests.get(
        f"{NFLVERSE_API}/{spec.release_tag}", headers={"User-Agent": USER_AGENT}, timeout=30
    )
    if resp.status_code != 200:
        return {}
    return {a["name"]: a["updated_at"] for a in resp.json().get("assets", [])}


def _get_with_retries(url: str, attempts: int = 4) -> bytes:
    delay = 2.0
    for i in range(attempts):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            resp.raise_for_status()
            return resp.content
        except FileNotFoundError:
            raise
        except requests.RequestException:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def existing_assets(dataset: str | None = None) -> list[SourceAsset]:
    out = []
    for rec in sorted(config.RAW_DIR.glob("*/*/*/retrieval.json")):
        asset = SourceAsset.load(rec)
        if dataset is None or asset.dataset == dataset:
            out.append(asset)
    return out


def latest_asset(dataset: str, season: int | None) -> SourceAsset | None:
    matches = [a for a in existing_assets(dataset) if a.season == season]
    return max(matches, key=lambda a: a.retrieved_at) if matches else None


def fetch(dataset: str, season: int | None) -> tuple[SourceAsset, bool]:
    """Download one asset. Returns (asset, is_new). Identical bytes reuse the prior record."""
    spec = SOURCES[dataset]
    url = spec.url(season)
    content = _get_with_retries(url)
    digest = hashlib.sha256(content).hexdigest()
    for prior in existing_assets(dataset):
        if prior.season == season and prior.sha256 == digest:
            return prior, False

    published = _release_asset_times(spec).get(spec.asset_name(season))
    retrieved_at = utcnow()
    retrieval_id = f"{retrieved_at.replace(':', '').replace('-', '')}_{digest[:12]}"
    out_dir = config.RAW_DIR / spec.provider / dataset / retrieval_id
    out_dir.mkdir(parents=True, exist_ok=False)
    file_path = out_dir / spec.asset_name(season)
    file_path.write_bytes(content)
    asset = SourceAsset(
        source_asset_id=f"{dataset}_{season or 'all'}_{digest[:16]}",
        provider=spec.provider,
        dataset=dataset,
        season=season,
        url=url,
        sha256=digest,
        size_bytes=len(content),
        retrieved_at=retrieved_at,
        source_published_at=published,
        source_available_basis="github_release_asset_updated_at" if published else "ingested_at",
        license_url=spec.license_url,
        attribution=spec.attribution,
        path=str(file_path.relative_to(config.DATA_DIR)),
    )
    (out_dir / "retrieval.json").write_text(json.dumps(asdict(asset), indent=2))
    return asset, True


def asset_path(asset: SourceAsset) -> Path:
    return config.DATA_DIR / asset.path


def verify(asset: SourceAsset) -> bool:
    return hashlib.sha256(asset_path(asset).read_bytes()).hexdigest() == asset.sha256
