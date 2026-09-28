"""Command-line entry point: ingest, audit, snapshot, profile, train, deploy, serve, recommend."""

from __future__ import annotations

import json
import uuid

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
DEFAULT_SEASONS = "2022,2023,2024,2025,2026"


def _seasons(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


@app.command()
def ingest(seasons: str = DEFAULT_SEASONS) -> None:
    """Download raw assets (idempotent: identical bytes create no new record)."""
    from coordinator.data.ingest import fetch
    for ds in ("schedules", "pbp", "ftn_charting", "participation"):
        for s in [None] if ds == "schedules" else _seasons(seasons):
            try:
                a, new = fetch(ds, s)
                typer.echo(f"{ds:14s} {s!s:5s} {'NEW' if new else 'unchanged'} {a.sha256[:12]} published={a.source_published_at}")
            except FileNotFoundError:
                typer.echo(f"{ds:14s} {s!s:5s} NOT PUBLISHED (capability disabled)")


@app.command()
def build_silver() -> None:
    """Rebuild canonical tables from every retrieved raw version."""
    from coordinator.data.canonical import build_silver as run
    typer.echo(json.dumps(run(), indent=2))


@app.command()
def snapshot(cutoff: str = typer.Option(None, help="UTC cutoff ending in Z; default now"),
             mode: str = "archived_as_of", seasons: str = DEFAULT_SEASONS) -> None:
    """Build an immutable snapshot (reports 'unchanged' when inputs are identical)."""
    from coordinator.data.ingest import utcnow
    from coordinator.data.snapshot import build_snapshot
    snap, new = build_snapshot(cutoff or utcnow(), mode, _seasons(seasons))
    typer.echo(f"{snap.snapshot_id} {'NEW' if new else 'unchanged'} audit={snap.manifest['audit_status']}")


@app.command()
def audit(snapshot_id: str) -> None:
    from coordinator.data.snapshot import load_snapshot
    typer.echo((load_snapshot(snapshot_id).path / "audit.json").read_text())


@app.command()
def profile(snapshot_id: str, defense: str, season: int, cutoff: str = typer.Option(None)) -> None:
    """Print a saved defensive scouting report."""
    from coordinator.analysis.profile import observed_tendencies
    from coordinator.data.snapshot import load_snapshot, parse_utc
    from coordinator.features.builder import FeatureBuilder
    snap = load_snapshot(snapshot_id)
    cut = parse_utc(cutoff) if cutoff else snap.cutoff_at
    rep = observed_tendencies(FeatureBuilder(snap).frame, defense, season, cut)
    typer.echo(f"{defense} {season} as of {cut.isoformat()} | {rep['eligible_plays']} plays, {rep['distinct_games']} games")
    for r in rep["tendencies"]:
        if r["value"] is None:
            val = "suppressed"
        elif r["interval"] is None:
            val = f"{r['value']:.3f} (no interval: <5 games)"
        else:
            val = f"{r['value']:.3f} [{r['interval'][0]:.3f}, {r['interval'][1]:.3f}]"
        typer.echo(f"  {r['label']:40s} {val:30s} n={r['denominator']:5d} missing={r['missing_label_count']}")
    for u in rep["unavailable"]:
        typer.echo(f"  unavailable: {u['field']} ({u['reason']})")


@app.command()
def train(snapshot_id: str, quick: bool = False) -> None:
    """Run the full research experiment and write a model bundle."""
    from coordinator.training.experiment import run
    r = run(snapshot_id, quick=quick)
    typer.echo(f"{r['experiment_id']} -> {r['model_bundle_id']}")


@app.command()
def deploy(bundle_id: str, snapshot_ids: list[str]) -> None:
    """Create an immutable deployment record and atomically activate it."""
    from coordinator.deploy import activate, create_record
    rec = create_record(bundle_id, snapshot_ids)
    activate(rec["deployment_id"])
    typer.echo(f"active -> {rec['deployment_id']}")


@app.command()
def rollback() -> None:
    from coordinator.deploy import rollback as rb
    typer.echo(f"active -> {rb()}")


@app.command()
def card(bundle_id: str) -> None:
    from coordinator.reports import model_card
    typer.echo(json.dumps(model_card(bundle_id), indent=2, default=str))


@app.command()
def recommend(defense: str, offense: str, season: int, down: int, distance: int, yardline_100: int,
              quarter: int = 1, seconds: int = 900, score_diff: int = 0, mode: str = "pbp_baseline") -> None:
    """Score a hypothetical situation with the active deployment."""
    from coordinator.api.app import load_deployment
    from coordinator.recommendation.service import recommend as rec
    dep, _ = load_deployment()
    snap = max(dep.snapshots.values(), key=lambda s: s.manifest["cutoff_at"])
    req = {"schema_version": "1.0.0", "request_id": str(uuid.uuid4()), "scenario_kind": "hypothetical", "mode": mode,
           "season": season, "offense_team": offense, "defense_team": defense, "game_id": None,
           "cutoff_at": snap.manifest["cutoff_at"], "snapshot_id": snap.snapshot_id, "model_bundle_id": dep.bundle.bundle_id,
           "situation": {"quarter": quarter, "seconds_remaining_in_quarter": seconds, "down": down, "yards_to_go": distance,
                         "yardline_100": yardline_100, "score_differential": score_diff, "goal_to_go": distance == yardline_100},
           "candidate_ids": ["designed_rush", "dropback"], "utility": {"policy": "expected_epa"}}
    typer.echo(json.dumps(rec(dep, req), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8010) -> None:
    import uvicorn
    uvicorn.run("coordinator.api.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    app()


