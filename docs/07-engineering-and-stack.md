# Engineering architecture and technology stack

This document specifies the proposed software boundaries, storage design, API, and development workflow. The application has not been implemented. The starter contracts included with this suite describe intended behavior; their synthetic examples are not model results. Data definitions live in [Data sources and schema](02-data-sources-and-schema.md), model behavior in [ML architecture](04-ml-architecture.md), and operating procedures in [Deployment and MLOps](10-deployment-and-mlops.md).

## System boundary

Build one Python package, one web application, and an offline pipeline. Keep training and ingestion outside request handling. A request must never download a season, train a model, infer unobserved current-snap coverage from an outcome, or quietly switch to a newer snapshot.

```text
Approved data adapters → raw immutable objects → validation/quarantine
    → canonical Parquet tables → as-of feature snapshots → training/backtests
    → model bundle + calibration + capability manifest

Next.js web app → FastAPI → snapshot resolver → capability/candidate checks
    → Model A or available baseline → Model B → reliability gates
    → ranked broad candidates + evidence + fixed factual explanation
```

The MVP supports pregame scouting and hypothetical situations from a selected as-of snapshot. An entered quarter and clock describe the hypothetical decision; they do not turn the product into a live game feed. Historical replay uses an explicit game identifier and a separately constructed eligible snapshot. See [Evaluation and reliability](06-evaluation-and-reliability.md) for replay restrictions.

## Stack and installation gate

| Component | Proposed choice | Responsibility and acceptance check |
|---|---|---|
| Runtime | Python 3.13, provisional | Install every required package and complete one CPU train/save/load/infer cycle before accepting the version |
| Ingestion and transforms | nflreadpy, Polars, NumPy | Source adapters return typed tables; transforms preserve source and availability metadata |
| Analytical storage | Parquet and DuckDB | Columnar snapshots, SQL quality checks, offline scouting aggregates |
| Statistical baseline | scikit-learn and SciPy | Calibrated classifiers, regression, uncertainty and diagnostic utilities |
| Neural models | PyTorch | Structured sequence encoders, masked multitask heads, deterministic inference configuration |
| Service | FastAPI and Pydantic | Strict request validation, response serialization, generated OpenAPI |
| Interface | Next.js, React, TypeScript, Recharts | Scenario forms, scouting charts, accessible evidence displays |
| Development | uv, pytest, Ruff, mypy, pre-commit | Locked environment, meaningful tests, formatting and type checks |
| Automation and tracking | GitHub Actions, MLflow | CI, repeatable experiments, model artifacts and promotion records |

The table is a selection, not a claim that an exact combination has been tested. Record a compatible version matrix for macOS development and Linux CPU deployment. Test CUDA separately if training uses it. If Python 3.13 blocks a required dependency, choose a supported interpreter and record that decision before creating the initial lockfile. Do not mix GPU runtime requirements into a CPU serving image without need.

Commit `pyproject.toml`, `uv.lock`, the interpreter version, a frontend package lock, and a Node version file. Use `uv sync --locked` in CI so dependency changes cannot silently update the lock. uv documents that `--locked` fails when the lockfile is inconsistent with project metadata. [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)

Use a small Streamlit prototype only if it accelerates early scouting review. Keep all data and inference logic in the package so the final Next.js interface calls the same implementation.

## Repository and dependency rules

```text
ai-offensive-coordinator/
  src/coordinator/
    data/           # ingest, canonicalize, validate, snapshot manifests
    taxonomy/       # versioned labels and mappings
    features/       # as-of joins, priors, opponent adjustment, sequences
    models/         # baselines, transformers, losses, calibration
    training/       # datasets, folds, fit, backtest, experiment runner
    recommendation/ # candidate filtering, scoring, abstention, facts
    analysis/       # defense profiles and report queries
    api/            # schemas, routes, service lifecycle, error mapping
    cli.py
  web/              # application and generated API client types
  configs/          # data, experiment, inference, evaluation policies
  contracts/        # schemas and synthetic fixtures
  tests/            # unit, data contract, integration, replay, UI
  notebooks/        # exploration importing package code
  data/             # ignored raw, staging, canonical, snapshots
  artifacts/        # ignored models and reports; manifests tracked
  docs/             # design, cards, decisions, runbooks
```

`features` may depend on taxonomy and data contracts but cannot call an API route. `models` cannot query live providers. `recommendation` consumes typed feature objects and loaded predictors; it owns score ordering and abstention. `api` translates transport objects and delegates. The web app receives complete results and never independently recalculates EPA utility or statistical confidence. No notebook becomes the sole source of a production transform.

## Immutable data and serving artifacts

Raw ingestion records source URL or provider identifier, content checksum, `ingested_at`, and any verified `source_available_at`. Unknown publication times remain null. Event time alone cannot establish that a field was available to a historical decision. Store corrected provider releases alongside earlier versions; never overwrite them in place.

The ingestion process writes to staging, validates joins and uniqueness, and then publishes a manifest with file checksums. Canonical play identity uses string `game_id` plus the canonical source-mapped play key; do not convert identifiers into floats. A `snapshot_id` identifies an immutable manifest of permitted rows, transformations, feature schema, availability policy, and cutoff. Production identifiers should derive from a manifest digest or immutable registry record; the `synthetic_` identifiers in this suite are only fixtures.

Every `model_bundle_id` binds Model A and Model B versions or their baseline replacements, preprocessing statistics, categorical vocabularies, taxonomy, ensemble members, calibration artifacts, abstention policy, training manifest, feature schema, and `outcome_cohort_id`. The primary cohort is `reg_scrimmage_penalty_free_v1`. A bundle cannot accept a snapshot merely because the field names look similar. Check a compatibility matrix and reject unsupported taxonomy versions or missing required features.

Serve read-only materialized snapshots. A separate job owns writes and atomically changes a small deployment manifest after checks pass. DuckDB documents a distinction between one process performing reads/writes and multiple processes reading in read-only mode; this design avoids relying on multiple API workers writing the same database. [DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency)

Use explicit migrations such as `canonical_v1` to `canonical_v2`: rebuild into a new location, compare counts and meaning, validate downstream features, and retain the earlier snapshot. An additive nullable column still requires a feature-policy decision. A renamed or redefined label requires a new taxonomy/schema version and retraining or a verified adapter.

## Capability and candidate resolution

Compute capabilities for each source combination, season, defense, cutoff, and snapshot. The manifest declares usable inputs, available targets, candidate families, label counts, freshness, and permitted use. The same team can support broad recommendations in one snapshot and historical coverage analysis in another.

`pbp_baseline` permits `designed_rush` and `dropback` only after the intended-action mapping passes the data audit. It does not expose mesh, flood, inside zone, or outside zone. `charted_history` permits only audited historical attributes. `enriched_current` requires a verified lawful and timely source for its additional fields. The latter is a capability target, not a promised launch feature.

Client fields are requests to condition the model, not proof of data support. Reject a submitted hash, personnel group, formation, or motion flag when the selected bundle lacks a validated feature for it. Do not silently ignore the field. Exclude candidates unavailable to the offense or training cohort, record why, then apply overlap and reliability checks. If no candidate remains, return `unavailable`; if score estimates exist but evidence cannot justify choosing one, return `abstained`.

## API and wire contracts

| Route | Inputs | Result |
|---|---|---|
| `GET /v1/capabilities` | season, defense, cutoff; optional exact snapshot | Exact selected snapshot, usable modes/inputs/candidates, unavailable reasons |
| `GET /v1/defenses/{team}/profile` | season, cutoff, snapshot, supported filters | Observed counts, smoothed estimates, coordinator-prior components, denominators |
| `POST /v1/recommendations` | Validated scenario and exact snapshot/bundle IDs | Candidate scores, selection status, data evidence, model evidence |
| `GET /v1/models/{id}/card` | Immutable bundle ID | Cohorts, metrics, intended use, calibration, caveats, provenance |
| `GET /health/live` and `GET /health/ready` | None | Process alive; then exact bundle/snapshot loaded and validated |

The recommendation [request schema](../contracts/recommendation-request.schema.json) and [response schema](../contracts/recommendation-response.schema.json) use JSON Schema Draft 2020-12. Enable format checking in validators; a `format` annotation alone is not sufficient in every implementation. [JSON Schema 2020-12](https://json-schema.org/draft/2020-12)

The [request example](../contracts/recommendation-request.example.json) is a hypothetical CHI offense against GB in 2026: second-and-seven, 38 yards from the defense's goal line, Q2 with 582 seconds remaining. It uses a synthetic September 27 cutoff, zero score differential, and null `game_id`; it makes no schedule claim. Left hash and 11 personnel from the original concept are omitted because broad mode has not established those capabilities. The [response example](../contracts/recommendation-response.example.json) contains fictional numbers solely to illustrate serialization.

Required inputs include `mode`, `season`, `cutoff_at`, `snapshot_id`, `model_bundle_id`, offense, defense, and situation. UTC timestamps end in `Z`. A hypothetical `game_id` is null; a replay identifier is a string and must resolve to the selected season and teams. `yardline_100` is distance to the defense's goal, so GB 38 is 38 when CHI has possession. The initial contract covers regulation quarters only; overtime needs a later rules-aware schema extension.

Pydantic validates numeric types strictly and forbids unknown properties. FastAPI supports Pydantic request models; Pydantic's strict mode reduces type coercion, with documented JSON-specific exceptions. Add semantic validators for UTC, offense different from defense, cutoff compatibility, goal-to-go consistency, season/team eligibility, and optional personnel counts. Frontend checks improve feedback but do not replace server validation. [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/), [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)

Response status has four meanings:

- `ok`: requested supported scope is available and the ranking passes the release policy.
- `limited`: a useful permitted ranking exists with explicit limitations, such as broad-only candidates.
- `abstained`: predictions may be shown, but no recommended candidate is selected.
- `unavailable`: required data or a compatible model is absent; candidates are empty.

The latter two require a null recommendation and at least one reason code. A nullable outcome metric means unavailable or not estimated; zero is a numerical result. Sack probability for a verified designed run is zero by the action definition, whereas an unestimated pass sack probability is null. Every nullable outcome metric and interval has an accompanying reason code. Probabilities apply within the candidate, scenario, and declared outcome cohort; they are not additionally conditioned on completion or survival without a sack.

Return `outcome_cohort_id` in every response and label the primary estimate “expected EPA on eligible penalty-free plays.” Penalty occurrence is unknown before the decision. Excluding penalty plays therefore defines a restricted comparison, not full-call policy value. The penalty-inclusive sensitivity cohort needs its own version and evaluation before it can replace the primary cohort.

An interval object names its quantity, method, nominal coverage, and calibration artifact. `epa_prediction_interval` concerns the outcome of a future play; it is not a confidence interval for mean EPA or a statement about causal benefit. Seed disagreement has a separate field. Cross-field checks enforce ordered bounds and consistency with the selected calibration artifact. The schema catches shape errors; application validators enforce football and statistical semantics.

## Request execution, errors, and caching

Resolve and pin IDs once at request start. Load context, validate capabilities, construct a shared history representation, batch-score eligible candidates, apply calibrated transformations and abstention, and serialize. Use stable candidate-ID ordering to break exact score ties. Run models in evaluation mode with gradient recording disabled. An optional text model receives only approved facts after scoring; deterministic templates remain the default explanation.

Cache by a digest of canonical normalized input, cutoff, snapshot, bundle, utility-policy version, and response-contract version. Include optional inputs and missingness explicitly. Do not round yard line or distance to increase hits unless the model itself defines those buckets. Cache eviction affects speed only; cached evidence must remain bound to its IDs. An exact historical request must never read a cache entry produced with “latest.”

Use 422 for malformed or unsupported input, 404 for unknown IDs, 409 for known incompatible snapshot/bundle pairs, 429 for rate limits, and 503 for service unavailability. Domain abstention is a successful 200 response. Error bodies contain `request_id`, a stable `code`, safe `message`, and affected `fields`; never expose filesystem paths, credentials, or traces. Apply a proposed 16 KB body limit, 30 recommendations per minute per client, and a bounded inference queue; benchmark before changing these defaults.

## Security and implementation checks

Use read-only public routes; keep ingestion, registry, and deployment operations private. Restrict CORS to the actual interface origin, serve HTTPS when hosted, parameterize SQL, and allowlist query filters. Do not accept arbitrary model paths, provider URLs, SQL, serialized Python objects, or user-uploaded checkpoints. Load only trusted artifacts with verified digests. Minimize request logs and apply a defined retention period.

Test behavior at boundaries: unavailable source, unknown team, cutoff violation, duplicate play mapping, future feature injection, missing required input, unsupported optional field, null metric, unstable rank, cache-key changes, and restart with a corrupted artifact. A training-serving parity fixture must yield the same features through both code paths. Contract tests should validate JSON fixtures and reject invalid ones; integration tests should prove a refresh cannot change a response pinned to an older snapshot.

**Source verification:** official technical references checked September 27, 2026. Package versions, hosting choices, and measured performance remain to be established during implementation.
