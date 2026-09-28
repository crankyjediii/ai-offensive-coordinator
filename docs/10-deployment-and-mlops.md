# Deployment, reproducibility, and MLOps

This is an operating specification for the proposed project. No model has been trained, deployed, or measured as part of this documentation task. Start with local CPU inference and a reproducible offline pipeline; add hosting only after the data, contracts, and evaluation gates pass. [Engineering architecture](07-engineering-and-stack.md) defines the service boundaries, while [Evaluation and reliability](06-evaluation-and-reliability.md) defines statistical release criteria.

## Environments and release unit

Use development, staging, and a public portfolio demo. Development permits experiments against explicit snapshots. Staging runs the release image with representative fixtures and a candidate bundle. The public demo serves an approved deployment manifest and exposes its cutoff, mode, and limitations.

The release unit is a tuple:

```text
(application_image_digest, model_bundle_id, snapshot_id,
 feature_schema_version, taxonomy_version, outcome_cohort_id, response_contract_version,
 capability_manifest_digest, evaluation_report_id)
```

Store this tuple in a small immutable deployment record. A mutable `active` pointer selects that record. Each response includes the resolved bundle and snapshot IDs so users can reproduce a result after promotion. Updating data alone is still a release because capabilities and scenario outputs may change. Do not update preprocessing, calibration, or one ensemble member independently under an existing bundle ID.

## Initial deployment topology

Run the Python API as one CPU process with the model loaded once at startup. Mount immutable Parquet, model, and metadata artifacts read-only. A separate CLI job refreshes data and builds snapshots. Keep the local MLflow tracking store and artifact directory outside the public service. The frontend may run as a Node process or static client that calls the API; select one after confirming required framework features. Next.js documents both server deployment and static export, with limits on server-dependent features in the latter. [Next.js deployment](https://nextjs.org/docs/app/getting-started/deploying)

A single small hosted machine can later run the API and reverse proxy; the frontend may be separately hosted. Use an object store or private artifact repository for durable manifests and files. A managed SQL database, Redis, task queue, Kubernetes, and always-on GPU are optional responses to measured needs. None is required for the first portfolio release.

The API process does not own refresh or training schedules. Configure an external scheduler after the refresh command is idempotent. For an initial in-season schedule, propose one nightly refresh plus an operator-triggered run after a provider correction. This is a project schedule, not a claim about source publication timing. A refresh with no newly published eligible rows should complete without creating a semantically different snapshot.

## Resource assumptions and measurement plan

Begin local development on a machine with roughly 16 GB RAM and 20 GB of free working storage. These are planning allowances, not dataset size measurements. Inspect downloaded file sizes before committing to multi-season charting or tracking data. Partition by dataset and season, prune unused columns, and bound sequence materialization.

For hosted inference, start benchmarking with two CPU cores and 4 GB RAM, one worker, a bounded queue, and a three-seed ensemble using the proposed 128-snap, width-128, two-layer configuration. Measure actual peak resident memory, cold load, and sustained inference before buying capacity. If this footprint fails, reduce duplicate feature work, batch candidates, or use the validated baseline. Any reduced model requires its own evaluation.

Proposed performance acceptance targets, measured after warmup on the documented deployment hardware:

| Workload | Initial target | Test conditions |
|---|---|---|
| Warm recommendation | p95 under 1 second | 500 mixed supported requests, no cache, concurrency 1 |
| Cached recommendation | p95 under 200 ms | Same corpus, populated local cache |
| Small concurrent demo | p95 under 2 seconds | Five clients, fixed bounded queue, all failures reported |
| Cold readiness | Under 30 seconds | Empty process, artifacts already on local disk |

These are targets rather than benchmarks. Report sample size, CPU, memory, operating system, thread settings, image digest, and whether network time is included. First measure with synchronous bounded inference; additional workers replicate model memory and must be budgeted. Set a request timeout with an explicit 503 or timeout error; do not substitute an unvalidated answer to meet latency.

A proposed configuration surface:

```yaml
service:
  device: cpu
  workers: 1
  max_inflight: 5
  timeout_seconds: 10
  rate_limit_per_minute: 30
model:
  history_length: 128
  ensemble_members: 3
  deterministic_inference: true
data:
  require_immutable_snapshot: true
  allow_unverified_availability_for_strict_replay: false
explanations:
  engine: template
```

Treat this as an example configuration contract, not an executable service file. Separate secrets from versioned configuration. Keep support and abstention thresholds in the evaluated model policy, not arbitrary runtime settings that an operator can relax unnoticed.

## Reproducible experiments

Each experiment records a Git commit, dirty-tree flag, config digest, interpreter and lockfile, operating system, hardware, PyTorch/device runtime, random seeds, data snapshot hashes, fold definitions, feature/taxonomy versions, sampler settings, optimizer, checkpoint-selection rule, calibration split, and evaluation report.

Seed Python, NumPy, PyTorch, data-loader workers, and randomized estimators. Record deterministic-algorithm settings and known exceptions. PyTorch states that complete reproducibility is not guaranteed across releases, platforms, or CPU/GPU execution even with identical seeds; the project should promise reproducibility within a documented environment and numerical tolerance. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

Track model parameters, metrics, artifacts, and input manifests in MLflow. Log the entire ensemble under one parent run with child member runs. Store the best checkpoint selected by the predeclared validation rule, final checkpoint, feature statistics, and calibration artifacts. Preserve the candidate pool and failed experiments relevant to model selection so the chosen result is not presented without context.

Use immutable artifact IDs for serving. MLflow aliases can organize candidate and champion versions, but resolve an alias to an exact bundle before deployment; a mutable alias must not alter an in-flight request. MLflow documents aliases and tags as its current model organization workflow. [MLflow Model Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

## Data refresh procedure

1. Acquire a refresh lease so only one writer runs. Resolve allowed source URLs and credentials from approved configuration.
2. Discover changed provider releases using checksums or supported metadata. Store downloaded bytes, retrieval time, and source availability evidence.
3. Validate raw schemas, parse types, and quarantine unknown fields or labels requiring a mapping decision. Do not coerce a new coverage class into a familiar class silently.
4. Normalize and join into staging. Check duplicate keys, one-to-many expansion, team aliases, roster/coaching effective dates, and per-source row coverage.
5. Build as-of features using the declared cutoff. Recompute capability manifests and compare label coverage, missingness, joins, and revisions with the previous snapshot.
6. Run a dry run that produces a manifest and difference report without changing the active deployment. Smoke-test compatible models and representative scenarios.
7. Publish the immutable snapshot after validation. If it is approved for serving, create a new deployment record and atomically move `active`.

Retries use source object checksums and an operation ID to avoid duplicate ingestion. Failures leave the previous deployment intact. A partially written snapshot is never visible to readers. Unknown historical availability prevents a claim of strict historical replay; a reconstructed dataset remains explicitly labeled as a revised-data historical test.

Late corrections create a new revision and may change labels as well as features. Preserve the original forward prediction log, then report performance against the chosen revised-label snapshot and record that choice. Do not rewrite the prediction as though the corrected value had been known earlier.

## CI and release gates

Pull-request CI installs locked dependencies, runs lint/type checks, validates schema examples, tests feature chronology and taxonomy behavior, runs a tiny CPU train/save/load/infer integration fixture, checks generated API client consistency, and builds the web app. Fixture data should be synthetic or redistribution-permitted. A complete season backtest belongs in a separately triggered evaluation workflow with an artifact budget.

Before model promotion, require an approved data/cohort audit, leakage checks, chronological holdout results, calibration diagnostics, subgroup support, candidate-overlap checks, and documented abstention behavior. Use the metrics and predeclared thresholds in the evaluation document; a passing build cannot establish model quality. A transformer that loses to the chosen baseline may remain a research artifact while the baseline serves the demo.

Before service promotion, require artifact checksum verification, API compatibility, training-serving feature parity, representative 200/422/409/429/503 cases, missing-data behavior, inference measurements, and a rollback rehearsal. Do not use the locked final test repeatedly as a tuning set. After a failed final test, label any subsequent changes exploratory and obtain a new untouched evaluation window before making new confirmation claims.

The maintainer records a release decision with exact artifacts, comparison to the current version, known regressions, and accepted scope. In this one-person project, that record can be a reviewed pull request and signed release note; no artificial committee is needed.

## Model card and prediction log

Each model card must state intended use, unsupported use, modes and teams supported, target definitions, cohort sizes, excluded plays, data rights, availability policy, training/validation/test dates, feature list, baseline comparisons, hyperparameters, calibration method, interval meaning, subgroup results, abstention coverage, inference environment, and unresolved limitations. Include the coordinator-prior methodology and whether actual current-season charting is present. Bind the primary outcome cohort as `reg_scrimmage_penalty_free_v1` and state that the expected-EPA comparison is conditional on eligible penalty-free plays. Penalty occurrence is unknown at decision time, so this restricted estimand is not full-call policy value. Publish the penalty-inclusive sensitivity analysis separately.

For every forward prediction retain request ID, normalized scenario or its reproducible reference, time generated, cutoff, deployment tuple, permitted candidate set, scores, selected candidate, reason codes, and evidence summaries. Save feature fingerprints and sufficient trace metadata to diagnose discrepancies without logging unnecessary personal data or licensed raw records. Keep raw client IPs out of the research prediction table. A proposed operational-log retention is 30 days; preserve approved aggregate metrics and research traces according to license terms and storage budget.

## Monitoring and trigger policy

Monitor service and data separately from predictive performance. Service metrics cover latency, queue length, memory, cold starts, error rates, and cache hits. Data metrics cover last successful refresh, provider changes, missing fields, unmatched joins, new categories, per-team counts, and publication delays. Model metrics cover abstention rates, selected-candidate mix, ensemble spread, out-of-distribution scores, and calibration once labels become available.

A change in input distribution is an investigation trigger, not proof that accuracy fell. A new season, coordinator change, or missing source can affect different fields differently. Examine per-mode and per-team trends alongside sample sizes. Outcome monitoring evaluates predictions for the action actually observed; it cannot score the unplayed recommendation as though its counterfactual result were known.

Initial operational triggers are proposed: any checksum or temporal-eligibility violation blocks publication; a source-required field disappearing disables that capability; five consecutive health failures trigger service investigation; repeated timeout/error spikes over a 15-minute window trigger resource review. Statistical deterioration triggers use the evaluated control policy and minimum label counts, not an invented universal weekly threshold. Small rare-event cohorts should accumulate evidence before declaring calibration failure.

## Incident runbooks

**Provider delay or outage.** Keep the last valid snapshot, show its cutoff and freshness, and disable features whose declared freshness policy expires. Retry with bounded backoff. Do not relabel stale data as current. Record provider status and the earliest eligible refresh opportunity.

**Schema change or broken join.** Quarantine the new ingestion, compare changed fields and row counts, update the adapter in a branch, and rerun contract and cohort checks. Publish a new schema version if meaning changed. The existing deployment remains pinned throughout.

**Corrupted or incompatible bundle.** Fail readiness before accepting traffic. Restore the last verified deployment tuple. Check all file hashes and load the saved compatibility report before restarting. A missing ensemble member is a different model and cannot be ignored silently.

**Suspected leakage or invalid target definition.** Stop serving the affected model or switch to a separately validated baseline. Mark impacted cards/reports withdrawn, identify affected snapshots and folds, rebuild features, and rerun evaluation. Preserve the incident record; do not silently replace reported metrics.

**Bad rollout.** Move `active` to the previous complete deployment record, restart or atomically swap the loaded bundle, invalidate only caches keyed to the failed deployment, and run readiness plus known-scenario checks. Because cache keys include exact IDs, old valid responses remain interpretable. Proposed recovery objective: restore the last valid demo within one working hour; this is an operating target, not an availability guarantee.

**Restore test.** At least before the first hosted release, recover the deployment record, artifact files, and tracking metadata into a clean directory. Verify hashes and reproduce a saved synthetic or permitted scenario. Backup copies without a restore rehearsal do not demonstrate recoverability.

## Budget controls and portfolio evidence

Treat costs as owner-selected ceilings, not provider quotes. A starting monthly planning envelope is $0–40 for small CPU hosting and storage, up to $50 for optional burst training, and $0 for text-model calls while templates are used. A $100 total alert threshold leaves limited headroom; actual feasibility depends on measured artifact size, training time, traffic, region, and current prices. Licensed charting is outside this envelope and requires a separate decision before use.

Tag jobs with experiment IDs and record wall time, CPU/GPU hours, storage growth, and estimated billable usage. Cap search runs, stop failed training early, and delete expendable intermediates only after preserving reproducible inputs and final artifacts. Keep model checkpoints and raw provider archives according to rights and retention needs. Do not maintain an idle GPU for a model that can meet its inference target on CPU.

The portfolio should show one reproducible release, a chronological evaluation report, a source-availability audit, a readable model card, a recorded rollback exercise, and a short demonstration of honest abstention. Publish measured latency and model metrics only after the corresponding runs exist. Documenting a limitation and enforcing it in the application is stronger evidence than an unverified accuracy claim.

**Source verification:** official technical references checked September 27, 2026. Resource sizes, performance thresholds, schedules, and budget ceilings are proposed starting assumptions.
