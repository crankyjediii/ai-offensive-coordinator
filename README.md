# NFL AI Offensive Coordinator

A Python/PyTorch project that builds dated defensive scouting reports from real NFL data and compares **designed rush** against **dropback** for a situation you enter. Every result names its snapshot, cutoff and model bundle, and the system declines to rank when evidence is thin.

**v0.1 · September 27, 2026 · implemented vertical slice (Phases 0–9, 11, 12 core).** The numbered documents below remain the design specification; [docs/decisions.md](docs/decisions.md) records where the implementation chose among options or deviated.

## Quick start

```bash
uv sync                                   # Python 3.13, locked dependencies
uv run coordinator ingest                 # raw nflverse assets, SHA-256, idempotent
uv run coordinator build-silver           # canonical tables
uv run coordinator snapshot --mode reconstructed_conservative_delay --cutoff 2026-09-28T01:00:00Z
uv run coordinator snapshot --mode archived_as_of          # current-season serving snapshot
uv run coordinator train <research_snapshot_id>            # ~17 min on an M-series laptop
uv run coordinator deploy <bundle_id> <serving_snapshot_id> <research_snapshot_id>
uv run coordinator serve                  # FastAPI on :8010
(cd web && pnpm install && pnpm dev)      # UI on :3000
```

Other commands: `profile <snapshot> GB 2026` prints a scouting report, `recommend GB CHI 2026 2 7 38 --quarter 2 --seconds 582` scores a situation, `card <bundle>` prints the model card, and `rollback` restores the previous deployment. Tests run with `uv run pytest` (the API suite runs when a local deployment exists).

## Status: implemented, experimental, planned

| Area | Status |
|---|---|
| Ingestion of play-by-play, schedules, FTN charting and participation for 2022–2026, with immutable raw storage, checksums and provider upload times | Implemented |
| Canonical warehouse, audit gates, content-addressed snapshots, latency-aware capabilities per season | Implemented |
| Taxonomy 1.0.0: broad-action classifier, outcome labels, coverage pooling rule, null semantics | Implemented, with fixture tests |
| As-of feature builder shared by training, backtests and the API (priors, lagged strengths, 128-snap histories) | Implemented, with leakage sentinels |
| League/team empirical-shrinkage priors with current-season updating | Implemented. The coordinator component is **unavailable** because no sourced playcaller registry exists (D-011). |
| Model A transformer, pooled and query-only networks; Model B multi-task MLP; linear and gradient-boosting baselines; calibration; split-conformal intervals | Implemented and evaluated |
| Recommendation service: capability checks, support/overlap/OOD/stability gates, abstention, templates | Implemented |
| FastAPI service (strict validation, runtime contract checks, rate limit, cache, prediction log) and deployment records with rollback | Implemented |
| Next.js Analyst / Playcaller / Evaluation / Methodology interface | Implemented |
| Off-policy evaluation (Phase 10), enriched current charting, named concepts, LLM explanations, hosted deployment, MLflow | Planned; not built |

## Measured results

Experiment `exp_20260927T201315`, reconstructed historical test (not an archived as-of replay; D-003). Train on 2023, select on 2024 weeks 1–12, calibrate on 2024 weeks 13–18, report the 2025 regular season once. Apple M-series, 17 minutes.

- **The transformer did not beat the baseline.** The frozen gate required ≥1% improvement on both development windows. The served Model A is the league/team smoothed prior. A follow-up with prior-offset heads (exploratory, D-023) narrowed the transformer's gap from +19.6% to +3.9% log loss but still lost. Shuffling history order did not hurt, so the sequence model is not using snap order (R03).
- **Priors help modestly.** On 2025, the prior mixture beats league-only rates in summed per-head log loss (5.654 vs 5.694). In 2024, early weeks (1–4) improve 0.4–1.4% by head.
- **Outcome model:** gradient boosting on play-by-play features ties a simple situational table on EPA (MSE 1.849 vs 1.850). Lagged offense/defense strength adds nothing measurable yet. Event probabilities are well calibrated (slopes 0.96–1.07).
- **EPA prediction interval:** 90.0% empirical coverage at 90% nominal on 2025 (game-clustered 95% CI 89.7–90.3%).
- **Coverage forecasts fail validation.** Top-1 accuracy is 24.1%, below the majority-class rate of 27.8%, so coverage and man/zone predictions are never displayed (D-024).

These are prediction-quality measurements. They are not evidence that any recommended call would have produced more points.

## Document index

| Document | What it specifies | Start here when |
|---|---|---|
| [01 Product vision](docs/01-product-vision.md) | Context, users, capability levels, scope and requirements | Explaining the project or reviewing its boundaries |
| [02 Data sources and schema](docs/02-data-sources-and-schema.md) | Providers, joins, table grains, timestamps, nulls and source audits | Building ingestion and the warehouse |
| [03 Football ontology](docs/03-football-ontology.md) | Play/action taxonomy, coverage, personnel and label rules | Implementing labels or deciding what is a valid candidate |
| [04 ML architecture](docs/04-ml-architecture.md) | PyTorch models, tensors, losses, stacking, offense adjustment and ranking | Building training and inference |
| [05 Coordinator priors](docs/05-coordinator-priors.md) | Historical mixtures, temporal updates, transfers and uncertainty | Implementing early-season defensive estimates |
| [06 Evaluation and reliability](docs/06-evaluation-and-reliability.md) | Chronological replay, calibration, support, abstention and policy limits | Designing experiments or approving a model |
| [07 Engineering and stack](docs/07-engineering-and-stack.md) | Package structure, storage boundaries, API and implementation checks | Starting the repository and services |
| [08 Roadmap](docs/08-roadmap.md) | Phases 0–14, subphases, dependencies, effort and exit evidence | Creating issues or choosing the next milestone |
| [09 Product UX](docs/09-product-ux.md) | Analyst/Playcaller flows, forms, states, explanations and accessibility | Implementing the interface |
| [10 Deployment and MLOps](docs/10-deployment-and-mlops.md) | Model bundles, CI, refresh, promotion, monitoring and rollback | Preparing a hosted release |
| [11 Research agenda](docs/11-research-agenda.md) | Hypotheses, ablations, experiment records and stop rules | Planning research beyond the baseline |
| [12 Limitations and ethics](docs/12-limitations-and-ethics.md) | Missing evidence, causal limits, licensing and public claims | Reviewing a public release or portfolio description |

The [source register](references/SOURCE-REGISTER.md) collects the primary references and the licenses of the retrieved data assets. [docs/decisions.md](docs/decisions.md) is the implementation decision register, including the Phase 1 audit findings.

## The agreed design

Use a structured-data football transformer, supported by statistical and tree-based baselines. Model A predicts defensive behavior from the situation and admissible historical context. Model B estimates candidate-conditioned outcomes using the same information available at inference. A recommendation service enforces capability, support and reliability rules before ordering eligible choices. A text LLM, if added later, explains verified evidence.

Coordinator priors blend normalized historical league, team and documented playcaller tendencies with current observations. Their influence depends on learned evidence strength and continuity. A coach transfer follows the person only when attribution and label compatibility support it.

The product supports an offense selection as well as a defense selection. It includes lagged offensive strength and opponent adjustment rather than treating a play as equally suitable for every roster.

The MVP uses completed observations at a recorded cutoff. It supports broad play-by-play analysis while detailed coverage and concept work remains dependent on actual labels. Historical charting cannot be presented as contemporaneously observed current-season data.

The initial outcome cohort is `reg_scrimmage_penalty_free_v1`: regular-season eligible scrimmage plays without a flagged penalty. It supports restricted predictive comparisons. Penalties are unknown at decision time, so this is not an estimate of full-call value. Penalty-inclusive analysis is an explicit later gate.

## Decisions that prevent misleading results

- **Timing:** store event, publication, ingestion and prediction times separately. A final historical file cannot prove what existed at an earlier cutoff.
- **Actions:** keep sacks and scrambles in dropback labeling where intent evidence supports it. Keep ambiguous intent unknown.
- **Concepts:** do not infer inside zone from run gap or mesh from one receiver's route.
- **Training:** generate downstream Model A features out of time; never train Model B on privileged current-snap coverage and silently remove it at deployment.
- **Evaluation:** keep chronological holdouts, calibration data and tuning separate. Distinguish factual prediction metrics from alternative-play value.
- **Uncertainty:** separate event probabilities, predictive intervals, model disagreement and data completeness. Do not invent a single confidence percentage.
- **Release:** keep the best-supported baseline if a transformer does not improve validation. A documented negative research result is acceptable.

These decisions refine the original project discussion. They preserve the requested Python/PyTorch direction, all-team selection goal, current-season focus, prior-season coordinator history, conditional defense profiles, multi-task outcomes, two interface modes, ensemble experiments and chronological backtesting.

## Starter contracts

| File | Purpose |
|---|---|
| [Recommendation request schema](contracts/recommendation-request.schema.json) | Strict shape of a saved scenario and its snapshot/bundle references |
| [Recommendation response schema](contracts/recommendation-response.schema.json) | Result status, candidates, evidence, nulls and reliability information |
| [Synthetic request](contracts/recommendation-request.example.json) | CHI versus GB, 2nd and 7, GB 38, Q2 9:42; no actual schedule claim |
| [Synthetic response](contracts/recommendation-response.example.json) | Demonstrates serialization only; numbers are fictional |
| [Synthetic data manifest](contracts/data-manifest.example.json) | Snapshot provenance and capability structure |

Schema validation checks structure. The API also validates every response against the schema at runtime, and it enforces the semantic checks (capability support, identifiers, cutoffs, goal-to-go consistency, unsupported inputs) in `src/coordinator/recommendation/service.py`.

## Review and maintenance

Keep the numbered documents and contracts in the repository. Update them in the same change that alters behavior. A new action definition changes the taxonomy version. A new data field requires a timing and source decision. A different model or calibrator changes the model bundle. A new snapshot never rewrites an old saved result.

Before implementation, review definitions and source access. Before a model release, review chronological evidence, calibration, failure slices and reproducibility. Before publication, review data rights, attribution, security and the wording of claims.

## Context and provenance

The originating conversation is “Develop NFL AI Plan,” supplied with this request. Its original user request and the retrieved architecture discussion informed the suite. The conversation reader capped the long earlier response at 20,000 characters near the roadmap's counterfactual-evaluation phase; the remaining text was not accessible through the signed-out browser. The requested later topics are fully specified here as new design proposals, without attributing invented details or results to that missing tail.

The earlier CHI/GB coverage, EPA, success and confidence numbers were illustrative. This suite does not reuse them as evidence. No assumptions are made about the user's school level, existing engineering experience or actual 2026 coordinator appointments.

Downloaded data, trained weights and generated reports live under git-ignored `data/` and `artifacts/`. The repository contains no licensed NFL data, credentials or paid-source assets.
