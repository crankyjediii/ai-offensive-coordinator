# Development roadmap and delivery plan

Build the project in releases that remain useful if detailed labels or neural-model gains do not materialize. The first release is a reproducible defensive scouting tool. The research release adds the transformer comparison and outcome estimates. The public portfolio release adds a reviewed interface, deployment, and a measured research report.

The schedule below assumes one developer and access to a laptop, with optional rented accelerator time. Effort ranges are planning estimates, not commitments or provider prices. Work includes implementation, review and documentation; acquiring licensed labels or substantial football annotation would add time.

## Delivery sequence and effort

| Phase | Main output | Estimated hours | Dependency |
|---|---|---:|---|
| 0 | Research specification and decision register | 8–12 | None |
| 1 | Audited, versioned warehouse | 24–36 | 0 |
| 2 | Taxonomy and label pipeline | 14–22 | 1 |
| 3 | Defensive scouting reports | 12–18 | 1, 2 |
| 4 | Coordinator prior system | 16–24 | 1, 3 |
| 5 | Temporal baseline suite | 18–28 | 2, 4 |
| 6 | Defense transformer | 24–40 | 5 |
| 7 | Multi-task outcome model | 24–40 | 5, cross-fitted 6 where used |
| 8 | Recommendation and abstention | 16–24 | 7 |
| 9 | Locked evaluation and reliability review | 24–36 | 5–8 |
| 10 | Optional off-policy research | 24–40 | 9 |
| 11 | Web application | 24–40 | Contract work after 3; integration after 8 |
| 12 | Deployment and operations | 20–32 | 9, 11 |
| 13 | Portfolio publication | 8–12 | 12 |
| 14 | Optional enriched-data research | 40–100+ | Independent source/label gate |

Required phases total approximately 232–364 hours. At 15 hours per week this is about 16–25 weeks before contingency. A 20% allowance gives roughly 19–30 weeks. Phases 10 and 14 are excluded. A smaller resume milestone can stop after the baseline scouting release; label it according to what it actually does.

Each task needs an issue with an owner, prerequisites, artifact paths, acceptance criteria and a link to its governing document. For a solo project, owner roles still separate developer, data reviewer and research reviewer responsibilities.

## Phase 0 Research specification

**0.1 Fix the decision time.** Define the MVP as preparation before a game or a saved hypothetical situation using completed, available observations. Specify UTC cutoff semantics, eligible seasons, and how late source releases enter a snapshot.

**0.2 Freeze the questions.** Choose primary Model A and Model B metrics, the incumbent baseline, a minimum practically useful improvement, and allowed model-selection attempts. Write down why each output head belongs in the product.

**0.3 Record scope decisions.** Confirm the broad candidate set, the difference between descriptive labels and callable actions, offensive context, handling of no-plays and penalties, and planned model modes. Set local compute and hosting budgets.

**0.4 Create the experiment ledger.** Assign experiment IDs, chronological folds and untouched test season. Treat 2022–2026 as the desired ingest range; choose the actual common labeled cohort after Phase 1.

**Exit evidence:** a signed-off specification in the repository, an initial risk register, and tests expressed as examples. No claim of a 2026 coverage feed or an exact playbook taxonomy survives without a source.

## Phase 1 Data warehouse

**1.1 Audit access and rights.** Inspect actual assets for each desired season, record provider, dataset, schema, release time evidence, row counts, field null rates, and redistribution requirements. Distinguish FTN charting from participation. If a source is unavailable, record the failed capability and continue with the broad mode.

**1.2 Build source adapters.** Fetch play-by-play first, then schedules/teams/rosters, charting and participation where available. Store original files by content hash. Add retries with bounded backoff, conditional downloads where supported, and a deterministic manifest.

**1.3 Normalize identity and time.** Resolve provider IDs to game/play/player identities using audited mappings. Store source records and canonical records separately. Implement season-versus-calendar-year handling, including January postseason games.

**1.4 Build the canonical warehouse.** Create the grains and key constraints in [Data sources and schema](02-data-sources-and-schema.md). Validate joins before publishing. Make orphaned or contradictory rows inspectable in quarantine.

**1.5 Implement snapshot promotion.** One refresh command builds a candidate snapshot; checks run before an atomic pointer change. A repeated ingest of identical bytes produces no new logical dataset. Archive current-season source versions from this point forward.

**Exit evidence:** one fully reproducible historical slice and one dated current-season audit; unique keys; measured join coverage; a reproducible snapshot ID; failed source refresh leaves the previous snapshot readable.

## Phase 2 Football taxonomy and labeling

**2.1 Encode canonical vocabularies.** Implement personnel, broad action, field location, coverage and pressure vocabularies as versioned definitions with explicit unknown values. Store raw labels alongside mapped values.

**2.2 Implement intent rules.** Establish how a dropback includes sacks and scrambles, how a designed rush is identified, and which ambiguous plays remain unknown. Do not classify only completed passes and ordinary rushes.

**2.3 Build retrospective descriptors.** Add throw depth, target route, run gap, play action and screen labels only where sources support them. Mark their availability class. They do not automatically become candidate actions.

**2.4 Review boundary cases.** Sample ordinary and rare cases for sacks, scrambles, RPOs, kneels, spikes, penalties, laterals, unknown routes and label conflicts. Review raw records against the written rule. Freeze a small gold fixture set with provenance.

**Exit evidence:** taxonomy version, confusion/unknown report, supported action counts by season, and tests that reject unsupported concept inference. A run at tackle is never automatically “outside zone.”

## Phase 3 Defensive scouting engine

**3.1 Create conditional aggregates.** Build down/distance/field/score splits and supported personnel/formation splits. Report known-label denominator separately from eligible plays.

**3.2 Add opponent context.** Build lagged offensive-strength summaries using data strictly before the cutoff. Compare adjusted and raw rates without using end-of-season strength in early-season profiles.

**3.3 Produce saved reports.** Generate one broad profile and one historical charted profile, if available. Include sample counts, dates, source mode, and missing attributes.

**3.4 Establish the first usable release.** Provide a repeatable command and simple report viewer. Review a small set of football questions with a knowledgeable reader.

**Exit evidence:** changing the cutoff removes later games; unsupported charting is not drawn as zero; another person can reproduce the report. This is the first independent portfolio milestone.

## Phase 4 Coordinator priors

**4.1 Build effective-dated coaching records.** Record team, coordinator, actual defensive playcaller when verified, start/end dates, source URL and confidence. Allow unknown and interim assignments.

**4.2 Implement hierarchical counts.** Build normalized league, team and coordinator tendencies. Separate weighted evidence mass from effective sample size. Add transfer and roster-continuity features using only known information.

**4.3 Tune using past seasons.** Compare league-only, prior team, prior coordinator, combined prior and current-only models on early-season prediction. Tune prior strength, history decay and shrinkage on inner folds.

**4.4 Test changes and sparse history.** Simulate a new coordinator with no independent history, a midseason playcaller change and a coordinator move to a new roster.

**Exit evidence:** a saved posterior calculation can be independently recomputed; early-week and full-season ablations are reported; a prior that harms validation is reduced or removed. No automatic Week 4 or Week 10 switch is assumed.

## Phase 5 Baselines

**5.1 Build a temporal dataset interface.** Each row identifies prediction cutoff, observed action, supported labels and snapshot. Feature fitting occurs within the training fold.

**5.2 Establish simple comparators.** Train league/situation frequencies, smoothed team profiles, logistic or multinomial models, and a tree-based model. Add an MLP comparator before attributing improvement to attention.

**5.3 Fit outcome baselines.** Estimate factual EPA and event probabilities conditioned on supported context/action. Evaluate calibration and performance across action cohorts.

**5.4 Set the incumbent.** Select using validation metrics and freeze its configuration before testing. Save failure slices and latency measurements.

**Exit evidence:** one command reproduces model metrics from a snapshot; temporal leakage sentinel tests pass; the simplest useful serving model is selected. This release can support the final app while neural research continues.

## Phase 6 Defense transformer

**6.1 Implement snap encoding.** Categorical embeddings, normalized numeric features, missingness and timing fields produce a fixed-width token. Add game boundaries, relative recency and padding masks.

**6.2 Implement sequence prediction.** Start with 128 prior snaps, width 128, two layers and four attention heads. These are initial experiment defaults. Current-snap realized labels never enter the query.

**6.3 Train and debug.** Check finite losses, gradient flow, small-batch learning and correct mask behavior. Then run historical validation, multiple seeds and comparison with the incumbent.

**6.4 Run architecture ablations.** Compare shorter context, no sequence, no coordinator prior and shuffled historical order. Inspect whether any gain depends on unavailable historical labels.

**Exit evidence:** complete training config, parameter count, hardware/time record and baseline comparison. If the transformer loses, retain it as a documented research result and keep the baseline in production.

## Phase 7 Outcome model

**7.1 Build leakage-safe Model A features.** Generate forward out-of-fold defensive predictions for Model B training. Do not supply fitted-on-the-same-label probabilities to the downstream model.

**7.2 Implement multi-task heads.** Use the canonical event definitions, label masks and action eligibility rules. Fit the EPA head to the estimand that will be displayed; a robust Huber location is not automatically a conditional mean.

**7.3 Add offensive context.** Compare league-average, lagged team and supported player-level inputs. Prevent end-of-season roster/quality information from entering older predictions.

**7.4 Diagnose factual predictions.** Report per-action error, event calibration, residuals by situation and provider-era sensitivity. Check target leakage through air yards, target receiver or realized coverage.

**Exit evidence:** inference and training use the same input contract; unsupported heads are masked; candidate-conditioned outputs are finite and physically sensible. Factual outcome accuracy is reported separately from recommendation value.

## Phase 8 Recommendation engine

**8.1 Generate the feasible candidate set.** Restrict by mode, label support and trained domain. Broad mode has at most the designed-rush and dropback families. Richer choices require an approved action-definition and support gate.

**8.2 Score deterministically.** Use expected EPA by default; specify tie handling and stable candidate ordering. Add user risk weights only with explicit units and tests of their effect.

**8.3 Implement evidence gates.** Combine capability checks, sample support, overlap diagnostics, out-of-distribution rules and ranking stability. Return structured reason codes and a status.

**8.4 Add explanations.** Begin with templates and typed numerical evidence. Test that no explanation introduces an unsupported coverage or play concept.

**Exit evidence:** a reproducible synthetic contract example and a real saved model run; low support produces an abstention; identical saved inputs reproduce output. The synthetic example is never displayed as a trained prediction.

## Phase 9 Historical replay and reliability

**9.1 Freeze evaluation artifacts.** Lock source manifests, outer test season, feature lists, thresholds, model bundles and calibration method. Define what source-availability evidence actually exists.

**9.2 Replay cutoffs.** Iterate game or weekly cutoffs, reconstruct admissible historical context, update only permitted priors and record every prediction before evaluating its target.

**9.3 Distinguish evaluation regimes.** Report archived as-of replay separately from tests on later-corrected historical files. Run conservative source-delay simulations where true snapshots are absent.

**9.4 Evaluate uncertainty.** Check event calibration, predictive-interval coverage where meaningful, ranking stability, abstention coverage and performance among retained examples. Use game-clustered intervals.

**9.5 Apply the release gate.** Compare against the incumbent on the same eligible rows, then examine rare outcomes, early weeks, coordinator changes and missing-label cohorts.

**Exit evidence:** a model card with measured metrics, interval definitions and known failures; independent reproduction of a saved fold; no unsupported claim of causal improvement. Failed gates return the release to the relevant phase.

## Phase 10 Optional counterfactual research

**10.1 Define a narrow policy estimand.** Freeze eligible situations and action space. State the assumptions needed to interpret observational policy values.

**10.2 Fit the behavior policy out of fold.** Measure candidate overlap and propensity distributions. Estimate effective sample size after clipping or trimming.

**10.3 Compare estimators.** Compute direct modeling, importance-weighted and doubly robust estimates with sensitivity analyses and grouped uncertainty.

**10.4 Publish limits.** Show how hidden playbook decisions, injuries and unobserved intent may change the conclusion. Decline a value claim when support fails.

**Exit evidence:** estimator agreement/disagreement, support diagnostics and sensitivity report. This phase is optional and cannot block a truthful baseline or predictive-model release.

## Phase 11 Web app

**11.1 Build the form and capability states.** Implement the selected teams, scenario validation, cutoff and mode controls against contract fixtures.

**11.2 Build Analyst mode.** Add conditional charts, tables, denominators and observed/predicted labels.

**11.3 Build Playcaller mode.** Add supported candidate comparisons, explicit abstention and evidence panels. Keep old results associated with the old scenario while a new request runs.

**11.4 Review accessibility and mobile behavior.** Test keyboard flow, labels, contrast, table access, zoom and small screens.

**11.5 Integrate real inference.** Replace demonstration fixtures with validated model responses. Save one reproducible result and export it.

**Exit evidence:** user acceptance scenarios in [Web app UX](09-product-ux.md) pass; synthetic examples are clearly identified; unavailable data never creates a false zero chart.

## Phase 12 Deployment and MLOps

**12.1 Package inference.** Build a pinned container, verify CPU loading, add readiness checks and bind compatible models, calibration and taxonomy into one bundle.

**12.2 Build the refresh job.** Schedule ingest and validation independently from retraining. Publish a new snapshot only on success.

**12.3 Add promotion and rollback.** Require measured release gates, save the old manifest pointer and exercise a rollback.

**12.4 Add operating controls.** Monitor source delay, schema drift, service errors and latency; later monitor predictive performance when labels arrive. Protect secrets and administrative routes.

**12.5 Rehearse failures.** Test missing source, partial download, incompatible model, insufficient memory and unavailable current labels.

**Exit evidence:** a deployable artifact, operating runbook, health checks, a rollback record and a monthly budget cap. Deploy only sources and outputs whose distribution conditions have been reviewed.

## Phase 13 Portfolio release

**13.1 Prepare reviewer materials.** README, architecture diagram, data/model cards, reproducible example and a brief recorded walkthrough.

**13.2 Write the research result.** Report the actual comparison, even if the baseline won. Include hardware, data dates, main metric, uncertainty and limitations.

**13.3 Clean the repository.** Remove credentials and disallowed raw data; verify a fresh setup path; keep artifacts discoverable.

**13.4 Write the resume description.** State implemented features and measured outcomes only. Link the demo and reproducibility instructions.

**Exit evidence:** another person can find the working product, reproduce the documented example, and understand what has not been demonstrated.

## Phase 14 Later research

Split each extension into its own proposal and gate: approved timely charting, manually labeled concepts, tracking-based alignment, player matchup modeling, an explanation LLM, or game-sequence decision models. A new feed requires a source audit. A new label requires an annotation protocol. A new policy claim requires an evaluation design.

Do not extend a candidate menu merely because a football concept is familiar. Data support and an identifiable decision-time action remain prerequisites.

## First ten working sessions

At roughly two to three hours per session, the initial work should produce a small reproducible data slice before any neural training:

1. Create repository, environment, issue tracker and decision register.
2. Audit actual play-by-play assets for two completed seasons and the current season.
3. Audit the separate charting and participation assets and document gaps.
4. Ingest a small historical slice with hashes and source metadata.
5. Normalize IDs and inspect joins; quarantine conflicts.
6. Write eligibility and timing tests.
7. Implement broad action labels and unknown handling.
8. Produce one conditional defensive report.
9. Add a cutoff and prove that later records disappear.
10. Record source and labeling failures, then estimate Phase 1/2 work from the observed problems.

The phase exits, not the hour estimates, determine readiness. When a dependency fails, reduce capability or keep the previous release rather than silently weakening the specification.
