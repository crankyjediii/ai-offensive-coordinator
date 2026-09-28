# Research questions and experiments

The project should answer a small set of testable questions while producing a usable scouting application. This agenda distinguishes required comparisons from optional research. Every experiment begins with a written hypothesis, a frozen cohort and information cutoff, and a decision rule. A negative finding can simplify the product and still make a useful portfolio result.

## 1. Experiment record

Create one versioned record per experiment containing its ID, question, owner, status, hypothesis, mechanism, prerequisites, dataset/snapshot IDs, capability mode, target, primary metric, smallest useful effect, comparison, split manifest, tuning budget, seeds, runtime budget and expected failure modes. After completion, append artifacts, measured results, uncertainty, deviations and the resulting decision. Do not replace an unsuccessful hypothesis with the successful one after viewing the holdout.

Use identical eligible rows for paired model comparisons. When testing a richer source, report both the common-row comparison and the change in population coverage; otherwise a charted subset can appear easier simply because difficult rows disappeared. Development findings may inform the next experiment, but the final test remains locked according to [the evaluation protocol](06-evaluation-and-reliability.md).

## 2. Required research sequence

### R01 — Does coordinator history help early-season forecasts?

**Hypothesis:** the normalized coordinator/team/league prior reduces early-season coverage or pressure log loss relative to league-plus-team smoothing. Measure weeks 1–4 separately from later weeks and evaluate on common supported targets. Compare league-only, team-only, coordinator-only, the mixture and current-season-only rates. Keep conditioning variables and source availability identical.

**Falsification:** if gains disappear after adjusting for situation or occur only for one long-tenured team, coordinator attribution has not generalized. A coach-ID permutation within comparable seasons provides a negative control. Retain the coordinator component only with repeatable earlier-fold benefit; otherwise use team/league smoothing and report the negative result.

### R02 — How much transfers with the playcaller?

**Hypothesis:** coordinator history is more useful when the person changes teams than a destination-team-only prior, after accounting for measured personnel continuity. Hold out complete transfer stints, rather than random snaps from the same coach-team combination. Contrast known playcaller changes with unchanged-staff seasons and explicitly exclude uncertain assignments from person-level claims.

**Limitation and gate:** a small number of transfers may not identify a stable effect. Report stint-level variation and wide intervals. Do not claim to separate coaching skill from roster quality. If data are insufficient, keep this as descriptive analysis and omit learned transfer-specific weights.

### R03 — Does sequence order add information?

**Hypothesis:** a snap transformer improves Model A's probabilistic forecast over the best aggregated tabular baseline. Compare the full sequence model with a same-width pooled-history network, history shuffled within the eligible prefix, a transformer without positional information, and a recent-rate baseline. All receive the same fields and snapshot.

**Falsification:** if shuffling does not hurt, the model may be learning a bag of recent observations rather than useful order. That result can favor a cheaper architecture. Ablate lengths 32, 64, 128 and 256 and report latency, memory and training time. Promote a transformer only when its predictive gain passes the predeclared threshold and its added serving cost is acceptable.

### R04 — Do Model A forecasts improve outcome estimates?

**Hypothesis:** out-of-time predicted defensive distributions improve Model B relative to the same network using only situation, offense and profile aggregates. Compare no-A features; calibrated A features; uncalibrated A features; and a deliberately labeled retrospective oracle using true defensive labels only as a research ceiling.

The oracle is prohibited from product ranking and cannot establish an attainable gain. If only the oracle helps, better forecast quality or data may be required. If in-sample A features outperform temporally cross-fitted features dramatically, investigate stacking leakage before reporting an architectural result. Keep the shipped model simpler if A adds no reliable benefit.

### R05 — Does offense and schedule context improve portability?

**Hypothesis:** lagged offense strength and opponent-adjusted defense effects improve outcome calibration across matchups. Compare situation-only; team IDs; lagged numeric strengths; and regularized offense/defense effects. Evaluate held-out team-season and quarterback-change slices while preserving chronological ordering.

**Falsification:** gains restricted to familiar team identities, with failure after personnel changes, suggest memorization. Test unknown-ID fallbacks and the league-average offense option. Only add detailed player embeddings when the number of observed players and their histories support them; cold-start behavior is a required result.

### R06 — Can the system abstain usefully?

**Hypothesis:** combining support, source age, distribution-shift indicators and ranking disagreement reduces factual prediction error among accepted cases while retaining useful coverage. Compare support-only, ensemble-only and combined gates. Plot accepted fraction against factual loss and interval coverage on later folds.

Do not choose the threshold that makes a final test chart look clean. A system that declines every case is not useful, while one that accepts every case has not demonstrated selective reliability. Report which teams, personnel and situations are disproportionately excluded. Promotion requires a predeclared acceptable operating point and truthful unavailable states.

## 3. Conditional experiments

| ID | Question and setup | Proceed only if |
|---|---|---|
| R07 | Compare Huber warm-up plus mean fitting, direct MSE and distributional/quantile heads for EPA | Common factual targets and enough tail events exist; mean and interval claims are evaluated separately |
| R08 | Add screen, play-action or RPO attributes and test whether label ambiguity changes rankings | Intent/attribute definitions, source rights and prospective availability are audited |
| R09 | Compare static prior decay with a past-only online change detector | Enough documented changes and unchanged control stints exist to measure false alarms |
| R10 | Evaluate doubly robust broad-action policies under several overlap and clipping settings | Behavior propensities, cross-fitting, support diagnostics and sensitivity analyses are available |
| R11 | Label a small film sample for multi-receiver concepts and inter-annotator agreement | Lawful film access and a written labeling manual exist; no automated high-level concept claims precede validation |
| R12 | Compare deterministic explanation templates with a constrained text model | A factual-output test set, citation checks and a template fallback exist |

For R11, label independently before adjudication, keep ambiguous/unknown classes, and report agreement by concept. Split by game and offense when checking whether a concept recognizer generalizes. A primary receiver's route is insufficient ground truth for mesh, flood or another full concept. A small annotated corpus can support a data-quality study without supporting a production recommender.

For R12, evaluate numerical fidelity, provenance, unsupported assertions and consistency across repeated runs. Human preference alone does not justify an explanation that invents a defensive weakness. No language-model fine-tuning is required for the MVP.

## 4. Controls and stress tests

Run a future-information sentinel that attempts to add a post-cutoff row and confirms the feature builder rejects it. Test target-label permutations to expose accidental label encodings or cached outcomes. Drop charting fields using the actual historical availability pattern, then test whether a model trained on rich completed-season data fails under the deployed missingness pattern.

Delay-source stress tests should vary assumed latency, not rewrite unknown publication dates as facts. Test unseen coordinator IDs, a defense with no current labels, a late data correction, contradictory source labels and a roster change. Track whether each case produces the intended capability or abstention reason.

Sensitivity analyses should alter the penalty policy, regular-season/postseason cohort, prior decay and support window one at a time. They can reveal dependence on definitions. They do not license selecting whichever definition yields the largest apparent improvement.

## 5. Research budget and reporting

The proposed first development budget is a coarse search capped at 12 configurations per model family, using one exploratory seed and three confirmation seeds for finalists. Set a wall-clock cap after a small hardware benchmark; do not promise a GPU cost before measuring the workload. Record all attempted configurations, including failures. Reserve compute for correct temporal stacking, calibration and diagnostics before enlarging the transformer.

Use one primary metric per question and disclose secondary metrics. Repeated experiments increase the chance of selecting noise; distinguish exploratory findings from a prespecified confirmatory comparison. Report effect sizes and clustered intervals, not only whether a threshold was crossed. When an experiment is underpowered, state what observations would be needed to revisit it rather than asserting no effect.

A completed research report contains the question, exact information set, cohort, comparison, metrics, uncertainty, failure slices, cost and product decision. It should distinguish an empirical finding from a design choice. A credible portfolio can show that a simpler model won, that coordinator transfer was inconclusive, or that finer play recommendations were blocked by missing labels. The deliverable is a reproducible result, not a requirement that every proposed component succeed.

Method references are linked in [ML architecture](04-ml-architecture.md), [coordinator priors](05-coordinator-priors.md) and [evaluation](06-evaluation-and-reliability.md). This agenda is a proposed experiment plan as of 2026-09-27; it contains no completed findings.
