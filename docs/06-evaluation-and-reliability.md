# Evaluation, backtesting and reliability

The project must demonstrate what its data and models can predict, under the information actually available to the proposed user. This document separates factual prediction quality, recommendation stability and exploratory policy evaluation. None of these has been measured yet. The thresholds below are proposed development gates to freeze before testing, not achieved results or guarantees of coaching value.

## 1. Claims and evaluation units

Model A predicts defensive behavior on a future eligible snap. Model B predicts the observed outcome for an eligible action and situation. The recommender ranks supported alternatives using those estimates. Only the chosen action has a factual outcome; there is no test-set label identifying the best unchosen play.

Use play-level losses but preserve game membership throughout splitting and inference. Report performance both pooled across plays and averaged across team-games, because high-volume teams should not silently determine every result. Include distinct games, teams, opponents, seasons, positive events and missing-label counts beside metrics. State whether the regular season and postseason are separate cohorts.

The first release excludes special teams, no-plays, kneels and spikes. Freeze the penalty policy and ambiguous-intent rules from [the taxonomy](03-football-ontology.md). Report excluded counts by season and team. Do not decide after seeing results that a difficult cohort was outside scope.

## 2. Availability and historical replay

Every historical request has a timezone-aware UTC `cutoff_at`, usually before kickoff, and an immutable `snapshot_id`. All target-game plays share that pregame snapshot. Historical charting fields can be missing while play-by-play fields for the same old snap are available. Label eligibility for later scoring is independent of feature eligibility at decision time.

Maintain three explicitly named evaluation levels:

| Level | Permitted claim | Required evidence |
|---|---|---|
| Archived as-of replay | Reproduces the stored information available to the project then | Immutable historical snapshots and availability/ingestion records |
| Reconstructed historical test | Tests the model on earlier games using a present-day historical dataset | Revision status disclosed; no claim of exact original publication state |
| Delay-simulated historical test | Tests an explicit latency assumption | Conservative field-specific delay policy, sensitivity analysis and stated unknowns |

`event_at` alone never establishes availability. If `source_available_at` is unknown, do not infer a historical timestamp from today's download. A current snapshot can be used to reconstruct historical event ordering, but cannot prove which corrections or charting labels were visible then. Applying an assumed seven-day delay still leaves the test a simulation. Historical participation released only after a season cannot become a same-season input by imposing a short assumed delay.

Archive new 2026 predictions prospectively before outcomes and labels arrive. Preserve both the original feature snapshot and later scoring labels. Revised labels generate a new evaluation version; they never rewrite the original prediction. If EPA comes from a subsequently revised expected-points model, version that outcome definition and distinguish it from an EPA model known at decision time. Strict training replay requires historically available targets or an expected-points model fitted only to earlier data.

## 3. Chronological experiment layout

Audit 2022–2026 sources first, then choose the common label cohort. A useful provisional schedule is: train through 2023, tune using the first portion of 2024, calibrate using its later portion, and keep 2025 as the locked final test. This schedule applies only when the required labels exist. A target available for fewer years needs a shorter, explicitly weaker study or must remain an experiment.

Within development, use expanding game blocks. For each validation block, fit all transforms, priors, offense ratings and models on earlier data; generate Model A features for Model B using earlier-only fits. Keep all rows from a game in the same block. Time ordering is by actual kickoff, not only a week number, and availability filters can impose a further gap.

A concrete proposed 2024 allocation is weeks 1–12 for model selection and weeks 13–18 for calibration and abstention-threshold selection, with final training refit ending before week 13. The small calibration block may be insufficient for rare events; in that case use additional earlier out-of-time predictions or suppress the affected probability claim. Do not borrow the 2025 test to improve calibration. If calibration settings themselves are numerous, divide calibration development again or select a simple prespecified calibrator.

Before opening the final test, lock the feature manifest, taxonomy, candidate set, selected model family, seeds, loss weights, prior parameters, calibration recipe, utility weights, support rules and primary metrics. Test weights remain frozen, while the history/profile builder may incorporate earlier completed games only when those fields were available by each new cutoff. Report this distinction. Any test-driven change consumes that holdout; future 2026 predictions become the next honest evaluation.

## 4. Metrics and comparisons

| Component | Primary measures | Required supporting measures |
|---|---|---|
| Coverage forecast | Multiclass log loss and Brier score | Per-class precision/recall, macro F1, confusion matrix, class support |
| Binary defense forecasts | Log loss and Brier score | Reliability curves, AUROC and precision-recall performance |
| Rush/box counts | Distribution log loss, count MAE | Error by count and interval/distribution calibration |
| EPA outcome | Mean squared error and mean prediction bias | MAE, RMSE, residuals by situation and action |
| Binary outcome heads | Log loss and Brier score | Calibration, positive-event counts and precision-recall curves |
| Outcome intervals | Empirical coverage and width | Coverage by season, broad action and risk slice |
| Recommendation system | Acceptance rate and supported-candidate coverage | Rank stability, abstention reasons and factual error among accepted cases |

Coverage accuracy alone rewards common classes; it does not measure the quality of the whole distribution. A rare turnover head can have high accuracy by predicting no turnovers. Report pooled and per-class probabilistic scores, and publish the exact Brier-score normalization. For binary events use the mean of `(p-y)^2`; for multiclass use the mean sum over classes, without silently dividing by class count.

Compare models on paired identical rows. Report differences with game-clustered bootstrap intervals; the initial proposal is 1,000 bootstrap replicates, with a sensitivity check by team-season or multi-game blocks where data permit. Preserve temporal prediction provenance in each resampled unit. Conditional bootstrap intervals for a fixed fitted model do not include all training, selection or future-distribution uncertainty.

Slices include early-season weeks 1–4; later regular season; short/medium/long distance; red zone; leading/trailing; action family; known/unknown playcaller; coaching transfer; high/low continuity; missing charting; and sparse versus common personnel. Report only slices with enough events to make their uncertainty legible. A slice too small to assess is unavailable, not a pass.

## 5. Probability calibration

Calibrate Model A outputs before supplying them to Model B, using temporally held-out predictions. Calibrate Model B's binary probabilities on a later held-out block after its weights are frozen. Multiclass temperature scaling is a simple candidate for coverage; logistic calibration can be used for binary heads. Isotonic regression is an alternative only with enough data to avoid fitting a staircase to a handful of events. Temperature scaling is supported by prior research as a useful post-processing method, but the best choice here must be tested. [Guo et al., *On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html).

Show reliability diagrams with denominators and uncertainty; use bins fixed before evaluating the test. Expected calibration error depends on binning and should remain a secondary diagnostic. Report calibration-in-the-large and slope where estimable. Do not declare calibration from one attractive plot or pool incompatible capability modes into a single curve.

Class rebalancing and action-conditioned sampling can distort event frequencies. Keep an untouched representative calibration cohort and document any sampling corrections. Fit separate calibrators by broad action only when enough support exists; otherwise use one supported calibrator and disclose slice performance. Store every calibrator with the model bundle and training boundary.

## 6. Four different kinds of uncertainty

**Outcome variability:** similar snaps have different results. A prediction interval targets this distribution for observed-action cases. Quantile heads or held-out residual calibration can estimate it; coverage and width must be measured prospectively.

**Estimation uncertainty:** finite training data and model choices change estimated means or probabilities. Independent-seed ensembles give one diagnostic of this variation. They do not cover shared model bias, missing features or all plausible training datasets. The deep-ensemble paper motivates the method; it is not a guarantee for this application. [Lakshminarayanan et al., *Deep Ensembles*](https://arxiv.org/abs/1612.01474).

**Data uncertainty:** missing labels, stale snapshots and uncertain playcaller attribution affect the evidence. Display those facts directly through support and availability fields. Do not compress them into a decorative percentage.

**Counterfactual uncertainty:** an unchosen action has no observed outcome on that play. An interval calibrated on factual chosen actions does not automatically retain coverage after the recommender selects a different action or a rare state-action combination. Mark extrapolated candidates and abstain where support is inadequate.

A proposed interval experiment fits 5th and 95th percentile heads, calibrates residual nonconformity on a later block, and reports realized 90% coverage. Ordinary conformal coverage depends on exchangeability; NFL games and changing seasons do not satisfy that automatically. Weighted methods address specified forms of covariate shift, not arbitrary coaching changes or hidden confounding. Report empirical coverage and assumptions rather than a universal guarantee. [Tibshirani et al., *Conformal Prediction Under Covariate Shift*](https://arxiv.org/abs/1904.06019).

## 7. Abstention and consistency

Use statuses `ok`, `limited`, `abstained` and `unavailable`. Distinguish unavailable target capability from a supported model declining to rank. Initial reason codes include `MISSING_CAPABILITY`, `LOW_ACTION_SUPPORT`, `STALE_SOURCE`, `UNKNOWN_REQUIRED_CONTEXT`, `OUT_OF_DISTRIBUTION`, `UNSTABLE_RANKING` and `INSUFFICIENT_CALIBRATION`.

Tune support gates from earlier validation coverage-risk curves. Candidate support should consider similar situations, distinct games/opponents and behavior propensity, not just total league count. A proposed starting reporting floor is 100 comparable labeled snaps from at least five games for a narrow slice, subject to review after the audit; it is not proof of adequate statistical power. Rare-event calibration also needs enough positive events, assessed separately.

Measure stability over training seeds, bootstrap refits where feasible, and small valid scenario changes. A deterministic deployed bundle should return the same ranking for identical inputs; independent training runs need not. Large rank changes after a one-yard distance change merit inspection but can be reasonable at a first-down or goal-line boundary. Do not force artificial smoothness across real football discontinuities.

Return a tied or unordered supported set when the ranking margin is not meaningfully distinguishable. Specify a deterministic secondary sorting key for display; do not portray that tie-breaker as football evidence. Ensemble agreement is one screen among several, never a probability that the top call is correct.

## 8. Off-policy evaluation as a separate study

Let `x_i` be predecision information, `a_i` the logged action, `y_i` its EPA, `e(a|x)` the behavior propensity, `m(x,a)` the outcome estimate and `π(a|x)` a frozen candidate policy. A contextual doubly robust estimate combines predicted rewards with a propensity-weighted residual:

`V_DR = mean_i [ Σ_a π(a|x_i)m(x_i,a) + π(a_i|x_i)/e(a_i|x_i) × (y_i-m(x_i,a_i)) ]`.

This is a standard contextual-bandit construction, not validation from observing alternative NFL calls. [Dudík, Langford and Li, *Doubly Robust Policy Evaluation and Learning*](https://arxiv.org/abs/1103.4601).

Its causal interpretation requires a well-defined action, consistency, adequate overlap and no relevant unmeasured confounding conditional on the supplied context. Football data may miss audibles, injuries, intended assignments, protection calls or information the coaches saw. A broad action can contain multiple materially different play versions. Doubly robust estimation does not repair those omissions merely because one nuisance model predicts well.

Fit behavior and outcome models using expanding earlier-only blocks, and compute each evaluation row's nuisance predictions without fitting on that row or its game. Freeze policy selection before its evaluation block. Cross-fitting limits overfitting bias but does not eliminate confounding or establish time-series independence. [Chernozhukov et al., *Double/Debiased Machine Learning*](https://arxiv.org/abs/1608.00060).

For each policy comparison, publish action counts by situation; propensity calibration; target-policy mass outside supported regions; the minimum and quantiles of relevant propensities; importance-weight quantiles and maximum; and weight concentration `ESS=(Σw)^2/Σw²`. State-action exclusions must depend on predecision information and prespecified support rules. Restricting to well-supported situations changes the estimand; report how much of the original population remains.

Compare direct outcome-model estimates, inverse-propensity estimates, self-normalized estimates and doubly robust estimates. Report both unclipped and prespecified clipped-weight sensitivity analyses. Clipping trades variance for bias; a stable clipped result does not erase an unstable untrimmed analysis. Use paired game-clustered uncertainty for policy differences and inspect whether one game dominates them.

The initial study covers one-snap EPA under the logged context distribution. It does not estimate whole-game wins, drive optimization or a defense's adaptation to repeated new calls. Sequential policy evaluation would require a different model and stronger assumptions. If overlap fails or estimates disagree materially, publish the failure and omit policy-value claims from the product and résumé.

## 9. Release gates and evidence package

Integrity gates are mandatory: zero forbidden-feature violations; unique eligible play keys; documented source/label versions; reproducible snapshots; no target-game history; and train/API feature parity. Failing integrity invalidates performance comparisons.

Proposed model-promotion gates are at least a 1% relative improvement in the selected primary loss across two chronological development windows, no material deterioration on prespecified high-support slices, acceptable held-out calibration, and successful abstention tests. The exact tolerance must be fixed before evaluation and supported by sample size. A clustered interval for the primary paired difference should support the claimed direction; otherwise describe the result as inconclusive and retain the simpler champion. The locked test is reported once, even if it disappoints.

Package a split manifest, cohort flow table, metric CSV, calibration plots, interval report, ablation table, representative successes and failures, model card and prediction log. Include training and inference cost measurements on named hardware. The final claim should match this evidence: measured prediction improvements are possible; better real-world playcalling remains unproven without appropriate intervention or stronger identification.

Sources verified: 2026-09-27. No evaluation metrics in this specification are results from a trained model.
