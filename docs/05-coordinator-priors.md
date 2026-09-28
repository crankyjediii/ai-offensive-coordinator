# Defensive coordinator priors

This document specifies how past defensive behavior informs a current-season forecast when observations are sparse. It separates coordinator history, team continuity and league rates, updates them as eligible observations arrive, and records uncertainty about who called the defense. The procedure is an empirical smoothing model to implement and test, not a claim that a coach's tendencies are permanent. It supplies [Model A](04-ml-architecture.md) and the analyst-facing defense profile.

## 1. What the prior describes

Estimate a distribution for one well-defined target at a time: coverage category, a provider-defined blitz flag, five-or-more pass rushers, or box count. Condition on a limited set of prospective variables that the data can support. A starting hierarchy is league overall; down and distance bucket; down/distance plus field zone; and then offensive personnel if charting exists. Avoid crossing every feature into a cell with one observed snap.

Define distance buckets before evaluating them, for example 1–3, 4–6, 7–10 and 11-plus yards. Treat goal-to-go explicitly and record the exact taxonomy version. Formations or motion can be added as model features without creating a complete Cartesian product of reporting cells. Unknown personnel is a missing-information category; it is not league-average personnel.

A probability concerns comparable observed behavior under the training data's play-selection pattern. A coordinator's raw blitz rate can change because the opponents changed. For this reason, compare situation-adjusted and unadjusted priors, and avoid describing a raw difference as a deliberate change in philosophy.

## 2. Coaching and playcaller registry

Maintain a bitemporal registry with `team_id`, `person_id`, `role`, `effective_from`, `effective_to`, `source_url`, `source_available_at`, `ingested_at`, `attribution_status` and a revision identifier. A title of defensive coordinator does not prove playcalling responsibility. Keep defensive coordinator, head coach and documented defensive playcaller as separate roles. Use a stable person ID across team changes.

Allowed attribution statuses are confirmed, reported, ambiguous and unknown. Store the underlying source statement, not a synthetic precise probability of correctness. If playcalling responsibility is uncertain, do not arbitrarily assign every snap to the coordinator. Use a team prior and carry the unknown status into the model. A documented rotation can receive explicit shares only when the evidence supplies a meaningful allocation; otherwise preserve ambiguity.

A retrospective announcement can have an effective date before its publication date. At a historical cutoff, use the registry version known then. Later corrections belong in a separate reconstructed analysis. This prevents an October explanation of a September change from becoming a September feature.

Track interim appointments and midseason handoffs as separate stints. A coordinator's college or other-league history is not automatically pooled with NFL data. It requires compatible labels and a separately tested transfer model. Never manufacture a complete current staffing table to satisfy all-32-team selection.

## 3. Historical source distributions

For target class `k`, construct historical league `p_L,k`, team `p_T,k` and coordinator `p_C,k` distributions using only earlier seasons available before the cutoff. Apply age decay and shrink sparse cells toward their parent cells. Each historical distribution sums to one. The coordinator component follows the person across teams only when playcalling attribution and label compatibility are acceptable.

Use nonnegative historical weights `u_i = exp(-age_days_i / τ_hist) × quality_i`, where quality rules are documented and bounded between zero and one. The decay timescale and quality rules are selected on past folds. A provider's label confidence can influence a weight only when its meaning is understood; an invented confidence number cannot.

For a historical component `r`, compute class weights `H_r,k = Σ_i u_i 1[y_i=k]`, total `H_r = Σ_k H_r,k`, and a smoothed distribution `(H_r,k + η p_parent,k)/(H_r + η)`. Handle the empty case as the parent distribution. Keep a small positive league pseudocount for eligible rare classes, while excluding structurally impossible classes through the target definition.

Blend the normalized components:

`p0,k = a_L p_L,k + a_T p_T,k + a_C p_C,k`, with `a_r ≥ 0` and `Σ_r a_r = 1`.

A logit/softmax parameterization can learn these mixture weights from continuity features using historical folds. The initial implementation can use a small grid of fixed weight combinations. Either way, redistribute unavailable components over the remaining components and record the rule. Do not fill an absent coordinator component with a guessed scheme label.

The same historical snap may influence team and coordinator distributions. A convex mixture treats them as overlapping predictors. It must not add both sets of raw counts as if they were independent evidence. Total prior strength is controlled separately by `κ`; having two overlapping sources does not double it.

## 4. Current-season updating

For eligible current-season observations, define weights `w_i` using a documented current-season decay and quality policy. Let `C_k = Σ_i w_i 1[y_i=k]` and `W = Σ_k C_k`. The smoothed estimate is:

`p_hat,k = (C_k + κ p0,k) / (W + κ)`.

Here `κ` is the historical prior's pseudocount strength in the same scale as `W`. It is not a number of actual historical snaps. With `W=0`, the forecast is the prior. With increasing `W`, current observations receive more weight. No fixed week forces the prior to disappear. If charted current-season labels have not yet been released, `W` for that target remains zero even if the team has played games.

Store three different support quantities: raw eligible snap count `N`; weighted mass `W`; and Kish weight-concentration diagnostic `n_eff = (Σ_i w_i)^2 / Σ_i w_i^2`, defined as zero when no observations exist. `n_eff` is not the number of independent NFL snaps; clustering within games can reduce information further. It must not silently replace `W` in the update, because doing so changes the model's prior-to-data balance.

Illustrative arithmetic only: suppose a binary prior is 0.30, `κ=40`, and 20 fully weighted current snaps contain eight positive labels. The updated rate is `(8 + 40×0.30)/(20+40) = 0.333`. It is neither the raw 0.40 rate nor evidence that the true rate is known precisely. This invented example is not a measurement of any NFL defense.

Select `κ` from a coarse past-fold grid such as 0, 10, 25, 50, 100 and 200, then refine only if validation warrants it. Permit different strengths for coverage and pressure targets because they may have different stability and label counts. A hierarchical parameterization by continuity group is preferable to a separate free parameter for every team-season. Do not tune `κ` against the season being reported.

## 5. Roster continuity, transfers and changes

Estimate preseason roster continuity as the share of the previous season's defensive snaps attributable to players documented on the roster at the cutoff. This measures retained historical participation, not projected playing time. Do not use final current-season snap shares. In-season features may incorporate completed, already available current participation, while retaining a timestamp and missing flag.

Use continuity to moderate historical transfer. A coach moving teams may retain a coordinator component while receiving less team-history weight from the destination if staff or personnel changed. A team keeping most players but changing the playcaller can emphasize team and league components. These are candidate weight rules to test; there is no fixed truth that one source dominates in Week 1 or that roster continuity determines a scheme.

On a documented midseason playcaller change, start a new stint counter and retain prior-stint observations as historical context with an explicit decay or mixture weight. Do not erase raw data. Keep a separate experiment that treats all current-season observations alike to measure whether the change handling helps.

An online change detector may compare recent forecast log losses or class frequencies with their earlier baseline. It must use only accumulated past observations, require a minimum number of completed games and trigger a documented update rule. A change point chosen after seeing the entire season is retrospective research and cannot be reported as a live detection. Test false alarms during ordinary opponent changes and stretches with missing labels.

## 6. Uncertainty and statistical interpretation

With fixed `p0`, fixed `κ` and unweighted independent categorical observations, a Dirichlet prior with parameters `κ p0` yields the familiar posterior mean. Real implementation weights, empirical hyperparameter tuning, overlapping historical sources and clustered football plays weaken that literal interpretation. Describe the initial system as empirical shrinkage or a weighted pseudo-posterior, and do not present its analytic credible intervals as calibrated real-world guarantees.

For displayed uncertainty, evaluate interval behavior on held-out games and use game-clustered resampling to assess sampling variability of reported tendencies. Distinguish uncertainty in the long-run rate from uncertainty in the next snap's categorical outcome. A precise estimated probability does not imply that a particular snap is predictable.

A later hierarchical multinomial model can jointly estimate situation, coordinator, team and season effects with partial pooling. Coordinator and team effects are difficult to separate when people rarely change teams; require identifiability constraints, transfer holdouts and broad intervals. The Stan user guide explains multilevel regression and the need to constrain overlapping effects; adopting those methods would still require football-specific validation. [Stan, hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression).

## 7. Stored outputs and acceptance tests

Persist one profile per target, situation and cutoff with the historical component distributions, mixture weights, `κ`, current class weights, raw counts, `W`, `n_eff`, number of distinct games and opponents, last observation date, staffing attribution status and source snapshot IDs. Store a `prior_method_version`; never overwrite the explanation of a published forecast.

The analyst interface should be able to show a statement such as: “Estimate combines league, team and documented playcaller history; current-season coverage labels are unavailable.” Numeric component contributions may be shown, but must not be called measured causal influences of the coach and roster.

Tests must cover zero-history fallback; probability normalization; missing target labels; the illustrative update arithmetic; different weights on identical counts; unknown playcaller; overlapping team/coach history; a transfer; an interim stint; a late registry correction; and a source with no current-season publication. In backtests, compare league-only, team-only, coordinator-only, mixture and current-season-only forecasts under identical information cutoffs. The prior ships only if it improves the selected probabilistic score or provides a defensible sparse-data fallback without unacceptable calibration deterioration.

Sources verified: 2026-09-27. Weight grids, continuity rules and transfer handling are proposed experiments, not measured findings.
