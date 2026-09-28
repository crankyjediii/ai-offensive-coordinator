# Football ontology and play taxonomy

This proposed taxonomy separates a call a user could choose before a snap from descriptions learned after the snap. Its purpose is to make training labels, candidate actions, scouting filters and explanations use the same definitions. Version `1.0.0-proposed` is a design contract; the project has not yet completed source audits or annotation validation.

## 1. Five different objects

| Object | Meaning | Example in this project |
|---|---|---|
| Situation | Known or user-specified state before the decision | Third down, seven yards to gain, offense trailing by four. |
| Candidate action | An offensive intent that could be selected at the decision time | Broad dropback or designed rush; finer calls require stronger labels. |
| Observed defensive behavior | What charting reports happened on a completed snap | Coverage annotation, rush count, box count. |
| Predicted defensive behavior | A distribution for the requested scenario | Model probability assigned to a coverage category. |
| Outcome | What happened after the decision | EPA, yards, sack, turnover, first down. |

A field can belong to different objects at different times. The observed box count of a completed earlier play can enter a lagged history after publication. The current play's box count is unavailable in the pregame MVP. A user-specified box is a hypothetical scouting filter; it does not become an observed input. Store evidence role and time alongside the value.

Use `observed`, `predicted`, `user_assumed`, and `unavailable` as presentation provenance. Keep these separate from prediction status (`ok`, `limited`, `abstained`, `unavailable`). A value may be observed yet too sparse to support a recommendation.

## 2. State and context vocabulary

Continuous inputs remain continuous as well as optionally bucketed. Proposed scouting bins are distance `short` 1–3, `medium` 4–6, `long` 7–10 and `very_long` 11+; field zones use yards to the opponent's goal: goal line 1–5, red zone 6–20, open field 21–80 and backed up 81–99. These are product conventions, not provider definitions. Zero-distance, boundary and goal-to-go cases require explicit validation, not silent assignment to a normal bin.

Canonical state contains offense, defense, season type, quarter/overtime period, seconds remaining, down, distance, yards to goal, goal-to-go, score margin from the offense's perspective and remaining timeouts. The cutoff is separate from the hypothetical game clock. Home/away and venue may be additional context if consistently available. Weather is deferred until a dated forecast or known-at observation is available; a retrospective full-game description is not automatically a pregame feature.

Personnel is the count of players by listed role, not their exact alignment. For normal offensive packages, store RB, TE and WR counts independently; derive `11` only for the validated one-RB/one-TE package under the defined counting convention. Fullbacks, extra linemen and hybrid roles retain source positions and parsing notes. Do not force an unusual package into a two-digit code. Player identity, roster position and on-play assignment are different fields.

Defensive personnel is likewise a role-count vector, with optional coarse DB-count groups for analysis. `nickel` and `dime` refer here to observed five-DB and six-DB packages under the source position mapping. They do not imply a particular front, coverage or rush call. If source player positions are inconsistent, retain `unmapped` instead of inventing a package.

Quarterback alignment has `under_center`, `shotgun`, `pistol`, `other_observed`, and missing states. Formation is a separate source-preserving category. Formation width, bunch, stack, trips, empty, strength, shifts and motion type require direct labels or validated annotations; a shotgun flag does not supply them. FTN's published charting provides quarterback position and a motion indicator, among other attributes, but the project must not expand a boolean into an unobserved motion direction or player assignment. [FTN field reference](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)

## 3. Candidate-action hierarchy

The first usable action space contains two broad intents, subject to cohort and label validation:

| Candidate ID | Decision meaning | Training evidence | Main limitation |
|---|---|---|---|
| `designed_rush` | Select a rushing call | Non-kneel rushing event with supported absence of dropback/scramble evidence | Public flags are an intent proxy, not a complete call sheet. |
| `dropback` | Select a passing/dropback call | Supported dropback evidence, including sacks and scrambles | Bundles many calls with different risks and read structures. |

This comparison answers a broad situational question. It cannot rank mesh against flood or inside zone against counter. A rush gap describes a recorded run location; it does not identify blocking rules. An observed throw's depth and target route reflect what developed after the snap, so they cannot be freely chosen as if they were fully observed pre-snap actions.

Use the following proposed classifier order on already eligible scrimmage plays:

```text
if kneel, spike, conversion, special teams, administrative or nullified:
    action = null; reason = excluded_play_type
elif inconsistent or missing critical action flags:
    action = null; reason = ambiguous_intent
elif supported qb_dropback OR supported sack OR supported scramble:
    action = dropback
elif supported rushing attempt AND explicitly no dropback/scramble:
    action = designed_rush
else:
    action = null; reason = unsupported_intent
```

The adapter must define how supported evidence is established; it must not treat null as false. Cross-field contradictions enter a review queue. A handoff recorded on an RPO and a pass selected on an RPO share a higher-level option call that the broad intent proxy may not recover. Mark directly charted RPOs, report their proportion, and perform an exclusion sensitivity test. Do not claim that the baseline estimates the effect of choosing an RPO's branch before the defense responds.

### Finer calls are conditional research extensions

The original idea proposed roughly 20–40 families. Keep that as a possible mature scope, not a required class count. Add classes only when the source supports intent on unsuccessful plays as well as successful ones.

| Proposed extension | Evidence needed before recommendation use | Until then |
|---|---|---|
| Play-action dropback vs straight dropback | Validated call/intent annotation including sacks, scrambles and aborted throws | Historical attribute analysis with masks. |
| Screen vs other dropback | Screen intent coverage on failed plays and consistent provider semantics | Charted screen tendency/outcome description. |
| Designed QB run and QB sneak | Explicit intent or reviewed label rule, with rush/scramble separation | Broad run action plus descriptive tags. |
| Draw, inside/outside zone, power, counter, pin-pull | Blocking-scheme annotation and negative/ambiguous labels | Unavailable. Never infer from gap alone. |
| RPO package | Whole-call definition, available options and read structure | Observed RPO attribute only. |
| Mesh, flood/sail, four verticals, dagger, smash, drive, stick, spacing, slant/flat | Multi-receiver route concept evidence, protections where relevant, independent review | Research annotation candidates. |

Attributes can overlap: a call may use play action and motion, and a screen may be attached to an option. Avoid a single flat label that creates arbitrary precedence among these traits. When intent is validated, define an action object with a base family and optional attributes, then enumerate only supported combinations. Count support for the joint combination, not each attribute separately.

Candidate feasibility includes a user-selected offense, its known personnel, intended package and the supported action vocabulary. A user may explore a league-average offense. This is a pooled model context, not a roster that can execute every named concept. Protection, receiver assignments, eligibility and detailed playbook legality remain outside a broad-family recommendation.

## 4. Retrospective offensive descriptors

Preserve recorded run direction and gap without translating them into a scheme. Separate designed QB runs from scrambles whenever supported. A scramble belongs in dropback outcome denominators; putting it in the designed-run bucket moves one consequence of passing into the competing action.

For pass analysis, proposed realized target-depth bins are behind the line `<0`, short `0–9`, intermediate `10–19`, and deep `20+` air yards. These are project bins, not the source's short/deep labels. Direction is left/middle/right when supplied. Sacks and scrambles have no realized target depth: mask that label rather than assign short or zero. Incomplete passes may still have target descriptors. Missing target information does not mean a throw was quick.

Normalize the participation route vocabulary into stable IDs: `corner`, `deep_out`, `go`, `hitch_curl`, `in_dig`, `post`, `quick_out`, `screen`, `shallow_cross_drag`, `slant`, `swing`, `texas_angle`, `wheel`. These describe the provider's primary-receiver route field. They do not establish every receiver's route, the initial play call or the quarterback's original read. [Participation route and coverage dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html)

An analysis labeled “outcomes on recorded intermediate targets” conditions on a realized target choice. It must not become “expected outcome if we call an intermediate pass” without a separate intent-label study. Record this restriction in the model feature registry, not just in chart captions.

## 5. Defensive labels and denominators

Use source-preserving coverage IDs before any aggregation: `COVER_0`, `COVER_1`, `COVER_2`, `2_MAN`, `COVER_3`, `COVER_4`, `COVER_6`, `COVER_9`, `COMBO`, `BLOWN`. Add a model-level `OTHER_OBSERVED` only under an explicit pooling rule; unknown labels are masked. No current season is assumed to contain this full set.

Coverage is a post-snap classification. A pre-snap one-high or two-high shell is a different label, requiring its own observation method. Do not derive shell from coverage or assert that the defense “showed two-high” because a model predicts Cover 4. Preserve provider distinctions such as Cover 6 and Cover 9; mirroring or strength conventions need documented alignment information before normalization.

Man/zone uses the provider's own category when available. Do not force every mixed, match, combo or blown annotation into pure man or zone. The Model A task dictionary must list which labels are eligible and which are pooled. Evaluate label-rich historical coverage on its own cohort; missing coverage on rushing plays must not become a negative example for each coverage class.

Maintain separate quantities for `pass_rusher_count`, `blitzer_count`, `five_plus_rushers`, `pressure_observed`, `qb_hit` and `sack`. Proposed `five_plus_rushers = pass_rusher_count >= 5` is a transparent derived label. It is not automatically equivalent to the provider's notion of a blitz. If a provider's blitzer count is used to define `blitz_observed`, document that mapping and test disagreement with the five-plus measure. Pressure is an outcome of the rush/protection interaction, not evidence that a blitz was called. A sack is a narrower event than pressure.

Box count is an observed pre-snap alignment measure available retrospectively. It does not identify defensive techniques, front spacing, a particular run fit or the assignment of each defender. Front, shell, gap responsibility, match rules, stunts and simulated pressure are future labels requiring charting or tracking. Never generate these as factual explanations from ordinary PBP.

Each rate carries its own denominator: labeled eligible dropbacks for pressure/rush labels, labeled eligible plays for box count, and explicitly scoped charted snaps for coverage. A denominator excludes unknown labels and displays the missing-label count. Distinguish raw count, sum of recency weights, and effective sample size; they answer different questions.

## 6. Outcomes and objective definitions

| Label | Proposed definition | Edge handling |
|---|---|---|
| `epa` | Frozen provider EPA from the possession offense's perspective | Null EPA masks this head; do not substitute yards. Preserve provider/version. |
| `success` | `epa > 0` | Exactly zero is not success; missing EPA stays missing. |
| `yards` | Audited official recorded gain for the play | Define the underlying source field; keep laterals/recoveries and penalty accounting caveats. |
| `explosive` | Gain ≥10 for designed rush; ≥20 for dropback | A long scramble uses the dropback threshold. This is a chosen project convention. |
| `sack` | Supported sack event | Zero for a verified designed-rush candidate by definition; missing label is never zero. |
| `turnover` | Interception or offensive lost fumble on the play | Count a single binary event if both occur; review multiple-fumble/return sequences. |
| `first_down_or_td` | Offensive first down or offensive touchdown | A defensive return touchdown is not offensive success. |
| `failed_fourth_down` | Failed offensive fourth-down conversion | Separate from interception/fumble turnover probability. |

Provider event indicators and EPA support the raw outcomes; canonical definitions and masks above are project choices. [PBP field definitions](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv)

Every candidate probability is unconditional over eligible plays for that action. For example, explosive probability on a dropback includes incompletions, interceptions, sacks and scrambles in its denominator. An explosive rate among completed passes is a different statistic. Never compare one with the other in a ranking.

Default ranking maximizes expected EPA. Optional risk weights act on these explicitly defined probabilities and use documented units. EPA already incorporates consequences of turnovers and sacks; extra penalties express user preference and may double count those consequences. Win probability and clock-aware endgame strategy require separate objectives and validation.

## 7. Annotation evidence and taxonomy governance

Every derived label stores `taxonomy_version`, rule ID, source field references, source era, eligibility, missing reason and evidence level: `provider_charted`, `deterministic_proxy`, `human_reviewed`, or `model_inferred`. These are evidence categories, not ordered numerical confidence scores. Model-inferred labels cannot silently become evaluation truth.

For a future concept annotation study, write a call definition and exclusion cases first. Draw a stratified sample across teams, situations, seasons and unsuccessful plays. Have two reviewers label independently using lawful footage/material, mark ambiguity, then adjudicate disagreements. Report per-class agreement, confusion, ambiguous fraction and sample sizes. A target route alone cannot resolve a multi-route concept; if footage does not show the full design, keep `insufficient_evidence`.

Maintain a taxonomy changelog. A changed threshold, class merge or intent rule changes the version and requires regenerated labels, capability audits, split summaries and affected model training. Never compare models using different taxonomy versions without a shared evaluation mapping and an explicit compatibility report.

Minimum classifier fixtures cover a normal rush, pass attempt, sack, scramble, kneel, spike, nullified penalty, RPO, missing flag, contradictory flags and a novel provider category. Leakage tests must reject realized target depth, target route, current coverage and current pressure in predecision features. A candidate containing an unvalidated concept must fail validation with an actionable reason code. [Data contracts](02-data-sources-and-schema.md), [model architecture](04-ml-architecture.md), and [evaluation](06-evaluation-and-reliability.md) define the downstream enforcement.
