# Limitations ethics and data caveats

This project estimates football tendencies and outcomes from incomplete observational data. It cannot observe the outcome of every uncalled play, reconstruct a team's private playbook from public results, or guarantee a better call. Those limits affect the model specification, evaluation and user interface.

## Limits of the observations

Public play-by-play describes game events. It is not a complete record of assignments, adjustments, disguises, route combinations or quarterback reads. A gap label is not a blocking scheme. A target route is not a full concept. Box count alone is not alignment. Pressure, blitz and pass-rusher count describe different phenomena.

Missing charting is not random by default. Label completeness may vary by season, provider, game, play type and unusual event. Training only on fully labeled rows can change the sample. Reports must show which eligible plays were excluded and compare observed outcomes of retained and missing-label cohorts.

Delayed labels create a second limit: information can exist in today's historical file without having been available at the old prediction time. A replay on revised files should be named accordingly. Only archived snapshots and valid timestamps support a claim that the replay exactly reproduces historical information availability.

The project should archive source versions from the first ingest. It cannot manufacture older release timestamps. Details and fallback timing rules are in [Data sources and schema](02-data-sources-and-schema.md).

## Limits of “best play”

NFL play selection depends on private scouting, injuries, protection assignments, personnel readiness, playbook installation, quarterback checks and opponent expectations. The dataset may omit these factors. An association between an observed action and EPA does not prove the same action would cause that outcome in a new situation.

The intended broad action may itself be uncertain when only the result is observed. A scramble follows a dropback; it must not automatically be treated as a designed run. Screen and play-action labels may fail to identify intended calls consistently on sacks or broken plays. These are action-definition problems, not merely missing feature values.

An outcome model can be evaluated on factual outcomes. It cannot directly validate its estimate for the action that was never taken. Off-policy estimates require explicit assumptions about overlap, confounding and policy stability; a doubly robust estimator does not remove unmeasured confounding.

The interface should say “model-ranked candidates” or “estimated outcomes.” Reserve “improved policy value” for a carefully qualified estimate with its assumptions and support diagnostics. Do not claim added wins, avoided sacks or points gained as observed results when they come from unexecuted alternatives.

## Statistical and model limits

NFL samples become small after conditioning on team, situation, personnel and concept. A large neural network can fit identifiers and noise. Baselines, shrinkage, chronological validation and abstention are required controls.

An ensemble's narrow spread measures agreement among models with shared data and assumptions. It can coexist with shared bias or missing information. It is not a calibrated probability that the recommendation is correct. Similarly, a predictive interval for the outcome of an observed action is not a confidence interval for the causal benefit of an alternative.

Coordinator history is uncertain when the nominal coordinator is not the actual playcaller, a head coach controls the defense, or responsibilities change midseason. Team, coach and roster effects may be statistically entangled. The system stores evidence confidence and uses unknown assignments when needed.

Changes in players, rules, provider definitions and offensive environments can make older data less comparable. Track such changes and test sensitivity. Do not assume a prior becomes irrelevant at a fixed week or that a player's historical average is their current ability.

## Data rights and attribution

Keep a source-specific register containing publisher, download location, dataset version, license or terms URL, retrieval date, required credit, and allowed redistribution decision. Software licenses and data licenses are separate. A package's open-source license does not automatically license all data it downloads.

The participation provider documentation specifies attribution to FTN Data via nflverse for 2023 onward and NFL NextGenStats via nflverse for earlier years, and identifies CC BY-SA 4.0 for the dataset. Verify the exact asset and applicable terms before distribution. Source: [nflreadr participation documentation](https://nflreadr.nflverse.com/reference/load_participation.html), checked September 27, 2026.

CC BY-SA 4.0 requires appropriate credit and notices, and ShareAlike conditions apply to covered adaptations. The license does not by itself settle every question about a particular derived model, database or combined publication. Keep source notices with distributions and record the review decision for each artifact instead of assuming either that all code must use the data license or that trained weights are always unrestricted. Sources: [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/) and [legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en), checked September 27, 2026.

No paid feed, restricted film, credentialed endpoint or licensed tracking archive is included in this documentation package. Do not bypass access controls to add one. Viewing a dataset or video is not itself permission to redistribute it. Use team names and abbreviations for the initial interface; obtain appropriate rights before bundling third-party logos, photographs or footage.

## Player and coach claims

Use public sports-performance data for the stated analytical purpose. Do not infer medical conditions, diagnoses or private facts from play outcomes or absence from a roster. An injury report, when legitimately used, remains a dated published status rather than a medical conclusion.

A low model rating should not be framed as a judgment of a person's intelligence, effort or character. Show context, sample size and uncertainty. Avoid a coordinator “skill” leaderboard when team assignments and player quality cannot be adequately separated.

The application is not represented as an NFL, club or provider product. A public demo should state that it is an independent portfolio project and credit the data providers.

## Product safety and honest communication

The current scope is scouting and research, not betting recommendations or automated in-game control. Do not imply bookmaker-level accuracy, guaranteed profit or tested live coaching value. A future change in use would require a new product and evaluation specification.

An optional language model receives a restricted evidence object. Treat retrieved descriptions as data, not instructions. Validate its numbers and entities, escape displayed text, and fall back to deterministic explanations if verification fails.

Do not make access to a “best call” depend on hiding uncertainty. An abstention is a valid result. A user can inspect the supported descriptive profile without receiving a fabricated ranking.

## Disclosure requirements at release

Each model card states its intended use, eligible seasons and play cohorts, source dates, label gaps, feature timing, training and evaluation protocol, metrics, uncertainty definition, known failure slices, and deployment domain. Each saved result identifies its snapshot and model bundle.

The README distinguishes implemented, experimental and planned features. Examples with invented values are labeled synthetic. A resume claim includes only completed work and measured results. If the transformer fails to improve the incumbent, publish that result and serve the better-supported model.

The project owner reviews source terms and public wording before release. The research reviewer checks timing, action semantics and evaluation claims. In a solo project, both reviews still need recorded checklists and evidence.
