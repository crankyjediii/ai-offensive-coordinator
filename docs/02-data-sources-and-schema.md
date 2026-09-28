# Data sources, schema, and availability

This document specifies the proposed warehouse and its evidence requirements. Provider documentation and source code were checked on September 27, 2026. No season assets were downloaded or audited for this documentation task; all coverage counts, freshness measurements, and release approvals remain implementation work. Data is allowed into a model only when its meaning, provenance, and availability support that use.

## 1. Source plan and operating modes

Use Python's `nflreadpy` loaders and preserve the downloaded inputs before transforming them. Its published API provides Polars data frames and loaders for the core datasets in this plan. Pin an installed, tested package version; loader defaults must never silently select a new season. [nflreadpy loader reference](https://nflreadpy.nflverse.com/api/load_functions/)

| Source | Project use | Initial acquisition and timing decision |
|---|---|---|
| nflverse play-by-play (PBP) | Situations, broad action labels, realized outcomes, team aggregates | Request 2022–2026 explicitly; inventory returned games and revisions. Required for every mode. |
| FTN charting via nflverse | Box count, motion, quarterback position, play-action/screen/RPO flags, rushing counts | Separate input from participation. Request 2022 onward and measure field coverage by season/team. |
| nflverse participation | Historical coverage, player participation, formation/personnel, pressure and route labels where populated | Historical research input. Do not assume that coverage labels exist throughout the loader's full date range. |
| Schedules and team reference | Canonical game identity, season type, kickoff, team aliases | Keep schedule revisions; distinguish scheduled kickoff from actual game completion. |
| Rosters, weekly rosters, player IDs | Identity mapping and lagged offensive/defensive context | Join with effective dates and release evidence. A final roster is not a historical roster. |
| Depth charts | Optional pregame roster scenario | Use timestamped records and preserve source-era differences. |
| Snap counts and weekly NGS statistics | Lagged playing-time weights and offensive strength features | Aggregate context only; never substitute these for snap tracking. |
| Team announcements and documented coaching reports | Coordinator tenure and playcaller evidence | Curated registry with URLs, effective dates, publication dates and uncertainty. |
| Licensed charting or lawful manual annotations | Future timely coverage or multi-player concepts | Optional. Require explicit field definitions, permitted use, latency evidence and quality audit. |

Published cadence is evidence for planning, not a guarantee that a particular game is ready. nflverse describes PBP updates after game days and recommends a later refresh for statistical corrections. Its FTN ingestion runs several times daily, while actual delivery depends on FTN. Participation from 2023 onward is released after the postseason, with no in-season updates. [nflverse availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)

The FTN charting loader describes charting within 48 hours and coverage beginning in 2022. Its licensing notice requires FTN attribution via nflverse and identifies CC-BY-SA 4.0. Treat the timing as a published expectation and record observed latency separately. [FTN charting loader](https://nflreadr.nflverse.com/reference/load_ftn_charting.html)

Participation has a source change: pre-2023 records originate from NFL NGS; subsequent records come from FTN. Preserve that distinction in every manifest and model card. The Python implementation also restricts the current-season request before its final-week condition, despite the generic loader docstring. Do not bypass that guard and assume a usable asset exists. [Participation loader and attribution](https://nflreadr.nflverse.com/reference/load_participation.html), [Python implementation](https://github.com/nflverse/nflreadpy/blob/main/src/nflreadpy/load_participation.py)

The capability service exposes three modes:

| Mode | Required evidence | Allowed product behavior |
|---|---|---|
| `pbp_baseline` | Audited PBP, valid broad intent labels, frozen features/model | Pregame scouting and supported comparison of `designed_rush` versus `dropback`. |
| `charted_history` | Audited historical charted fields and documented retrospective availability | Historical coverage/personnel analysis and charted experiments. Label reconstructed tests clearly. |
| `enriched_current` | Lawful current-season source with required labels available by cutoff | Current charted profiles and only the candidate distinctions validated on those labels. |

An FTN motion or box-count field does not establish timely coverage availability. Enable individual heads and controls independently. A current-season coverage chart can show a **forecast from historical evidence**, with its age and source, even when current observed coverage is unavailable. It cannot present that forecast as current charting.

## 2. Ingestion and immutable releases

Store raw assets under `data/raw/{provider}/{dataset}/{retrieval_id}/`; normalized data under `data/silver/{schema_version}/`; releases under `data/snapshots/{snapshot_id}/`. These are proposed repository paths, not supplied datasets. Each retrieval records URL, provider, package version, request parameters, UTC retrieval time, HTTP/release metadata when available, file checksum, schema fingerprint, row count and license reference. Do not overwrite a previously used raw file.

A snapshot is an immutable manifest of exact input checksums, transformations, normalization dictionaries, taxonomy version, quality report and cutoff. New corrections create a new snapshot. Comparing manifests identifies which games, labels and metrics changed. Large data stays outside Git; source adapters, contracts and small synthetic fixtures belong in Git.

The initial audit must enumerate requested seasons versus returned assets, expected completed games versus returned games, duplicate keys, unresolved joins, per-field non-null proportions, new categorical values and source publication evidence. Split results by season, team, week and provider. A column's presence alone does not grant a capability. Publish denominator counts for every proportion.

## 3. Canonical entities and keys

The following is the proposed logical schema, not a claim that providers supply these tables. `string` maps to Arrow string/DuckDB VARCHAR; identifiers never become floating-point values. `timestamp` means timezone-aware UTC. `?` marks nullable fields. Every normalized source row includes `record_version_id`, `source_asset_id`, `source_row_key`, `ingested_at`, `source_available_at?` and `source_available_basis`.

| Table and grain | Primary key | Required contents and relationships |
|---|---|---|
| `source_asset`: one retrieved asset version | `source_asset_id:string` | Provider/dataset, checksum, source URL, retrieval timestamp, schema hash, publication timestamp if supported, license URL. |
| `game`: one game version | `(game_id, record_version_id)` | `season:int16`, `season_type:string`, `week:int16`, home/away team IDs, `kickoff_at:timestamp?`, completion status/time, schedule provenance. |
| `play`: one normalized PBP event version | `(game_id, play_id, record_version_id)` | `play_id:int64`, `drive_id:int32?`, `play_order:int32`, teams, situation, event type, timing basis and source reference. |
| `charting_observation`: one provider/play version | `(provider, game_id, play_id, record_version_id)` | Optional charted attributes, provider schema era, typed values, label masks and conflict flags. |
| `play_participant`: one player/side/play version | `(game_id, play_id, player_id, side, record_version_id)` | `side` offense/defense, provider position, role if documented, identity confidence and asset reference. |
| `player_identity`: one provider ID mapping interval | `(provider, provider_player_id, valid_from)` | Canonical `player_id:string`, `valid_to:timestamp?`, mapping evidence. Names are display fields. |
| `roster_membership`: player/team interval version | `(player_id, team_id, valid_from, record_version_id)` | `valid_to?`, position, roster status, effective-date confidence, known-at time. |
| `coordinator_tenure`: staff/team interval version | `(team_id, tenure_id, record_version_id)` | Coordinator ID, playcaller ID if supported, effective start/end, announcement evidence, role confidence. See [prior methodology](05-coordinator-priors.md). |
| `play_label`: one task label per play/version | `(game_id, play_id, label_name, taxonomy_version, record_version_id)` | Typed label, evidence level, source-row references, eligibility flag, exclusion reason. |
| `feature_snapshot`: one frozen prediction context | `(snapshot_id, feature_context_id)` | `cutoff_at`, game/synthetic scenario, offense/defense, feature schema, feature values and source lineage. |
| `defense_profile`: one defense/cutoff/context | `(snapshot_id, defense_id, context_key)` | Counts, weighted counts, effective sample sizes, tendencies, intervals, prior version. |

Treat `(game_id, play_id)` as the intended cross-source join key only after uniqueness checks pass for the relevant asset. Retain drive and source row identifiers for diagnosis. Reject ambiguous matches; adding drive to a join without investigating can conceal upstream duplication. Keep nflverse game IDs verbatim through source adapters and map team aliases separately. Do not reconstruct IDs from an assumed home/away order.

## 4. Field mappings and feature roles

These mappings are adapter contracts. Validate names and values against each downloaded schema. Preserve originals alongside normalized values. The PBP source documents action indicators, preplay state and outcome fields; its play type alone is insufficient to distinguish a scramble from a designed run. [Machine-readable PBP dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv)

| Provider fields | Proposed canonical fields | Use |
|---|---|---|
| PBP `game_id`, `play_id`, `drive` | Same game/play IDs; `drive_id` | Identity, ordering checks. |
| `posteam`, `defteam` | `offense_id`, `defense_id` | Context. |
| `qtr`, `quarter_seconds_remaining`, `down`, `ydstogo`, `yardline_100` | `quarter`, `seconds_remaining`, `down`, `distance`, `yards_to_goal` | Preplay state; distinguish overtime. |
| `score_differential`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining` | `offense_score_margin`, timeouts | Preplay state. |
| `qb_dropback`, `qb_scramble`, `rush_attempt`, `qb_kneel`, `qb_spike` | Broad intent evidence and exclusion flags | Historical labeling; candidate input is supplied independently at inference. |
| `epa`, `yards_gained`, `sack`, `interception`, `fumble_lost` | Outcomes and event flags | Labels only on the target play. |
| `run_location`, `run_gap`, `pass_location`, `air_yards` | Realized direction/gap/target descriptors | Retrospective labels, never current decision inputs. |

| Charting fields | Proposed canonical fields | Adapter rule |
|---|---|---|
| FTN `nflverse_game_id`, `nflverse_play_id` | `game_id`, `play_id` | Rename before validated join. |
| `starting_hash`, `qb_location` | `hash`, `qb_alignment` | Explicit categorical lookup; unseen values quarantine. |
| `n_defense_box`, `n_offense_backfield` | `box_count`, `backfield_count` | Nullable integers. |
| `is_motion`, `is_play_action`, `is_screen_pass`, `is_rpo` | `motion_observed`, `play_action_observed`, `screen_observed`, `rpo_observed` | Nullable booleans; use as historical labels only after available. |
| `n_blitzers`, `n_pass_rushers`, `date_pulled` | `blitzer_count`, `pass_rusher_count`, `provider_pulled_at` | Counts retain source semantics; retrieval is not first publication. |

FTN's field names and semantics come from its [charting dictionary](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html). No geometric alignment or complete offensive play-call label is promised by this mapping.

Participation maps `nflverse_game_id`/`play_id` to the same join key; `offense_personnel`/`defense_personnel` to parsed position counts; `offense_formation` to a source-preserving formation field; and `defense_coverage_type`, `defense_man_zone_type`, `was_pressure`, `route` to separate historical labels. Explode `offense_players` and `defense_players` into participant rows after checking duplicate IDs and position-list alignment. Do not assume that every season populates every field. [Participation dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html)

The current depth-chart dictionary uses `dt` and player/position/rank fields after the 2024 source change. Parse `dt` with an explicit timezone policy and retain the original string; it documents loading, not necessarily the instant a coach changed a lineup. [Depth-chart dictionary](https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html)

NGS data in this plan consists of player-level summary measures, such as passing and receiving aggregates. A summary value does not reveal player coordinates on a particular play. Lag it by its supported publication time. [NGS summary dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)

### Null and conflict semantics

Store unavailable numeric values as null and use a missingness mask. Never fill unknown box counts with zero or unknown blitz flags with false. A categorical field has a nullable raw value plus `missing_reason`: `not_collected`, `not_applicable`, `not_yet_available`, `unmapped`, `source_conflict` or `unknown`. Training may encode these as separate reserved tokens; reporting must preserve the distinction.

If two providers disagree, retain both observations. A versioned source-precedence rule may select one for a specific label after measuring disagreement. It must not silently average categorical labels. `BLOWN` is a supplied annotation, not missing coverage; unknown is not `OTHER`, which means an observed value explicitly pooled by the model vocabulary.

## 5. Availability and as-of joins

Maintain four distinct times: `event_at` (when the underlying event happened), `source_available_at` (earliest supported publication of that exact version), `ingested_at` (local retrieval), and `cutoff_at` (the prediction's information boundary). Preserve `event_time_precision` such as `play`, `game_end`, or `date_only`. A game clock is not a wall-clock timestamp.

For forward operation, `known_at = max(source_available_at, ingested_at)` when publication is documented; if publication is unknown, local ingestion is the conservative known-at time. For a historical replay, use a genuinely archived provider version with publication evidence. A current retrieval of an old game does not prove that its final labels were available before a historical cutoff.

```text
eligible_version(row, context):
    row belongs to a game completed before context.cutoff_at
    and row.known_at <= context.cutoff_at
    and row is included in context.snapshot_id
    and row's effective interval applies to the requested date
select the latest eligible version, breaking ties by stable version ID
```

At MVP pregame cutoffs, remove every snap from the game being evaluated from defensive histories and aggregates. Player strength, opponent adjustment and coach priors use completed earlier games only. The hypothetical down, distance and clock may be specified by a user, but future game outcomes cannot enter that scenario's features.

Historical training labels can be learned from later charting only in training folds where the labels were available by the model-fit cutoff. A test target may be scored when its label later becomes available; it must not enter a feature, prior, calibration fit, early stopping decision or model selection first. Missing historical archives require the label **revised-data historical test**. A conservative delay simulation is a separate experiment with explicit assumptions, never a replacement for release evidence.

## 6. Cohorts and quality gates

Default training/evaluation cohort: regular-season scrimmage plays with valid preplay state, resolvable teams, a supported broad action and the required target mask. Exclude special teams, conversions, kneels, spikes, administrative rows and nullified/no-play events. Keep sacks and scrambles under dropback when intent evidence is valid. Do not exclude interceptions, fumbles, sacks or negative plays to improve model fit.

The primary outcome cohort uses valid, non-nullified snaps with available EPA and no flagged penalty. This is explicitly a **penalty-free outcome estimand**, so results cannot claim whole-playcalling policy value. Maintain a second penalty-inclusive cohort with preserved action, valid event accounting and official resulting state; compare selection and results before promoting it to the primary cohort. Nullified or ambiguous penalty plays stay in a separate audit table. Charted tendency cohorts may retain valid descriptive labels from other snaps under their own documented masks; never reuse their denominator for outcome metrics.

| Check | Proposed release behavior |
|---|---|
| Duplicate canonical keys or many-to-many enrichment joins | Fail affected partition; report before/after row counts and diagnostic keys. |
| Unknown enum value or changed field type | Quarantine affected field/rows; disable dependent capability until adapter reviewed. |
| Impossible state or non-finite numeric target | Quarantine; report by source, game and field. Do not silently clip errors. |
| Missing critical state/identity | Exclude from model cohort and retain in raw/audit layers. |
| Missing optional charting | Preserve PBP row; mask target/features and lower capability scope. |
| Insufficient labeled team/context support | Return `limited` or `abstained`; thresholds are selected in past validation folds. |
| Timing violation in a feature lineage | Fail snapshot publication. |
| New release revises old values | Create new snapshot and a revision report; never change a model's original training manifest. |

Use hard integrity gates for key uniqueness, finite values and temporal lineage. Treat feature completeness as a measured capability policy: do not require every optional field to be 100% complete or invent a universal percentage threshold. Personnel counts should generally total eleven on each side, but irregular legal/penalty situations and source errors require retained evidence, not automatic fabrication of missing players.

## 7. Manifest contract and completion criteria

The [example manifest](../contracts/data-manifest.example.json) is a valid JSON starter with empty audit results; it is not a published dataset. Required production additions are exact asset hashes, row counts, completed-game coverage, field capability decisions, temporal evidence, lineage paths and signed-off quality results. `audit_status=not_run` prohibits model training and public claims of coverage.

Phase completion requires one reproducible historical snapshot, a separate forward archive, tested source joins, a capability response for each requested team/season, a null/conflict report, and a demonstrated cutoff test that fails when a future record is injected. Refer to [evaluation](06-evaluation-and-reliability.md) for split design and [MLOps](10-deployment-and-mlops.md) for promotion and rollback.
