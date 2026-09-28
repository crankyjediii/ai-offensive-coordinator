# Project overview and product vision

The NFL AI Offensive Coordinator is a Python and PyTorch portfolio project for studying an opponent's defense and comparing offensive play families in a specified game situation. It combines a dated defensive profile, a model of likely defensive behavior, an outcome model, and an interface that explains the available evidence. The product is a scouting and decision-support application. Its recommendations are model estimates from observational football data.

**Specification version:** 1.0. **Planning date:** September 27, 2026. **Implementation status:** planned. No training results, completed application, or measured performance are claimed in this suite.

## Project origin and requirements

The project began with a request to train a language model on defensive personnel, alignment, coverage, and tendencies, use Python and PyTorch, let a user select any NFL defense during the current season, and incorporate the defensive coordinator's previous-season tendencies while the current sample is small. The intended result is a technically defensible resume project with accurate, reliable, and repeatable predictions.

The design retains those goals and uses a structured-data transformer as the main neural model. A text LLM is optional for later explanations. Predictive inputs are mostly categorical fields, numeric measurements, and sequences of defensive snaps; generating prose does not establish predictive accuracy.

The original CHI offense versus GB defense scenario is retained as a demo specification: 2nd and 7, ball on the GB 38, 9:42 in the second quarter, with optional 11 personnel and left hash when the selected model supports them. The earlier example EPA, coverage rates, success percentages, and confidence scores were illustrative. They are not findings about either team. No real coordinator assignment is assumed in this specification.

## Product promise

A user selects an opponent, an offense or league-average offensive context, a season, and a data cutoff. Analyst mode describes supported tendencies. Playcaller mode compares eligible candidate families for a user-entered situation, reports estimated outcomes, and returns an evidence status. It can return no recommendation.

The application makes five specific commitments:

1. Each result identifies the exact snapshot and model bundle used.
2. Observed historical tendencies, predicted current behavior, and unavailable information are visually distinct.
3. The same saved inputs, snapshot, model bundle, and scoring settings produce the same ordered result under the supported runtime.
4. Recommendations use only features available at the stated decision time.
5. Missing labels or weak support reduce the output's scope rather than becoming invented football detail.

These are acceptance criteria to implement and test, not statements about an existing system.

## Users and decisions

| User | Task | Useful result | Evidence required |
|---|---|---|---|
| Project developer | Investigate a tendency or failed prediction | Filtered plays, source fields, uncertainty, saved query | Reproducible snapshot and feature trace |
| Football analyst or fan | Compare a defense across situations or seasons | Conditional rates with denominators and historical context | Clear observed versus predicted labels |
| Portfolio reviewer | Assess software and research quality | Demo, model card, chronological backtest, source audit | Runnable build and honest measurements |
| Playcaller-mode user | Compare supported offensive choices | Ranked families or an abstention | Candidate support and model limitations |

The MVP is for preparation and scenario exploration. It does not consume a verified live stadium feed, call plays during games, or guarantee improved scoring.

## Capability levels

The season selector and the modeling capability are separate. All 32 current team identities can be listed, but a selected snapshot determines whether a team has supported observations and which fields can be shown.

| Mode | Intended scope | Inputs and outputs | Release condition |
|---|---|---|---|
| `pbp_baseline` | Current-season and historical broad scouting | Game state, lagged offense/defense summaries; supported designed-rush versus dropback comparisons | Play-by-play audit and intent-label rules pass |
| `charted_history` | Research on seasons with detailed labels | Historical coverage, pressure and play attributes; richer hypotheses | Season/field coverage and label quality pass |
| `enriched_current` | Current-season detailed scouting | Only timely attributes supported by an approved source | Source rights, delay, schema and measured completeness verified |

Charted historical outcomes may train a current model, but historical current-snap coverage is never silently supplied as a known current-snap input in a current-season prediction. Public FTN charting and participation are different products. Detailed participation is documented as an after-season release; field availability must be checked individually. See [Data sources and schema](02-data-sources-and-schema.md) for the provider references and source audit.

The original goal of using all available season data means all relevant, usable, licensed observations that existed before the selected cutoff. It does not mean every field ever published later about those games.

## The decision pipeline

```text
Approved source snapshots
        |
        v
As-of data warehouse and label registry
        |
        +--> Defense profile and coordinator prior
        |
        +--> Model A: defensive behavior distribution
                         |
Situation + offense + eligible candidate
                         |
                         v
              Model B: outcome estimates
                         |
                         v
          Support and uncertainty checks
                         |
                         v
         Ranked candidates or abstention
                         |
                         v
        Evidence-based explanation and UI
```

Model A answers what a defense might do given the supported situation and history. Model B estimates the outcome associated with each supported candidate. The recommendation service applies the selected scoring rule and checks whether it has enough evidence to order the candidates.

Default ranking uses estimated EPA on eligible penalty-free plays, identified by `outcome_cohort_id=reg_scrimmage_penalty_free_v1`. Penalty occurrence is unknown when a play is selected; this initial comparison therefore does not estimate the value of the entire call including penalties. A penalty-inclusive sensitivity cohort is required before broadening that claim. The interface also displays success, explosive gain, first down or touchdown, sack, and turnover probabilities on the same cohort when the corresponding heads have passed validation. User-selected risk penalties are a later feature; their units and effect must be visible.

## MVP boundaries and later additions

The first release contains a reproducible play-by-play warehouse, broad defensive reports, coordinator priors where verified, temporal baselines, a dated model card, and a web interface with a saved example. A baseline can remain the serving model if the transformer does not improve the agreed metrics.

The research release adds the PyTorch defensive sequence model, a multi-task outcome model, comparisons against simpler baselines, and calibrated uncertainty reporting. Historical coverage and play attributes are exposed only where the capability audit passes.

Later work may add manually reviewed concept labels, an approved current charting feed, player-level matchup features, a language explanation layer, or observational off-policy analysis. Full playbook calls, automated film interpretation, and an unrestricted 20–40-concept recommendation menu are outside the MVP.

Run direction does not identify a blocking scheme. A receiver's route does not identify the full passing concept. Those distinctions are formalized in [Football ontology](03-football-ontology.md).

## Requirements and acceptance evidence

| ID | Requirement | Acceptance evidence |
|---|---|---|
| P01 | Select offense, defense, season and cutoff | Saved request resolves to one valid snapshot |
| P02 | Show supported conditional defensive tendencies | Numerator, denominator, label coverage and source shown |
| P03 | Use coordinator history early in the season | Prior contribution is reproducible and improves or does not harm the agreed past-fold gate |
| P04 | Train PyTorch transformer candidates | Saved config, seeds, checkpoints and baseline comparison |
| P05 | Rank supported play families | Candidate set and exclusion reasons included in response |
| P06 | Report reliability | Calibration, support and abstention diagnostics in model card |
| P07 | Preserve temporal validity | Automated sentinel tests reject unavailable features and future rows |
| P08 | Explain output | Every numerical claim traces to returned evidence |
| P09 | Refresh season observations | Idempotent refresh produces a new manifest or reports no change |
| P10 | Support resume review | Reproducible demo, documented limits and measured results |

## What success means

Project completion is not conditional on a neural model beating every simpler model. A defensible negative result, a working baseline product, and a reproducible explanation of the transformer comparison satisfy the research objective.

Product success is measured by whether a reviewer can reproduce a saved example, understand what the system knew, locate the backtest evidence, and identify where it declines to make a detailed claim. Predictive release criteria are specified in [Evaluation and reliability](06-evaluation-and-reliability.md). Delivery milestones are in [Roadmap](08-roadmap.md).

The eventual resume should state what was built and measured. A suitable claim after implementation is a dated account of the pipeline, model comparison and evaluation design. Do not write a percentage improvement until a locked evaluation produces that number. Do not describe an observational policy estimate as wins added or points that an NFL team would actually have scored.
