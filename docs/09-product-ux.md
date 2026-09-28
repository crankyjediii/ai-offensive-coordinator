# Web application and product experience

The web application has Analyst and Playcaller modes. Both share team selection, season selection, a cutoff, and a persistent data-availability summary. A user must be able to distinguish a historical observation from a current prediction without opening a technical model card.

This document specifies a proposed interface. The documentation package is not an implemented application.

## Information structure

The initial screen provides a defense selector, an offense selector with a league-average option, a season selector, and a cutoff control. A capability request resolves the available data before charts or recommendation controls appear.

Use five navigation destinations: Overview, Analyst, Playcaller, Evaluation, and Data and Methodology. A saved analysis URL preserves non-sensitive scenario parameters and immutable identifiers. It must not contain provider credentials, raw licensed rows, or arbitrary serialized model objects.

The persistent summary says, for example, “2026 season · snapshot dated September 27 · play-by-play mode.” A more detailed panel gives UTC cutoff, source timestamps, covered games, missing fields and model bundle. The display date uses the viewer's timezone; exports preserve UTC.

## Analyst mode

Analyst mode starts with a situation selector and a readable rate chart. The first choices are down, distance band, field-position band, and game period. Additional personnel, motion, formation, coverage and box filters appear only when supported. An unsupported filter is disabled with a short explanation; it is never represented as a zero rate.

Each chart offers an accessible table. A tendency row contains its label, sample size, known-label denominator, rate, uncertainty method where valid, date range and provenance status. User filters update both the numerator and denominator. Small groups show counts before percentages and may be suppressed according to the published support policy.

Example questions carried over from the project discussion include third-and-medium behavior, blitz rates with and without motion, coverage against 12 personnel, explosive gains allowed, and change from the previous season. These are queries to implement, not findings about a named defense.

A coverage panel can show separate observed and model-predicted distributions. It must not blend historical observations with current-season predictions into one unlabeled bar. Coordinator-prior views show historical source seasons, prior effective strength and current observed support.

Offense adjustment gets a visible label and method link. A comparison against league average uses the same cohort definition and cutoff. It must not compare a red-zone subset with an all-field average while implying the denominators match.

## Playcaller input flow

The user first selects an opponent, offensive context, and capability mode. The situation form then requests:

| Input | Product behavior | Validation |
|---|---|---|
| Down and distance | Down 1–4 and positive yards to go; separate goal-to-go state | Reject inconsistent or impossible input |
| Ball position | Team-relative yard line control; canonical `yardline_100` behind the UI | Normalize before request; reject outside 1–99 |
| Quarter and time | Regulation first; overtime enabled only for supported schema/model | Time is nonnegative and within period bounds |
| Score differential | Offense score minus defense score | Explicit sign convention next to control |
| Personnel | Optional supported value with “unknown” choice | Never assume 11 personnel |
| Formation and motion | Optional capability-dependent controls | Unknown differs from no motion |
| Hash | Optional if supported by audited data/model | Never infer left or right from pass location |
| Offensive context | Selected team or league average | Unknown players do not acquire invented quality scores |

The initial demo scenario is CHI versus GB, 2nd and 7 at the GB 38, Q2 9:42. It must be labeled a saved example. Values such as 11 personnel and left hash appear only in an enriched mode that accepts those features. The demo must not prefill actual 2026 tendencies or invented trained-model results.

The MVP does not recommend special-teams calls, kneels or spikes. It may analyze offensive fourth-down plays within its trained scope, but it does not decide “go for it versus kick” without a separate decision model.

## Result layout and language

```text
SCENARIO
CHI offense / GB defense / 2nd & 7 / GB 38 / Q2 9:42
Snapshot date | Mode | Model version

DEFENSIVE EXPECTATION
Supported behavior distributions
Predicted or observed labels, each explicitly marked

CANDIDATE COMPARISON
Family | Expected EPA on eligible penalty-free plays | Metrics | Evidence
Up to three eligible choices; only two in the broad baseline

WHY THIS RESULT
Specific comparisons tied to historical rows or model outputs

RELIABILITY
Data coverage | Candidate support | Model agreement
Interval definition | Missing features | Reason codes
```

The first result sentence states the result's scope. Example approved copy for a limited response: “The available data supports a broad run-versus-dropback comparison on eligible penalty-free plays. Coverage-specific recommendations are unavailable for this snapshot.” The evidence panel explains that penalties are unknown before the call, so this initial outcome cohort does not estimate full-call value. Preserve `outcome_cohort_id` in every saved result and export.

For weak evidence, use “Insufficient evidence to rank these candidates reliably.” Keep any supported descriptive profile visible. For stale data, use “Showing the last validated snapshot from [the displayed date]” with the date supplied by the application. The literal template brackets are not displayed.

Do not put a single “confidence 78%” badge at the top. A probability of success is an event prediction; label completeness, ensemble disagreement, and ranking stability describe different things. Give each its own meaning and denominator.

## State handling

| State | What remains visible | Required behavior |
|---|---|---|
| Loading | Submitted inputs and snapshot request | Disable duplicate submit; allow cancel |
| `ok` | Supported ranking and evidence | Show limitations even when status is good |
| `limited` | Supported subset and restrictions | Identify unavailable attributes and excluded candidates |
| `abstained` | Situation, diagnostics and descriptive profile | No winner styling; explain reason and useful next adjustment |
| `unavailable` | Team/season selection and source status | No numerical fallback masquerading as a model response |
| Invalid input | User's form values | Inline validation and focus on first error |
| Service error | Last completed result if one exists | Keep its timestamp; allow retry without changing the scenario |

Changing an input marks the previous result as belonging to the previous scenario until resubmission. A late response for an older request must not overwrite a newer result.

## Explanations and exports

The first implementation uses templates populated from typed evidence. Example explanation fields include scenario, tendency estimate, observed count, prior contribution, comparison group, and excluded feature. Every displayed number must exactly match the API payload at the chosen rounding.

An optional later LLM receives only this evidence object and approved definitions. It does not query arbitrary URLs, invent a concept label, or change a ranking. Invalid output falls back to templates. The service logs explanation provenance separately from model inference.

Export a JSON analysis record and a compact PDF or HTML report only after source redistribution rules have been evaluated. Preserve model and snapshot IDs, cutoff, candidate set, metrics, reason codes and attribution. Suppress or aggregate raw provider data when required. A downloaded report must remain understandable without the live application.

## Accessibility and display

Use semantic labels, keyboard-operable controls, visible focus, text explanations of errors, and chart tables. Do not rely on red/green color alone for support or outcomes. Put sample counts and uncertainty definitions in text, not exclusively in hover content. On mobile, keep the scenario summary above the result and let comparison tables scroll within a labeled region.

Target WCAG 2.2 AA. Check contrast, zoom/reflow, focus visibility, keyboard order, and pointer target size; W3C's minimum target-size criterion is 24 by 24 CSS pixels subject to specified exceptions. Prefer larger controls in this app. These are implementation checks, not a claim of conformance. Source: [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/) and [target-size explanation](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum), checked September 27, 2026.

## UX acceptance scenarios

1. A reviewer opens the saved CHI–GB example, sees its snapshot and mode, and exports the exact analysis record.
2. A user selects a season without current coverage labels. Coverage controls are disabled and the interface says why.
3. A sparse personnel/situation combination returns an abstention without a highlighted top choice.
4. A league-average offense request produces no player-specific claim.
5. A data refresh occurs during use. The completed analysis keeps its old immutable snapshot; a new analysis can use the new one.
6. A keyboard-only user completes the situation form, reads result tables, and reaches the methodology.
7. A historical prediction is revisited after a model release. The saved bundle is used or the app explicitly reports that it cannot reproduce it.

The API schemas and version rules are defined in [Engineering architecture](07-engineering-and-stack.md). Reliability copy must stay consistent with [Evaluation and reliability](06-evaluation-and-reliability.md).
