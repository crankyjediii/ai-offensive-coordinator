# ML and research architecture

This specification defines two structured-data models implemented in Python and PyTorch: Model A forecasts a defense's behavior; Model B estimates outcomes for supported offensive choices. These are proposed models, not trained systems or demonstrated results. The first release is a pregame scouting application with explicit data cutoffs. See [the data contract](02-data-sources-and-schema.md), [the taxonomy](03-football-ontology.md), and [evaluation requirements](06-evaluation-and-reliability.md) before implementation.

## 1. Prediction contracts

Let `x` be the supplied hypothetical game situation, `h` the defense's available historical snaps, `o` the offense context, `a` a candidate action, and `c` the immutable data cutoff. All retrieved features must have been available by `c`. A scenario's down, distance, personnel and formation are user inputs, not claims that an actual future snap has been observed.

**Model A:** `q(d | x, visible_offense, h, coordinator_prior, c)` returns distributions over supported defensive labels. Coverage, man/zone, rush count and box count are separate targets with separate availability masks. A missing coverage label does not remove a play from a valid rush-count task. A blitz label is used only with an agreed provider definition; a derived five-or-more-rushers label keeps its own name.

**Model B:** `m(x, o, a, q, profile, c)` returns an EPA mean estimate and unconditional event probabilities for the candidate. This is an observational conditional-outcome model. Its rankings do not establish what a different call would have caused on a particular snap.

The `pbp_baseline` mode starts with designed rush and dropback, and only for rows with defensible intent labels. A scramble belongs to dropback when its original intent is known. Fine concepts are unavailable until validated labels exist. Model A's charted heads can be absent in this mode; Model B then uses a separately trained feature contract without those heads. Do not fill a missing coverage distribution with fabricated certainty.

`charted_history` supports retrospective analysis with charted targets. It does not make those labels available before their historical publication dates. `enriched_current` requires verified, lawful, timely inputs. Capabilities belong to a season and snapshot, not merely a team name.

## 2. Baselines and model sequence

Build and retain four reference systems: league situational rates; smoothed team/coordinator rates; regularized linear/logistic models; and tree-based models on the same legal features. A random forest is a useful comparison, while histogram gradient boosting provides a practical tabular challenger. Compare each on identical rows, targets, cutoffs and action definitions. A neural model that wins only because it received richer inputs has not demonstrated an architectural advantage.

The first neural challenger is a small multilayer perceptron on aggregated history. The transformer follows only after the loader, chronological splits and baselines work. Self-attention is an established sequence architecture; this project adapts it to snap records rather than language. The original transformer paper does not establish that transformers outperform tabular methods on NFL data. [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

Suggested initial architecture: 128 historical tokens, width 128, four attention heads, two encoder layers, feed-forward width 512 and dropout 0.1. These are starting settings to test, not evidence-based optima. Ablate lengths 32, 64 and 256. Start with three independently trained seeds; increase to five only if the evaluation benefit warrants the compute.

## 3. One training example

An example is a target play plus its pregame feature snapshot. All target-game plays use history that excludes that game; later snaps in the same game cannot appear in the history. This matches the MVP's scouting workflow. Within-game adaptation would require a separately specified live availability pipeline and a new evaluation protocol.

Order historical snaps by game kickoff and provider play order, then retain the latest eligible 128 for that defense. Use stable `(game_id, play_id)` keys to break ties. Include elapsed days, games since observation and a game-boundary indicator so a bye week is not treated as one ordinary token gap. End-of-season charting can remain unavailable even when the corresponding play-by-play token is present. Each field therefore carries its own missingness flag and source-age metadata.

For batch size `B`, padded history length `L`, categorical fields `Fcat`, numeric fields `Fnum` and model width `D`:

| Tensor | Shape | Meaning |
|---|---|---|
| `history_cat` | `B × L × Fcat` | IDs for observed historical categorical fields |
| `history_num` | `B × L × Fnum` | Standardized historical numeric fields |
| `history_missing` | `B × L × Fnum` | Explicit missing numeric values |
| `padding_mask` | `B × L` | `True` for padded positions |
| `query_cat`, `query_num` | `B × Qcat`, `B × Qnum` | Prospective scenario and offense information |
| `prior_features` | `B × P` | Smoothed probabilities, support and age |
| `target_values`, `target_masks` | task dependent | Labels and their eligibility flags |

Reserve different categorical IDs for `PAD`, `UNKNOWN` and `NOT_APPLICABLE`. Freeze vocabularies at training; unseen players or coordinators map to unknown embeddings. Fit means, scales, clipping bounds and category mappings on the training partition only. Use median-filled numeric inputs plus missing indicators; never confuse missing box count with an observed zero.

A historical token can contain the prior play's state, available offensive attributes, observed defensive attributes and result. Those are permitted because they concern an earlier completed play and passed the availability filter. The target query cannot contain realized coverage, pressure, air yards, target route, completion, actual gain or any outcome-derived candidate attributes. Planned route or personnel inputs require a supported product capability and compatible training labels.

## 4. Model A: history encoder and defensive heads

Embed categorical fields separately, concatenate their embeddings with standardized numerical fields and missingness indicators, then project into `D=128`. Add learned position and token-type embeddings. Append one query token containing the current scenario, known offensive presentation and prior features. The resulting encoder input has shape `B × (L+1) × D`. Read the query token's final hidden state for prediction.

Use `torch.nn.TransformerEncoderLayer(batch_first=True)` and `TransformerEncoder`; pass the padding mask explicitly. The MVP encodes one complete, already-eligible historical prefix per example, so bidirectional attention within that prefix is allowed. It contains no future observations. If an optimization later predicts multiple next snaps in one sequence, introduce a causal mask and shift target labels. A causal mask cannot repair features that were published after the cutoff. PyTorch documents separate attention and padding masks and cautions that an incorrect causal hint can produce incorrect execution. [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html).

A history-free case retains the unmasked query token. Masked pooling, if used instead, must divide by the count of real tokens and handle zero-history rows explicitly. Initialize cloned encoder layers deliberately and record the scheme. Unit checks should prove that changing padded values cannot change a prediction and that adding any post-cutoff row leaves a snapshot unchanged.

Forecast coverage with a categorical softmax over the audited coverage vocabulary. Derive man/zone from coverage only when the mapping is exhaustive for those classes; otherwise use a separate masked target and report consistency checks. Use categorical distributions for bounded integer box and rush counts, with documented overflow buckets where needed. Their expectations are sums over count probabilities. Independent heads provide marginals, not a joint distribution of all defensive attributes.

For task `j`, compute loss only where `m_ij=1`:

`L_A = Σ_j λ_j [Σ_i m_ij CE(y_ij, p_ij) / max(1, Σ_i m_ij)]`.

Skip a head's contribution when its batch contains no valid labels. Normalize each head separately so densely labeled play-by-play rows cannot drown out charting tasks. Begin with equal task weights after normalization, inspect gradient scales and tune weights using earlier validation folds. Prefer representative sampling with ordinary cross-entropy. Class weighting or oversampling changes probability training and requires validation and recalibration; it is not a free fix for rare coverages.

Model A conditions on what the defense can observe before the snap: formation, personnel or motion if supplied and supported. It does not receive the offense's unobserved final target or realized run gap. Candidate actions with identical visible presentation share this defensive forecast in the initial architecture. This is a simplifying model assumption, not a claim that hidden offensive intent is statistically independent of defensive behavior.

## 5. Model B: candidate-conditioned outcome model

Concatenate the scenario representation, lagged offense-strength features, opponent-adjusted defense profile, candidate embedding and Model A's predicted probabilities. Feed this representation through a small multilayer network, initially two hidden layers of width 128 with dropout 0.1. Batch candidates as `B × A × F`, where `A` is the number of supported actions. Mask invalid candidates before sorting. The history transformer remains the primary sequence model; a second transformer in Model B is an experiment, not a requirement.

Only the actually chosen action has observed training outcomes. One training row contributes loss for that action, not invented labels for every alternative. At scoring time the same head evaluates each eligible candidate. Never label the action with the best realized outcome in a group of unrelated plays as the correct call for this snap.

Use these initial outputs:

| Output | Training label and interpretation |
|---|---|
| EPA | Provider/versioned EPA on the included play cohort |
| Success | `EPA > 0`, not a down-specific yardage rule |
| Explosive play | Official gain at least 10 yards for designed rush, 20 for dropback |
| Sack | Provider sack event; structural zero for a verified designed rush |
| Turnover | Interception or lost fumble on the play; failed fourth down is separate |
| First down or touchdown | Explicit combined event, separately from success |
| Expected yards, optional | Official play gain under the documented penalty policy |

Event probabilities are unconditional within each candidate. A dropback explosive rate includes sacks and incompletions in its denominator. Do not compare a completion-conditional pass rate with an all-rush rate. A structurally impossible event is distinct from a missing label. Missing targets remain masked and never become negative examples.

A Huber EPA objective is useful for robust initial training, but its optimum is generally a robust location, not the conditional mean. Therefore use Huber for warm-up or an auxiliary head, then fit the production EPA mean head using squared error on the original, untrimmed target scale. If this stage or mean validation fails, label its output `robust_epa_score`; do not expose it as expected EPA. Do not silently clip true extreme outcomes to improve apparent errors.

The proposed final objective is `L_B = λ_mean MSE(EPA, μ) + Σ_j λ_j BCE(event_j, p_j)`, with per-head missing-label normalization. Keep the Huber auxiliary term only if it improves validation. EPA already reflects sacks, turnovers and explosive gains, so multi-task learning helps representation but does not justify adding those quantities to utility automatically. Optional quantile heads use pinball losses at 0.05, 0.50 and 0.95; they predict outcome quantiles and need separate validation.

Do not multiply independent Model A marginals to pretend to integrate a joint defensive state. The initial Model B directly consumes the forecast vector and learns its predictive association with outcomes. A later joint latent-state model must represent dependence, train using the same information available at inference, and demonstrate gains against this direct approach.

## 6. Cross-fitting and training-serving parity

Training Model B on Model A's in-sample forecasts would give it unrealistically clean inputs. Generate Model A features using expanding historical blocks: fit A on earlier games, calibrate on a later but still earlier block, and forecast the next block without fitting on its labels. Persist these predictions with `a_bundle_id`, training boundary, calibration boundary and target game key. The earliest block supplies burn-in history and receives no stacked training rows unless a separately trained earlier model exists.

Fit B only from these out-of-time forecasts. For an outer validation or test block, train A and B using their allowed preceding data and freeze both before predicting it. If A is later refit on more history, rebuild B's temporally generated features under the same recipe and assess the change. An A/B bundle cannot be mixed arbitrarily because a new A calibration can change B's feature distribution.

The feature builder is a shared package called by training, backtests and the API. It returns ordered arrays plus `feature_schema_version`, `taxonomy_version`, `snapshot_id`, `cutoff_at` and capability flags. Store an end-to-end fixture proving that offline and API scoring of the same snapshot produce matching inputs and outputs within a declared numeric tolerance.

## 7. Offense and opponent adjustment

Start with lagged offense EPA, success, sack and explosive rates, split by eligible broad action, with minimum support and league shrinkage. Add quarterback and roster features only when identity and availability were known at the cutoff. Team IDs can capture history but cannot substitute for measured current personnel. Unknown starters use a league or team prior and a missing-status flag. A league-average offense option uses the reference distribution, not an invented player lineup.

Fit a regularized two-way historical model such as `EPA = f(situation, action) + offense_effect + defense_effect + error` on games before the cutoff. Center effects for identifiability and tune shrinkage on earlier folds. Its residualized summaries supply opponent adjustment; the target game never estimates its own opponent rating. Compare this with a leave-one-opponent-out historical summary, and report instability for teams with few distinct opponents. These are adjustments for measured schedule composition, not causal estimates of coach or player ability.

## 8. Ranking, explanations and reproducibility

Default ranking uses the validated EPA mean estimate. An optional user utility can be `μ_EPA + w_s P(success) + w_e P(explosive) - w_t P(turnover) - w_k P(sack)`. Each weight has units of EPA per unit probability; a weight of one adds one EPA only at probability one. The UI must explain double counting and preserve the selected weights with the request. Do not switch weights silently based on score or time. Win-probability optimization needs its own validated objective.

Candidate support, data age, distribution shift and ranking instability can trigger abstention. A small ensemble spread is not proof of accuracy; see [reliability](06-evaluation-and-reliability.md). Explanation templates read persisted facts and distinguish observed tendency, model forecast and hypothesis. A later text LLM may phrase approved evidence but cannot invent calls, probabilities or source claims. Retrieved film notes must have provenance and access rights.

Record seeds, code revision, dependency lock, hardware, precision, training manifests, feature hashes and all selected hyperparameters. Use evaluation mode and inference without gradient tracking. PyTorch does not promise identical results across releases or platforms; deterministic settings have performance tradeoffs, so promise reproducibility within a tested environment and measure cross-device tolerance. [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Completion requires a baseline comparison, leakage tests, model cards, calibration artifacts, candidate eligibility checks and a saved bundle that can reproduce one full recommendation from an immutable snapshot. Architecture diagrams and untrained model classes alone do not satisfy that gate.

Sources verified: 2026-09-27. Architecture sizes, losses, thresholds and workflows above are project design proposals; cited publications do not validate this NFL application.
