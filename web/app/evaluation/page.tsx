"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import { Fragment, useEffect, useState } from "react";
import { useDeployment } from "@/components/Chrome";
import { ErrorNote, SectionHead } from "@/components/ui";
import { getModelCard, pct } from "@/lib/api";

const f3 = (x: number | null | undefined) => (x === null || x === undefined ? "—" : x.toFixed(4));
const rel = (d: any) => (d?.relative_change === null || d?.relative_change === undefined ? "—" : `${(d.relative_change * 100).toFixed(2)}%`);
// Interval of the paired difference expressed relative to the incumbent's mean loss, matching the Δ column.
const ci = (d: any) => {
  if (!d || !d.relative_change) return "—";
  const base = d.mean / d.relative_change;
  return `[${((d.ci_low / base) * 100).toFixed(2)}%, ${((d.ci_high / base) * 100).toFixed(2)}%]`;
};

function Table({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
      <table>
        <caption>{caption}</caption>
        <thead><tr>{head.map((h, i) => <th key={h} scope="col" className={i ? "num" : undefined}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r) => <tr key={String(r[0])}>{r.map((c, i) => i === 0 ? <th key={i} scope="row">{c}</th> : <td key={i} className="num">{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

export default function Evaluation() {
  const { data } = useDeployment();
  const [card, setCard] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => { if (data) getModelCard(data.model_bundle_id).then(setCard).catch(setError); }, [data]);
  if (error) return <ErrorNote error={error} />;
  if (!card) return <p className="muted">Loading model card…</p>;
  const selA = card.selection?.model_A ?? {};
  const selB = card.selection?.model_B ?? {};
  const decA = card.model_a.selection ?? {};
  const decB = card.model_b.selection ?? {};
  const testA = card.locked_test_2025?.model_A ?? {};
  const testB = card.locked_test_2025?.model_B ?? {};
  const heads = ["box", "rush", "blitz", "coverage", "man", "pressure"];
  return (
    <>
      <p className="eyebrow">Evaluation · model card</p>
      <h1>What was measured</h1>
      {card.is_quick_smoke_bundle && <p className="note" role="alert">This is a quick smoke-test bundle; its numbers are not evaluation results.</p>}
      <div className="grid cols-3" style={{ marginTop: 24 }}>
        <div className="card reveal"><p className="eyebrow">Model A served</p><p className="big-num">{card.model_a.served.replaceAll("_", " ")}</p>
          <p className="small">Incumbent: {decA.incumbent} · best neural: {decA.best_neural} · gate {decA.neural_passes_gate ? "passed" : "not passed"}</p></div>
        <div className="card reveal"><p className="eyebrow">Model B served</p><p className="big-num">{card.model_b.served}</p>
          <p className="small">Incumbent: {decB.incumbent} · MLP gate {decB.pbp_mlp_passes_gate ? "passed" : "not passed"} · contract {card.model_b.feature_contract}</p></div>
        <div className="card reveal"><p className="eyebrow">EPA interval coverage (2025, nominal {pct(testB.interval?.nominal_coverage, 0)})</p>
          <p className="big-num">{pct(testB.interval?.empirical_coverage_all?.mean)}</p>
          <p className="small">Game-clustered 95%: {pct(testB.interval?.empirical_coverage_all?.ci_low)} – {pct(testB.interval?.empirical_coverage_all?.ci_high)}</p></div>
      </div>

      <SectionHead title="Decision rule" tag="frozen before the test" />
      <p className="card" style={{ maxWidth: "none", whiteSpace: "pre-wrap" }}>{String(card.decision_rule ?? "").replace(/^\(frozen before the test is opened\):\s*/, "")}</p>
      <dl className="kv small" style={{ marginTop: 12 }}>
        {Object.entries(card.splits ?? {}).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{String(v)}</dd></Fragment>)}
        <dt>Temporal evidence</dt><dd>{card.temporal_evidence}</dd>
      </dl>
      <p className="note small" style={{ marginTop: 10 }}>{card.temporal_evidence_note}</p>

      <SectionHead title="Model A · development windows" tag="sum of per-head log loss (lower is better)" />
      <Table caption="Paired comparison against the incumbent baseline on identical rows; interval is game-clustered bootstrap"
        head={["Model", "Window 1 (2024 wk1–12)", "Window 2 (2024 wk13–18)", "Δ vs incumbent W1", "95% CI of Δ (W1)", "Δ W2"]}
        rows={Object.entries(selA).map(([k, v]: [string, any]) => [k, f3(v.dev_window_1.primary_sum_log_loss), f3(v.dev_window_2.primary_sum_log_loss), rel(v.vs_incumbent_window_1), ci(v.vs_incumbent_window_1), rel(v.vs_incumbent_window_2)])} />
      <div style={{ height: 14 }} />
      <Table caption="Per-head log loss, development window 1"
        head={["Model", ...heads]}
        rows={Object.entries(selA).map(([k, v]: [string, any]) => [k, ...heads.map((h) => f3(v.dev_window_1.per_head_log_loss[h]))])} />

      <SectionHead title="Model B · development windows" tag="feature contract : model family" />
      <Table caption="EPA mean squared error and event log loss; paired Δ in squared error vs incumbent"
        head={["Contract : model", "EPA MSE W1", "EPA MSE W2", "Bias W1", "Success LL W1", "Sack LL W1", "Δ vs incumbent W1", "95% CI (W1)"]}
        rows={Object.entries(selB).map(([k, v]: [string, any]) => [k, f3(v.dev_window_1.epa_mse), f3(v.dev_window_2.epa_mse), f3(v.dev_window_1.epa_bias), f3(v.dev_window_1.success_log_loss), f3(v.dev_window_1.sack_log_loss), rel(v.vs_incumbent_window_1), ci(v.vs_incumbent_window_1)])} />

      <SectionHead title="Locked 2025 test" tag="reported once" />
      <div className="grid cols-2">
        <div>
          <Table caption="Model A per-head log loss (2025)" head={["Model", ...heads]}
            rows={Object.entries(testA).filter(([, v]: [string, any]) => v.per_head_log_loss).map(([k, v]: [string, any]) => [k, ...heads.map((h) => f3(v.per_head_log_loss[h]))])} />
          {testA.served_vs_incumbent && <p className="small" style={{ marginTop: 8 }}>Served vs incumbent (stacked loss): Δ {rel(testA.served_vs_incumbent)} · 95% CI {ci(testA.served_vs_incumbent)}</p>}
          {card.locked_test_2025?.coverage_accuracy !== undefined && card.locked_test_2025?.coverage_accuracy !== null && (
            <p className="small">Coverage top-1 accuracy {pct(card.locked_test_2025.coverage_accuracy)} vs majority-class rate {pct(card.locked_test_2025.coverage_majority_class_rate)} — accuracy alone rewards common classes; log loss is primary.</p>)}
        </div>
        <div>
          <Table caption="Model B on 2025 (penalty-free cohort)" head={["Model", "EPA MSE", "Bias", "Success LL", "Explosive LL", "Turnover LL"]}
            rows={Object.entries(testB).filter(([, v]: [string, any]) => v && v.epa_mse !== undefined).map(([k, v]: [string, any]) => [k, f3(v.epa_mse), f3(v.epa_bias), f3(v.success_log_loss), f3(v.explosive_log_loss), f3(v.turnover_log_loss)])} />
          {testB.served_vs_incumbent_epa_sq_error && <p className="small" style={{ marginTop: 8 }}>Served vs incumbent EPA squared error: Δ {rel(testB.served_vs_incumbent_epa_sq_error)} · 95% CI {ci(testB.served_vs_incumbent_epa_sq_error)}</p>}
        </div>
      </div>

      {testB.event_calibration && (
        <>
          <SectionHead title="Event calibration (2025)" tag="fixed bins · denominators shown" />
          <div className="grid cols-2">
            {Object.entries(testB.event_calibration).map(([e, c]: [string, any]) => (
              <Table key={e} caption={`${e.replaceAll("_", " ")} · calibration slope ${c.slope.toFixed(2)}, mean pred ${pct(c.mean_pred)} vs observed ${pct(c.observed_rate)}`}
                head={["Bin", "n", "Mean predicted", "Observed"]}
                rows={c.table.filter((b: any) => b.n > 0).map((b: any) => [`${b.bin_lo.toFixed(1)}–${b.bin_hi.toFixed(1)}`, b.n, pct(b.mean_pred), pct(b.observed_rate)])} />
            ))}
          </div>
        </>
      )}

      {testB.recommendation_diagnostics && (
        <>
          <SectionHead title="Ranking diagnostics (2025 situations)" tag="not policy value" />
          <dl className="kv">
            <dt>Dropback ranked first</dt><dd>{pct(testB.recommendation_diagnostics.share_dropback_ranked_first)}</dd>
            <dt>Median |margin|</dt><dd>{testB.recommendation_diagnostics.median_abs_margin_epa.toFixed(4)} EPA</dd>
            <dt>Margin below seed spread</dt><dd>{pct(testB.recommendation_diagnostics.share_margin_below_spread)}</dd>
            <dt>Seed rank agreement</dt><dd>{testB.recommendation_diagnostics.seed_rank_agreement === null ? "—" : pct(testB.recommendation_diagnostics.seed_rank_agreement)}</dd>
          </dl>
          <p className="note small" style={{ marginTop: 10 }}>{testB.recommendation_diagnostics.note}</p>
        </>
      )}

      {card.R01_priors?.evaluation && (
        <>
          <SectionHead title="R01 · Do priors help early-season forecasts?" tag="2024, mean log loss" />
          {Object.entries(card.R01_priors.evaluation).map(([slice, heads2]: [string, any]) => (
            <div key={slice} style={{ marginBottom: 14 }}>
              <Table caption={`${slice.replaceAll("_", " ")}`} head={["Target", "League only", "Team only", "Prior (no current)", "Current only", "Mixture + current", "n", "Mixture vs league Δ"]}
                rows={Object.entries(heads2).map(([h, v]: [string, any]) => [h, f3(v.league_only), f3(v.team_only), f3(v.prior_mixture_no_current), f3(v.current_only), f3(v.mixture_with_current), v.n, `${rel(v.mixture_vs_league)} ${ci(v.mixture_vs_league)}`])} />
            </div>
          ))}
          <p className="small muted">Chosen prior: {JSON.stringify(card.R01_priors.chosen)} · coordinator component: {card.R01_priors.coordinator_component}</p>
        </>
      )}

      <SectionHead title="Known limitations" tag="enforced in the product" />
      <ul>{card.known_limitations.map((l: string) => <li key={l}>{l}</li>)}</ul>
      <p className="small muted">Hardware: {card.hardware?.platform} · train device {card.hardware?.train_device} · torch {card.hardware?.torch} · runtime {Math.round(card.runtime_seconds / 60)} min · experiment {card.experiment_id}</p>
    </>
  );
}
