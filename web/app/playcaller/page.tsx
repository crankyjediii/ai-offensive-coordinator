"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDeployment } from "@/components/Chrome";
import { Bar, ErrorNote, Prov, SectionHead } from "@/components/ui";
import {
  ApiRequestError, epa, fmtDate, getCapabilities, pct, postRecommendation, REASON_TEXT,
  type Capabilities, type Mode, type RecommendationRequest, type RecommendationResponse,
} from "@/lib/api";

interface Form {
  offense: string; defense: string; season: number; mode: Mode; snapshotId: string;
  down: string; distance: string; side: "own" | "opp"; yard: string; quarter: string; clock: string; score: string; goalToGo: boolean;
}
type Errors = Partial<Record<keyof Form, string>>;

const EXAMPLE: Partial<Form> = { offense: "CHI", defense: "GB", season: 2026, mode: "pbp_baseline", down: "2", distance: "7", side: "opp", yard: "38", quarter: "2", clock: "9:42", score: "0", goalToGo: false };
const MODES: [Mode, string][] = [["pbp_baseline", "Play-by-play baseline"], ["charted_history", "Charted history"], ["enriched_current", "Enriched current"]];

function yardline100(f: Form) { const y = Number(f.yard); return f.side === "opp" ? y : 100 - y; }
function clockSeconds(c: string) { const m = /^(\d{1,2}):([0-5]\d)$/.exec(c.trim()); return m ? Number(m[1]) * 60 + Number(m[2]) : NaN; }

function validate(f: Form): Errors {
  const e: Errors = {};
  const dist = Number(f.distance), yard = Number(f.yard), secs = clockSeconds(f.clock), score = Number(f.score);
  if (f.offense === f.defense) e.offense = "Offense and defense must differ.";
  if (!/^\d+$/.test(f.distance) || dist < 1 || dist > 99) e.distance = "Enter yards to go between 1 and 99.";
  if (!/^\d+$/.test(f.yard) || yard < 1 || yard > 50) e.yard = "Enter a yard line between 1 and 50.";
  if (!e.yard && !e.distance) {
    const yl = yardline100(f);
    if (dist > yl) e.distance = `Yards to go cannot exceed the ${yl} yards to the goal line.`;
    else if (f.goalToGo !== (dist === yl)) e.goalToGo = f.goalToGo ? "Goal to go requires yards to go to equal the distance to the goal line." : `This is goal to go (${yl} yards to the goal line); tick the box or change the distance.`;
  }
  if (Number.isNaN(secs) || secs > 900) e.clock = "Use m:ss between 0:00 and 15:00.";
  if (!/^-?\d+$/.test(f.score) || Math.abs(score) > 100) e.score = "Enter a whole number between −100 and 100.";
  return e;
}

function FieldError({ id, msg }: { id: string; msg?: string }) {
  return msg ? <div className="err" id={id}>{msg}</div> : null;
}

export default function Playcaller() {
  const { data } = useDeployment();
  const snapshots = useMemo(() => (data?.snapshots ?? []).slice().sort((a, b) => b.cutoff_at.localeCompare(a.cutoff_at)), [data]);
  const [form, setForm] = useState<Form>({ offense: "CHI", defense: "GB", season: 2026, mode: "pbp_baseline", snapshotId: "", down: "1", distance: "10", side: "own", yard: "25", quarter: "1", clock: "15:00", score: "0", goalToGo: false });
  const [isExample, setIsExample] = useState(false);
  const [errors, setErrors] = useState<Errors>({});
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [result, setResult] = useState<{ req: RecommendationRequest; res: RecommendationResponse; form: Form } | null>(null);
  const [serviceError, setServiceError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if (q.get("example") === "chi-gb") { setForm((f) => ({ ...f, ...EXAMPLE })); setIsExample(true); return; }
    const keys: (keyof Form)[] = ["offense", "defense", "mode", "down", "distance", "side", "yard", "quarter", "clock", "score"];
    const patch: Partial<Form> = {};
    keys.forEach((k) => { const v = q.get(k); if (v !== null) (patch as Record<string, string>)[k] = v; });
    if (q.get("season")) patch.season = Number(q.get("season"));
    if (q.get("gtg")) patch.goalToGo = q.get("gtg") === "1";
    if (q.get("snapshot")) patch.snapshotId = q.get("snapshot")!;
    setForm((f) => ({ ...f, ...patch }));
  }, []);
  useEffect(() => { if (!form.snapshotId && snapshots[0]) setForm((f) => ({ ...f, snapshotId: snapshots[0].snapshot_id })); }, [snapshots, form.snapshotId]);
  useEffect(() => {
    if (!form.snapshotId) return;
    getCapabilities(form.season, form.defense, form.snapshotId).then(setCaps).catch(() => setCaps(null));
  }, [form.season, form.defense, form.snapshotId]);

  const set = <K extends keyof Form>(k: K, v: Form[K]) => { setForm((f) => ({ ...f, [k]: v })); setIsExample(false); };
  const snap = snapshots.find((s) => s.snapshot_id === form.snapshotId);
  const stale = result !== null && JSON.stringify(result.form) !== JSON.stringify(form);

  const submit = useCallback(async (ev?: React.FormEvent) => {
    ev?.preventDefault();
    const e = validate(form);
    setErrors(e);
    if (Object.keys(e).length) {
      const first = document.getElementById(`f-${Object.keys(e)[0]}`);
      first?.focus();
      return;
    }
    if (!snap || !data) return;
    const req: RecommendationRequest = {
      schema_version: "1.0.0", request_id: crypto.randomUUID(), scenario_kind: "hypothetical", mode: form.mode, season: form.season,
      offense_team: form.offense, defense_team: form.defense, game_id: null, cutoff_at: snap.cutoff_at, snapshot_id: snap.snapshot_id,
      model_bundle_id: data.model_bundle_id,
      situation: { quarter: Number(form.quarter), seconds_remaining_in_quarter: clockSeconds(form.clock), down: Number(form.down),
        yards_to_go: Number(form.distance), yardline_100: yardline100(form), score_differential: Number(form.score), goal_to_go: form.goalToGo },
      candidate_ids: ["designed_rush", "dropback"], utility: { policy: "expected_epa" },
    };
    const q = new URLSearchParams({ offense: form.offense, defense: form.defense, season: String(form.season), mode: form.mode, snapshot: snap.snapshot_id,
      down: form.down, distance: form.distance, side: form.side, yard: form.yard, quarter: form.quarter, clock: form.clock, score: form.score, gtg: form.goalToGo ? "1" : "0" });
    window.history.replaceState(null, "", `?${q}`);
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    const mine = ++seq.current;
    setLoading(true);
    const submittedForm = { ...form };
    try {
      const res = await postRecommendation(req, ctl.signal);
      if (mine === seq.current) { setResult({ req, res, form: submittedForm }); setServiceError(null); }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      if (mine === seq.current) setServiceError(err);
      if (err instanceof ApiRequestError && err.status === 422) {
        setErrors((prev) => ({ ...prev, distance: err.body.fields.some((f) => f.includes("goal")) ? err.body.message : prev.distance }));
      }
    } finally {
      if (mine === seq.current) setLoading(false);
    }
  }, [form, snap, data]);

  const cancel = () => { abort.current?.abort(); seq.current++; setLoading(false); };
  const exportRecord = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), request: result.req, response: result.res }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `analysis_${result.res.request_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const modeCap = caps?.modes?.[form.mode];
  const defName = form.defense;
  const offName = form.offense === "LEAGUE_AVERAGE" ? "League-avg" : form.offense;
  const d = (k: keyof Form) => ({ id: `f-${k}`, "aria-invalid": errors[k] ? true : undefined, "aria-describedby": errors[k] ? `e-${k}` : undefined });

  return (
    <>
      <p className="eyebrow">Playcaller mode</p>
      <h1>Compare broad play families</h1>
      <div className="split" style={{ marginTop: 24 }}>
        <form className="card" onSubmit={submit} noValidate aria-label="Situation">
          {isExample && <p className="note small" role="status"><strong>Saved example.</strong> CHI offense vs GB defense, 2nd &amp; 7 at the GB 38, Q2 9:42. Values are inputs, not findings; results come from the deployed model.</p>}
          <fieldset>
            <legend>Matchup</legend>
            <div className="row">
              <div className="field"><label htmlFor="f-offense">Offense</label>
                <select {...d("offense")} value={form.offense} onChange={(e) => set("offense", e.target.value)}>
                  <option value="LEAGUE_AVERAGE">League average</option>{(data?.teams ?? []).map((t) => <option key={t}>{t}</option>)}
                </select><FieldError id="e-offense" msg={errors.offense} /></div>
              <div className="field"><label htmlFor="f-defense">Defense</label>
                <select {...d("defense")} value={form.defense} onChange={(e) => set("defense", e.target.value)}>
                  {(data?.teams ?? ["GB"]).map((t) => <option key={t}>{t}</option>)}
                </select></div>
            </div>
            <div className="row">
              <div className="field"><label htmlFor="f-season">Season</label>
                <select {...d("season")} value={form.season} onChange={(e) => set("season", Number(e.target.value))}>
                  {(snap?.actual_seasons ?? [2026]).slice().reverse().map((s) => <option key={s}>{s}</option>)}
                </select></div>
              <div className="field"><label htmlFor="f-mode">Mode</label>
                <select {...d("mode")} value={form.mode} onChange={(e) => set("mode", e.target.value as Mode)}>
                  {MODES.map(([m, l]) => <option key={m} value={m}>{l}{caps?.modes?.[m]?.status === "unavailable" ? " (unavailable)" : ""}</option>)}
                </select></div>
            </div>
            {modeCap?.status === "unavailable" && (
              <p className="hint" role="status">Unavailable: {modeCap.reason_codes.map((r) => REASON_TEXT[r] ?? r).join("; ")}. Submitting returns an explicit unavailable result.</p>
            )}
            <div className="field"><label htmlFor="f-snapshotId">Snapshot</label>
              <select {...d("snapshotId")} value={form.snapshotId} onChange={(e) => set("snapshotId", e.target.value)}>
                {snapshots.map((s) => <option key={s.snapshot_id} value={s.snapshot_id}>{fmtDate(s.cutoff_at)} · {s.temporal_evidence_mode === "archived_as_of" ? "archived" : "reconstructed"}</option>)}
              </select></div>
          </fieldset>
          <fieldset>
            <legend>Situation</legend>
            <div className="row">
              <div className="field"><label htmlFor="f-down">Down</label>
                <select {...d("down")} value={form.down} onChange={(e) => set("down", e.target.value)}>{[1, 2, 3, 4].map((x) => <option key={x}>{x}</option>)}</select></div>
              <div className="field"><label htmlFor="f-distance">Yards to go</label>
                <input {...d("distance")} inputMode="numeric" value={form.distance} onChange={(e) => set("distance", e.target.value)} /><FieldError id="e-distance" msg={errors.distance} /></div>
            </div>
            <div className="row">
              <div className="field"><label htmlFor="f-side">Ball on</label>
                <select {...d("side")} value={form.side} onChange={(e) => set("side", e.target.value as "own" | "opp")}>
                  <option value="own">{offName} side (own)</option><option value="opp">{defName} side (opponent)</option>
                </select></div>
              <div className="field"><label htmlFor="f-yard">Yard line</label>
                <input {...d("yard")} inputMode="numeric" value={form.yard} onChange={(e) => set("yard", e.target.value)} />
                <FieldError id="e-yard" msg={errors.yard} />
                {!errors.yard && form.yard && <div className="hint">{Number.isFinite(yardline100(form)) ? `${yardline100(form)} yards to the goal line` : ""}</div>}</div>
            </div>
            <div className="field" style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input type="checkbox" {...d("goalToGo")} checked={form.goalToGo} onChange={(e) => set("goalToGo", e.target.checked)} style={{ width: 24, minHeight: 24 }} />
              <label htmlFor="f-goalToGo" style={{ margin: 0 }}>Goal to go</label>
            </div>
            <FieldError id="e-goalToGo" msg={errors.goalToGo} />
            <div className="row">
              <div className="field"><label htmlFor="f-quarter">Quarter</label>
                <select {...d("quarter")} value={form.quarter} onChange={(e) => set("quarter", e.target.value)}>{[1, 2, 3, 4].map((x) => <option key={x}>{x}</option>)}</select>
                <div className="hint">Overtime is not in the v1 contract.</div></div>
              <div className="field"><label htmlFor="f-clock">Clock (m:ss)</label>
                <input {...d("clock")} value={form.clock} onChange={(e) => set("clock", e.target.value)} /><FieldError id="e-clock" msg={errors.clock} /></div>
            </div>
            <div className="field"><label htmlFor="f-score">Score differential</label>
              <input {...d("score")} inputMode="numeric" value={form.score} onChange={(e) => set("score", e.target.value)} aria-describedby="score-hint" />
              <div className="hint" id="score-hint">Offense score minus defense score (negative = offense trailing).</div>
              <FieldError id="e-score" msg={errors.score} /></div>
          </fieldset>
          <fieldset>
            <legend>Presentation</legend>
            {["Offensive personnel", "Formation", "Hash", "Motion"].map((l) => (
              <div className="field" key={l}><label htmlFor={`dis-${l}`}>{l}</label>
                <select id={`dis-${l}`} disabled aria-describedby="pres-hint"><option>Unknown (not supported)</option></select></div>
            ))}
            <p className="hint" id="pres-hint">The deployed bundle has no validated feature for these inputs. They are never assumed (no default 11 personnel).</p>
          </fieldset>
          <div className="btn-row">
            <button className="btn" type="submit" disabled={loading || !snap}>{loading ? <><span className="spinner" aria-hidden /> Scoring…</> : "Compare"}</button>
            {loading && <button className="btn ghost" type="button" onClick={cancel}>Cancel</button>}
            <button className="btn ghost" type="button" onClick={() => { setForm((f) => ({ ...f, ...EXAMPLE })); setIsExample(true); setErrors({}); }}>Load saved example</button>
          </div>
        </form>

        <section aria-live="polite" aria-busy={loading} aria-label="Result">
          {serviceError !== null && (
            <div style={{ marginBottom: 14 }}>
              <ErrorNote error={serviceError} />
              {result && <p className="small muted">Showing the last completed result from {fmtDate(result.req.cutoff_at)} snapshot below.</p>}
            </div>
          )}
          {!result && !serviceError && (
            <div className="card"><p className="muted">Enter a situation and choose <strong>Compare</strong>. Results state their scope first, then the evidence.</p></div>
          )}
          {result && <Result r={result.res} req={result.req} stale={stale} onExport={exportRecord} />}
        </section>
      </div>
    </>
  );
}

function Result({ r, req, stale, onExport }: { r: RecommendationResponse; req: RecommendationRequest; stale: boolean; onExport: () => void }) {
  const s = req.situation;
  const side = s.yardline_100 === 50 ? "MID 50" : s.yardline_100 < 50 ? `${req.defense_team} ${s.yardline_100}` : `${req.offense_team === "LEAGUE_AVERAGE" ? "OWN" : req.offense_team} ${100 - s.yardline_100}`;
  const ord = ["", "1st", "2nd", "3rd", "4th"][s.down];
  const clock = `${Math.floor(s.seconds_remaining_in_quarter / 60)}:${String(s.seconds_remaining_in_quarter % 60).padStart(2, "0")}`;
  const heading = { ok: "Supported ranking", limited: "Limited ranking", abstained: "No recommendation", unavailable: "Unavailable" }[r.status];
  const first = r.status === "abstained" ? "Insufficient evidence to rank these candidates reliably."
    : r.status === "unavailable" ? "Required data or a compatible model is absent for this request."
    : "The available data supports a broad run-versus-dropback comparison on eligible penalty-free plays.";
  const maxAbs = Math.max(0.25, ...r.candidates.map((c) => Math.abs(c.metrics.expected_epa ?? 0)));
  return (
    <div className={stale ? "stale" : undefined}>
      {stale && <div className="stale-note" role="status">Inputs changed — this result belongs to the previous scenario. Compare again to update.</div>}
      <div className="card reveal" style={{ marginBottom: 16 }}>
        <p className="eyebrow">Scenario</p>
        <div className="down-marker" aria-label={`${ord} and ${s.yards_to_go}, ball on ${side}, quarter ${s.quarter}, ${clock}`}>
          <span>{ord} &amp; {s.goal_to_go ? "Goal" : s.yards_to_go}</span><span>{side}</span><span>Q{s.quarter} {clock}</span>
        </div>
        <p style={{ marginTop: 12 }}><strong>{req.offense_team === "LEAGUE_AVERAGE" ? "League-average offense" : `${req.offense_team} offense`}</strong> vs <strong>{req.defense_team} defense</strong> · score {s.score_differential >= 0 ? "+" : ""}{s.score_differential}</p>
        <dl className="kv small">
          <dt>Snapshot</dt><dd>{r.snapshot_id} · cutoff {fmtDate(r.cutoff_at)}</dd>
          <dt>Mode</dt><dd>{r.mode}</dd><dt>Model bundle</dt><dd>{r.model_bundle_id}</dd><dt>Outcome cohort</dt><dd>{r.outcome_cohort_id}</dd>
        </dl>
      </div>

      <div className={`banner ${r.status}`} role="status">
        <span className="status">{r.status}</span>
        <div><strong>{heading}.</strong> {first}
          {r.status !== "unavailable" && <div className="small muted">Penalties are unknown before the call, so this cohort does not estimate full-call value.</div>}
        </div>
      </div>

      <SectionHead title="Defensive expectation" tag={<Prov kind={r.defense_expectation.status === "predicted" ? "predicted" : "unavailable"} />} />
      <div className="grid cols-3">
        <div className="card"><p className="eyebrow">Expected box count</p>
          {r.defense_expectation.expected_box_count !== null ? <p className="big-num">{r.defense_expectation.expected_box_count.toFixed(2)}</p> : <div className="bar unavailable" style={{ height: 40 }}>UNAVAILABLE</div>}</div>
        <div className="card"><p className="eyebrow">Blitz probability (on a dropback)</p>
          {r.defense_expectation.blitz_probability !== null ? <><p className="big-num">{pct(r.defense_expectation.blitz_probability, 0)}</p><Bar value={r.defense_expectation.blitz_probability} kind="predicted" label="Blitz probability" /></> : <div className="bar unavailable" style={{ height: 40 }}>UNAVAILABLE</div>}</div>
        <div className="card"><p className="eyebrow">Coverage</p>
          {r.defense_expectation.coverage_probabilities ? (
            <table><caption className="vh">Predicted coverage distribution</caption><tbody>
              {Object.entries(r.defense_expectation.coverage_probabilities).map(([k, v]) => (
                <tr key={k}><td className="small">{k.replaceAll("_", " ")}</td><td style={{ width: 90 }}><Bar value={v} kind="predicted" label={k} /></td><td className="num small">{pct(v, 0)}</td></tr>))}
            </tbody></table>
          ) : <><div className="bar unavailable" style={{ height: 40 }}>UNAVAILABLE</div><p className="small muted" style={{ marginTop: 6 }}>{r.defense_expectation.reason_codes.includes("MODEL_UNAVAILABLE")
            ? "Coverage forecasts failed locked-test validation for this bundle and are not shown."
            : "Current coverage labels are not published in-season."}</p></>}
        </div>
      </div>

      {r.candidates.length > 0 && (
        <>
          <SectionHead title="Candidate comparison" tag="model estimates" />
          <div className="table-wrap" role="region" aria-label="Candidate comparison" tabIndex={0}>
            <table>
              <caption>Expected EPA on eligible penalty-free plays · probabilities are unconditional within each candidate</caption>
              <thead><tr>
                <th scope="col">Rank</th><th scope="col">Family</th><th scope="col">Expected EPA</th><th className="num" scope="col">90% EPA interval</th>
                <th className="num" scope="col">Success</th><th className="num" scope="col">Explosive</th><th className="num" scope="col">Sack</th>
                <th className="num" scope="col">Turnover</th><th className="num" scope="col">1st/TD</th><th className="num" scope="col">Support</th>
              </tr></thead>
              <tbody>
                {r.candidates.map((c) => {
                  const m = c.metrics;
                  const winner = r.recommended_candidate_id === c.candidate_id;
                  return (
                    <tr key={c.candidate_id} className={winner ? "winner" : undefined}>
                      <td className="num">{c.rank ?? <span className="muted" title="Not ranked">—</span>}{winner && <span className="vh"> (recommended)</span>}</td>
                      <th scope="row">{c.label}{winner && <div className="small" style={{ color: "var(--turf)" }}>Ranked first</div>}</th>
                      <td style={{ minWidth: 150 }}>
                        <div className="num" style={{ textAlign: "left" }}>{epa(m.expected_epa)}{c.ensemble_epa_standard_deviation !== null && <span className="small muted"> ±{c.ensemble_epa_standard_deviation.toFixed(3)} seeds</span>}</div>
                        <Bar value={Math.abs(m.expected_epa ?? 0)} max={maxAbs} kind="predicted" label={`${c.label} expected EPA magnitude`} />
                      </td>
                      <td className="num">{m.epa_prediction_interval ? `${epa(m.epa_prediction_interval.lower)} to ${epa(m.epa_prediction_interval.upper)}` : <span className="small muted">{REASON_TEXT[m.missing_reasons.epa_prediction_interval] ?? "—"}</span>}</td>
                      <td className="num">{pct(m.success_probability)}</td>
                      <td className="num">{pct(m.explosive_probability)}</td>
                      <td className="num">{c.candidate_id === "designed_rush" && m.sack_probability === 0 ? <span title="Zero by definition for a verified designed rush">0 (def.)</span> : pct(m.sack_probability)}</td>
                      <td className="num">{pct(m.turnover_probability)}</td>
                      <td className="num">{pct(m.first_down_or_td_probability)}</td>
                      <td className="num">{c.support.observed_play_count.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>The EPA interval covers the outcome of a single future play (split-conformal on a held-out calibration block; measured coverage is on the Evaluation page) — not a confidence interval for mean EPA or a causal benefit.</p>
        </>
      )}

      <SectionHead title="Why this result" tag={`engine: ${r.explanation.engine}`} />
      <div className="card"><p style={{ maxWidth: "none" }}>{r.explanation.text}</p>
        {r.candidates.map((c) => <p key={c.candidate_id} className="small muted" style={{ maxWidth: "none" }}><strong>{c.label} support:</strong> {c.support.scope}</p>)}
      </div>

      <SectionHead title="Reliability" tag="separate meanings, separate denominators" />
      <div className="grid cols-2">
        <div className="card">
          <h3>Data coverage</h3>
          <dl className="kv small">
            <dt>Defense plays this season</dt><dd>{r.data_evidence.eligible_defense_play_count}</dd>
            <dt>Effective sample size</dt><dd>{r.data_evidence.effective_sample_size ?? "—"}</dd>
            <dt>Latest eligible game</dt><dd>{fmtDate(r.data_evidence.latest_eligible_event_at)}</dd>
            <dt>Last ingested</dt><dd>{fmtDate(r.data_evidence.last_ingested_at)}</dd>
            <dt>Availability basis</dt><dd>{r.data_evidence.availability_basis}</dd>
            <dt>Unavailable inputs</dt><dd>{r.data_evidence.unavailable_inputs.join(", ")}</dd>
          </dl>
        </div>
        <div className="card">
          <h3>Model agreement &amp; policy</h3>
          <dl className="kv small">
            <dt>Ensemble members</dt><dd>{r.model_evidence.ensemble_members}</dd>
            <dt>Ranking passed policy</dt><dd>{r.model_evidence.ranking_passed_policy ? "yes" : "no"}</dd>
            <dt>Ranking policy</dt><dd>{r.model_evidence.ranking_policy_id}</dd>
            <dt>Evaluation report</dt><dd>{r.model_evidence.evaluation_report_id}</dd>
            <dt>Calibration</dt><dd>{r.model_evidence.calibration_artifact_id}</dd>
          </dl>
          <p className="small muted" style={{ marginTop: 8 }}>Seed agreement is one screen among several, not a probability that the top family is correct.</p>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Reason codes</h3>
        {r.reason_codes.length === 0 ? <p className="muted">None.</p> : (
          <ul>{r.reason_codes.map((c) => <li key={c}><code className="mono small">{c}</code> — {REASON_TEXT[c] ?? c}</li>)}</ul>
        )}
      </div>
      <div className="btn-row" style={{ marginTop: 16 }}>
        <button className="btn ghost" type="button" onClick={onExport}>Export analysis record (JSON)</button>
      </div>
    </div>
  );
}
