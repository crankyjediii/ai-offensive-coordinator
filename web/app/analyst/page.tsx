"use client";

import { useEffect, useMemo, useState } from "react";
import { useDeployment } from "@/components/Chrome";
import { Bar, ErrorNote, Legend, Prov, SectionHead } from "@/components/ui";
import { fmtDate, getProfile, pct, type Profile, type SeasonProfile, type Tendency } from "@/lib/api";

const BANDS = [["", "Any distance"], ["short", "Short (1–3)"], ["medium", "Medium (4–6)"], ["long", "Long (7–10)"], ["very_long", "Very long (11+)"]];
const ZONES = [["", "Any field position"], ["goal_line", "Goal line (1–5)"], ["red_zone", "Red zone (6–20)"], ["open_field", "Open field (21–80)"], ["backed_up", "Backed up (81–99)"]];

function fmtVal(t: Tendency) {
  if (t.value === null) return t.denominator === 0 ? "No labels" : `Suppressed (n<${10})`;
  return t.kind === "rate" ? pct(t.value) : t.value.toFixed(3);
}

function TendencyTable({ sp, caption }: { sp: SeasonProfile; caption: string }) {
  return (
    <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr><th scope="col">Tendency</th><th scope="col">Chart</th><th className="num" scope="col">Value</th><th className="num" scope="col">95% interval</th>
            <th className="num" scope="col">Labeled n</th><th className="num" scope="col">Missing</th><th scope="col">Source</th></tr>
        </thead>
        <tbody>
          {sp.tendencies.map((t) => {
            const max = t.kind === "rate" ? 1 : t.label.includes("box") ? 11 : 0.6;
            const plottable = t.kind === "rate" || t.label.includes("box");
            return (
              <tr key={t.label}>
                <th scope="row" style={{ fontWeight: 600 }}>{t.label}<div className="small muted">{t.denominator_description}</div></th>
                <td style={{ width: 170 }}>
                  {plottable ? <Bar value={t.value} max={max} kind={t.value === null ? "unavailable" : "observed"} interval={t.interval} label={t.label} /> : <Prov kind="observed" />}
                </td>
                <td className="num">{fmtVal(t)}</td>
                <td className="num">{t.interval ? (t.kind === "rate" ? `${pct(t.interval[0])} – ${pct(t.interval[1])}` : `${t.interval[0].toFixed(3)} – ${t.interval[1].toFixed(3)}`) : <span className="muted small">{t.interval_method?.startsWith("not_estimated") ? "<5 games" : "—"}</span>}</td>
                <td className="num">{t.kind === "rate" && t.numerator !== null ? `${t.numerator}/${t.denominator}` : t.denominator}</td>
                <td className="num">{t.missing_label_count}</td>
                <td className="small">{t.source}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CoveragePanel({ sp, title }: { sp: SeasonProfile; title: string }) {
  const cd = sp.coverage_distribution;
  if (!cd.denominator) {
    const u = sp.unavailable.find((x) => x.field === "coverage");
    return (
      <div className="card">
        <h3>{title}</h3>
        <div className="bar unavailable" role="img" aria-label="Coverage unavailable" style={{ height: 36 }}>COVERAGE LABELS UNAVAILABLE</div>
        <p className="small muted" style={{ marginTop: 8 }}>{u?.detail ?? "No coverage labels published by this cutoff."}</p>
      </div>
    );
  }
  return (
    <div className="card">
      <h3>{title} <Prov kind="observed" /></h3>
      <div className="table-wrap" role="region" aria-label={title} tabIndex={0}>
        <table>
          <caption>{cd.denominator} labeled dropbacks · {cd.missing_label_count} missing labels</caption>
          <thead><tr><th>Coverage</th><th>Share</th><th className="num">Count</th></tr></thead>
          <tbody>
            {cd.rows.map((r) => (
              <tr key={r.coverage}>
                <td>{r.coverage.replaceAll("_", " ")}</td>
                <td style={{ width: 160 }}><Bar value={r.count / cd.denominator} kind="observed" label={r.coverage} /></td>
                <td className="num">{r.count} ({pct(r.count / cd.denominator)})</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Analyst() {
  const { data } = useDeployment();
  const snapshots = useMemo(() => (data?.snapshots ?? []).slice().sort((a, b) => b.cutoff_at.localeCompare(a.cutoff_at)), [data]);
  const [defense, setDefense] = useState("GB");
  const [season, setSeason] = useState(2026);
  const [snapshotId, setSnapshotId] = useState("");
  const [down, setDown] = useState("");
  const [band, setBand] = useState("");
  const [zone, setZone] = useState("");
  const [half, setHalf] = useState("");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { if (!snapshotId && snapshots[0]) setSnapshotId(snapshots[0].snapshot_id); }, [snapshots, snapshotId]);
  const snap = snapshots.find((s) => s.snapshot_id === snapshotId);

  useEffect(() => {
    if (!snapshotId) return;
    let live = true;
    setLoading(true);
    getProfile(defense, { season, snapshot_id: snapshotId, down, distance_band: band, field_zone: zone, half })
      .then((p) => { if (live) { setProfile(p); setError(null); } })
      .catch((e) => live && setError(e))
      .finally(() => live && setLoading(false));
    return () => { live = false; };
  }, [defense, season, snapshotId, down, band, zone, half]);

  const sm = profile?.smoothed;
  return (
    <>
      <p className="eyebrow">Analyst mode</p>
      <h1>Defensive scouting report</h1>
      <div className="split" style={{ marginTop: 24 }}>
        <form className="card" onSubmit={(e) => e.preventDefault()} aria-label="Report filters">
          <fieldset>
            <legend>Selection</legend>
            <div className="field"><label htmlFor="def">Defense</label>
              <select id="def" value={defense} onChange={(e) => setDefense(e.target.value)}>
                {(data?.teams ?? ["GB"]).map((t) => <option key={t}>{t}</option>)}
              </select></div>
            <div className="field"><label htmlFor="season">Season</label>
              <select id="season" value={season} onChange={(e) => setSeason(Number(e.target.value))}>
                {(snap?.actual_seasons ?? [2026]).slice().reverse().map((s) => <option key={s}>{s}</option>)}
              </select></div>
            <div className="field"><label htmlFor="snap">Snapshot</label>
              <select id="snap" value={snapshotId} onChange={(e) => setSnapshotId(e.target.value)}>
                {snapshots.map((s) => <option key={s.snapshot_id} value={s.snapshot_id}>{fmtDate(s.cutoff_at)} · {s.temporal_evidence_mode === "archived_as_of" ? "archived" : "reconstructed"}</option>)}
              </select>
              <div className="hint">Cutoff = snapshot cutoff. Later games are excluded.</div></div>
          </fieldset>
          <fieldset>
            <legend>Situation filters</legend>
            <div className="row">
              <div className="field"><label htmlFor="down">Down</label>
                <select id="down" value={down} onChange={(e) => setDown(e.target.value)}>
                  <option value="">Any</option>{[1, 2, 3, 4].map((d) => <option key={d}>{d}</option>)}</select></div>
              <div className="field"><label htmlFor="half">Half</label>
                <select id="half" value={half} onChange={(e) => setHalf(e.target.value)}>
                  <option value="">Any</option><option value="1">First</option><option value="2">Second</option></select></div>
            </div>
            <div className="field"><label htmlFor="band">Distance</label>
              <select id="band" value={band} onChange={(e) => setBand(e.target.value)}>{BANDS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
            <div className="field"><label htmlFor="zone">Field position</label>
              <select id="zone" value={zone} onChange={(e) => setZone(e.target.value)}>{ZONES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
            <p className="hint">Personnel, motion, formation, coverage and box filters stay disabled until an audited capability supports them.</p>
            <div className="field"><label htmlFor="pers">Offensive personnel</label>
              <select id="pers" disabled aria-describedby="pers-hint"><option>Unavailable in this mode</option></select>
              <div id="pers-hint" className="hint">No validated personnel feature in the deployed bundle.</div></div>
          </fieldset>
        </form>

        <div aria-busy={loading}>
          <ErrorNote error={error} />
          {profile && (
            <>
              <div className="card reveal">
                <dl className="kv">
                  <dt>Defense · season</dt><dd>{profile.defense} · {profile.season}</dd>
                  <dt>Cutoff</dt><dd>{fmtDate(profile.cutoff_at)} <span className="muted">({profile.cutoff_at})</span></dd>
                  <dt>Evidence basis</dt><dd>{profile.temporal_evidence_mode}</dd>
                  <dt>Current season</dt><dd>{profile.current_season.eligible_plays} eligible plays · {profile.current_season.distinct_games} games</dd>
                  <dt>Snapshot</dt><dd>{profile.snapshot_id}</dd>
                </dl>
              </div>
              <SectionHead title={`${profile.season} season to date`} tag={<Prov kind="observed" />} />
              <Legend />
              <TendencyTable sp={profile.current_season} caption={`${profile.defense} ${profile.season} observed tendencies`} />
              <SectionHead title="Coverage" tag="participation labels" />
              <div className="grid cols-2">
                <CoveragePanel sp={profile.current_season} title={`${profile.season} coverage`} />
                <CoveragePanel sp={profile.previous_season} title={`${profile.season - 1} coverage (historical)`} />
              </div>
              {sm && (
                <>
                  <SectionHead title="Smoothed estimate" tag={<Prov kind="predicted" />} />
                  <p className="note">{sm.statement} Component weights are model settings, not measured causal influences of coaches or rosters.</p>
                  <div className="grid cols-2" style={{ marginTop: 14 }}>
                    {(["faced_dropback", "five_plus", "blitz", "box"] as const).map((h) => {
                      const s = sm.heads[h];
                      if (!s) return null;
                      const names: Record<string, string> = { faced_dropback: "Opponent dropback share", five_plus: "Five+ rushers (dropbacks)", blitz: "Blitz (dropbacks)", box: "Box count distribution" };
                      return (
                        <div className="card" key={h}>
                          <h3>{names[h]}</h3>
                          <div className="table-wrap" role="region" aria-label={names[h]} tabIndex={0}>
                            <table>
                              <caption>N={s.support.raw_current_count_N.toFixed(0)} current · W={s.support.weighted_mass_W.toFixed(1)} · n_eff={s.support.kish_n_eff.toFixed(1)}</caption>
                              <thead><tr><th>Class</th><th>Estimate</th><th className="num">League</th><th className="num">Team hist.</th><th className="num">Final</th></tr></thead>
                              <tbody>
                                {s.classes.map((c, i) => (h === "box" || i === 1) && (
                                  <tr key={c}>
                                    <td>{c}</td>
                                    <td style={{ width: 130 }}><Bar value={s.estimate[i]} kind="predicted" label={`${names[h]} ${c}`} /></td>
                                    <td className="num">{pct(s.league_component[i])}</td>
                                    <td className="num">{pct(s.team_history_component[i])}</td>
                                    <td className="num"><strong>{pct(s.estimate[i])}</strong></td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <p className="small muted" style={{ marginTop: 10 }}>Prior method: κ={String(sm.method.kappa)}, team weight={String(sm.method.a_team)}, history decay τ={String(sm.method.tau_hist_days)} days · playcaller attribution: {sm.playcaller_attribution_status}</p>
                </>
              )}
              <SectionHead title={`${profile.season - 1} season (historical)`} tag={<Prov kind="observed" />} />
              <TendencyTable sp={profile.previous_season} caption={`${profile.defense} ${profile.season - 1} observed tendencies`} />
            </>
          )}
        </div>
      </div>
    </>
  );
}
