"use client";

import { useDeployment } from "@/components/Chrome";
import { SectionHead } from "@/components/ui";
import { fmtDate } from "@/lib/api";

export default function Methodology() {
  const { data } = useDeployment();
  return (
    <>
      <p className="eyebrow">Data &amp; methodology</p>
      <h1>How every number is produced</h1>

      <SectionHead title="Four times, kept separate" tag="as-of discipline" />
      <div className="grid cols-2">
        <div className="card"><dl className="kv">
          <dt>event_at</dt><dd>When the game finished (estimated kickoff + 4h)</dd>
          <dt>source_available_at</dt><dd>Release-asset upload time from the provider</dd>
          <dt>ingested_at</dt><dd>When this project downloaded the bytes</dd>
          <dt>cutoff_at</dt><dd>The prediction&apos;s information boundary</dd>
        </dl></div>
        <div className="card">
          <p>Forward snapshots use <code>known_at = max(published, ingested)</code>. Historical tests use a documented conservative delay:
            play-by-play 24h after the game, FTN charting 48h, participation labels on March 1 after the season. Because only current
            re-uploads could be retrieved, those tests are labeled <strong>reconstructed</strong>, never archived replays.</p>
        </div>
      </div>

      <SectionHead title="Sources and attribution" tag="licensed data stays out of git" />
      {data?.snapshots.map((s) => (
        <div className="card" key={s.snapshot_id} style={{ marginBottom: 12 }}>
          <h3>{s.snapshot_id}</h3>
          <p className="small">Cutoff {fmtDate(s.cutoff_at)} · {s.temporal_evidence_mode} · audit {s.audit_status}</p>
          <ul className="small">{s.attributions.map((a) => <li key={a}>{a}</li>)}</ul>
        </div>
      ))}

      <SectionHead title="Label decisions" tag="taxonomy 1.0.0" />
      <div className="grid cols-2">
        <div className="card"><h3>Broad actions</h3>
          <p>Dropback includes sacks and scrambles; designed rush requires a rushing attempt with explicit absence of dropback and scramble
            flags. Missing or contradictory flags stay <em>ambiguous</em> rather than defaulting to false. Run gap never implies a blocking
            scheme; a receiver&apos;s route never implies a full concept.</p></div>
        <div className="card"><h3>Coverage</h3>
          <p>Modeled on FTN-era participation labels (2023+). The 2022 NGS-era vocabulary and class mix differ, so those rows are excluded.
            Cover 9, combo, blown and prevent are pooled into <code>OTHER_OBSERVED</code> under an explicit rule; unknown is masked, never
            pooled.</p></div>
        <div className="card"><h3>Outcome cohort</h3>
          <p><code>reg_scrimmage_penalty_free_v1</code>: regular-season eligible scrimmage plays with no flagged penalty. Penalties are
            unknown when a play is chosen, so expected EPA on this cohort is a restricted comparison, not full-call value.</p></div>
        <div className="card"><h3>FTN zero encoding</h3>
          <p>The charting source encodes &ldquo;not charted&rdquo; as 0. Box and rusher counts of 0 become null with a missing reason; an
            impossible rusher count (&gt;11) is quarantined. Blitz (provider blitzer ≥1) and five-plus rushers disagree on about 8% of
            charted dropbacks and are kept as separate labels.</p></div>
      </div>

      <SectionHead title="Models" tag="baselines first" />
      <p>Model A forecasts box count, pass-rusher count, blitz, coverage, man/zone and pressure for a hypothetical snap from the
        defense&apos;s last 128 eligible snaps plus smoothed priors. Model B estimates expected EPA and unconditional event probabilities for
        each eligible broad family. Baselines (league and team-smoothed rates, regularized linear models, gradient boosting) are trained on
        identical rows; a neural model is served only if it clears a prespecified gate on two development windows.</p>
      <p>Priors blend league and team history with current-season observations: <code>p = (C + κ·p0) / (W + κ)</code>. No verified
        defensive playcaller registry is loaded, so the coordinator component is unavailable and its weight is redistributed.</p>

      <SectionHead title="What this cannot tell you" tag="limits" />
      <ul>
        <li>Which named concept to call — no validated concept labels exist.</li>
        <li>The outcome of a play that was not run — only the chosen action has a factual outcome.</li>
        <li>Current-season coverage as observed — it is only forecast from prior seasons.</li>
        <li>Coach or player skill — team, coach and roster effects are entangled.</li>
      </ul>
    </>
  );
}
