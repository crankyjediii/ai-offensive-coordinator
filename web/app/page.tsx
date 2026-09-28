"use client";

import Link from "next/link";
import { useDeployment } from "@/components/Chrome";
import { ErrorNote, Legend, SectionHead } from "@/components/ui";
import { fmtDate } from "@/lib/api";

export default function Overview() {
  const { data, error } = useDeployment();
  return (
    <>
      <section className="hero">
        <div className="reveal">
          <p className="eyebrow">Scouting report · as-of snapshots · honest abstention</p>
          <h1>Know what the defense has shown. Know what the data can’t say.</h1>
        </div>
        <div className="reveal">
          <p className="lede">
            Dated defensive profiles and a model comparison of <strong>designed rush</strong> versus <strong>dropback</strong> for a
            situation you enter. Every number names its snapshot, cutoff and model bundle. When support is thin, the system declines
            to rank.
          </p>
          <div className="btn-row">
            <Link className="btn" href="/playcaller?example=chi-gb">Open saved example</Link>
            <Link className="btn ghost" href="/analyst">Scout a defense</Link>
          </div>
        </div>
      </section>

      <SectionHead title="What each mode does" tag="product scope" />
      <div className="grid cols-3">
        <article className="card reveal">
          <h3>Analyst</h3>
          <p>Conditional tendencies with numerator, denominator, missing-label count and game-clustered intervals. Observed history and
            smoothed estimates are shown separately.</p>
        </article>
        <article className="card reveal">
          <h3>Playcaller</h3>
          <p>Expected EPA on eligible penalty-free plays for each supported broad family, with event probabilities, a calibrated
            prediction interval, support counts and reason codes. It can return no recommendation.</p>
        </article>
        <article className="card reveal">
          <h3>Not in scope</h3>
          <p>Named concepts (mesh, flood, inside zone), live in-game calls, and claims about points or wins from plays that were
            never run. Those require labels this project does not have.</p>
        </article>
      </div>

      <SectionHead title="Deployed evidence" tag="immutable identifiers" />
      <ErrorNote error={error} />
      {data && (
        <div className="table-wrap" role="region" aria-label="Deployed snapshots" tabIndex={0}>
          <table>
            <caption>Snapshots served by deployment {data.record.deployment_id}</caption>
            <thead>
              <tr><th>Snapshot</th><th>Cutoff</th><th>Evidence basis</th><th>Seasons</th><th className="num">Plays</th><th>Audit</th></tr>
            </thead>
            <tbody>
              {data.snapshots.map((s) => (
                <tr key={s.snapshot_id}>
                  <td className="mono small">{s.snapshot_id}</td>
                  <td>{fmtDate(s.cutoff_at)}</td>
                  <td>{s.temporal_evidence_mode === "archived_as_of" ? "Archived as-of (forward)" : "Reconstructed, conservative delay"}</td>
                  <td>{s.actual_seasons.join(", ")}</td>
                  <td className="num">{s.row_counts.play.toLocaleString()}</td>
                  <td>{s.audit_status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div style={{ marginTop: 18 }}><Legend /></div>
    </>
  );
}
