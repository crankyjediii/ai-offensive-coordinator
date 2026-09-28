import type { ReactNode } from "react";

export type Provenance = "observed" | "predicted" | "unavailable" | "user_assumed";

export function Prov({ kind }: { kind: Provenance }) {
  const label = { observed: "Observed", predicted: "Predicted", unavailable: "Unavailable", user_assumed: "User assumed" }[kind];
  return <span className={`prov ${kind === "user_assumed" ? "assumed" : kind}`}>{label}</span>;
}

/** A bar whose texture encodes provenance. Unavailable values never render as a zero-length bar. */
export function Bar({ value, max = 1, kind, interval, label }: {
  value: number | null; max?: number; kind: Provenance; interval?: [number, number] | null; label: string;
}) {
  if (value === null || kind === "unavailable") {
    return <div className="bar unavailable" role="img" aria-label={`${label}: unavailable`}>UNAVAILABLE</div>;
  }
  const w = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className={`bar ${kind}`} role="img" aria-label={`${label}: ${value.toFixed(3)}${interval ? `, interval ${interval[0].toFixed(3)} to ${interval[1].toFixed(3)}` : ""}`}>
      <i style={{ width: `${w}%` }} />
      {interval && (
        <span className="ci" style={{ left: `${(Math.max(0, interval[0]) / max) * 100}%`, width: `${((Math.min(max, interval[1]) - Math.max(0, interval[0])) / max) * 100}%` }} />
      )}
    </div>
  );
}

export function SectionHead({ title, tag, id }: { title: string; tag?: ReactNode; id?: string }) {
  return (
    <div className="section-head">
      <h2 id={id}>{title}</h2>
      {tag && <span className="tag">{tag}</span>}
    </div>
  );
}

export function Legend() {
  return (
    <p className="chips" aria-label="Legend">
      <Prov kind="observed" /> <Prov kind="predicted" /> <Prov kind="unavailable" />
      <span className="small muted">Texture, not color, carries provenance. Unavailable is never drawn as zero.</span>
    </p>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as { status?: number; body?: { code: string; message: string; fields: string[] }; message?: string };
  return (
    <div className="note" role="alert">
      <strong>{e.body?.code ?? "SERVICE_ERROR"}</strong> — {e.body?.message ?? e.message ?? "The service could not be reached."}
      {e.body?.fields?.length ? <div className="small mono">Fields: {e.body.fields.join(", ")}</div> : null}
    </div>
  );
}
