"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getDeployment, type DeploymentInfo, fmtDate } from "@/lib/api";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/analyst", label: "Analyst" },
  { href: "/playcaller", label: "Playcaller" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/methodology", label: "Data & Methodology" },
];

interface DeploymentState { data: DeploymentInfo | null; error: unknown }
const DeploymentCtx = createContext<DeploymentState>({ data: null, error: null });
export const useDeployment = () => useContext(DeploymentCtx);

export function Chrome({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [state, setState] = useState<DeploymentState>({ data: null, error: null });
  useEffect(() => {
    getDeployment().then((data) => setState({ data, error: null })).catch((error) => setState({ data: null, error }));
  }, []);
  const current = state.data?.snapshots.slice().sort((a, b) => b.cutoff_at.localeCompare(a.cutoff_at))[0];
  return (
    <DeploymentCtx.Provider value={state}>
      <a className="skip" href="#main">Skip to content</a>
      <header className="masthead">
        <div className="masthead-inner">
          <Link href="/" className="brand" aria-label="AI Offensive Coordinator home">
            <span className="brand-mark">AI&nbsp;OC</span>
            <span className="brand-sub">Defensive scouting · broad play families</span>
          </Link>
          <nav className="primary" aria-label="Primary">
            <ul>
              {NAV.map((n) => (
                <li key={n.href}>
                  <Link href={n.href} aria-current={path === n.href ? "page" : undefined}>{n.label}</Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>
        <div className="ruler" aria-hidden="true" />
        <div className="status-strip" role="status" aria-live="polite">
          <div className="status-strip-inner">
            {state.error ? (
              <span className="warn">API unreachable — start it with <code>uv run coordinator serve</code></span>
            ) : !current ? (
              <span>Loading deployment…</span>
            ) : (
              <>
                <span>Latest snapshot <b>{fmtDate(current.cutoff_at)}</b></span>
                <span>Seasons <b>{current.actual_seasons[0]}–{current.actual_seasons.at(-1)}</b></span>
                <span>Evidence <b>{current.temporal_evidence_mode.replaceAll("_", " ")}</b></span>
                <span>Bundle <b>{state.data!.model_bundle_id}</b></span>
              </>
            )}
          </div>
        </div>
      </header>
      <main id="main" tabIndex={-1}>{children}</main>
      <footer className="colophon">
        Independent portfolio project; not affiliated with the NFL, any club or any data provider. Data: nflverse play-by-play;
        FTN Data via nflverse (CC BY-SA 4.0); NFL NextGenStats via nflverse. Outcomes are model estimates from observational data.
      </footer>
    </DeploymentCtx.Provider>
  );
}
