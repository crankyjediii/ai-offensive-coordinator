// Typed client for the FastAPI service. The UI never recomputes EPA, utility or confidence;
// it displays exactly what the API returns.

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8010";

export type Status = "ok" | "limited" | "abstained" | "unavailable";
export type Mode = "pbp_baseline" | "charted_history" | "enriched_current";

export interface ApiError {
  request_id: string;
  code: string;
  message: string;
  fields: string[];
}

export class ApiRequestError extends Error {
  constructor(public status: number, public body: ApiError) {
    super(body.message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  const body = await res.json().catch(() => ({ code: "BAD_RESPONSE", message: "Unreadable response", fields: [], request_id: "" }));
  if (!res.ok) throw new ApiRequestError(res.status, body as ApiError);
  return body as T;
}

export interface SnapshotInfo {
  snapshot_id: string;
  cutoff_at: string;
  temporal_evidence_mode: "archived_as_of" | "reconstructed_conservative_delay";
  actual_seasons: number[];
  audit_status: string;
  row_counts: Record<string, number>;
  attributions: string[];
}
export interface DeploymentInfo {
  record: { deployment_id: string; model_bundle_id: string; evaluation_report_id: string; outcome_cohort_id: string; created_at: string };
  snapshots: SnapshotInfo[];
  model_bundle_id: string;
  teams: string[];
}
export const getDeployment = () => call<DeploymentInfo>("/v1/deployment");

export interface Tendency {
  label: string;
  kind: "rate" | "mean";
  source: string;
  provenance: "observed";
  denominator: number;
  denominator_description: string;
  missing_label_count: number;
  numerator: number | null;
  value: number | null;
  interval: [number, number] | null;
  interval_method: string | null;
  suppressed: boolean;
  distinct_games?: number;
}
export interface SeasonProfile {
  season: number;
  eligible_plays: number;
  distinct_games: number;
  date_range: [string | null, string | null];
  tendencies: Tendency[];
  coverage_distribution: { denominator: number; rows: { coverage: string; count: number }[]; missing_label_count: number };
  unavailable: { field: string; reason: string; detail?: string }[];
  min_denominator_for_rate: number;
}
export interface SmoothedHead {
  classes: string[];
  estimate: number[];
  league_component: number[];
  team_history_component: number[];
  prior: number[];
  current_season_counts: number[];
  support: { raw_current_count_N: number; weighted_mass_W: number; kish_n_eff: number; weighted_history_team: number };
}
export interface Profile {
  snapshot_id: string;
  cutoff_at: string;
  temporal_evidence_mode: string;
  defense: string;
  season: number;
  current_season: SeasonProfile;
  previous_season: SeasonProfile;
  smoothed: { method: Record<string, number | string>; statement: string; playcaller_attribution_status: string; heads: Record<string, SmoothedHead> };
}
export const getProfile = (team: string, q: Record<string, string | number | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(q).forEach(([k, v]) => v !== undefined && v !== "" && p.set(k, String(v)));
  return call<Profile>(`/v1/defenses/${team}/profile?${p}`);
};

export interface Capabilities {
  snapshot_id: string;
  cutoff_at: string;
  temporal_evidence_mode: string;
  defense_current_season_eligible_plays: number;
  defense_current_season_games: number;
  latest_eligible_event_at: string | null;
  modes: Record<Mode, { status: "available" | "unavailable"; reason_codes: string[]; candidates: string[];
    defense_heads: Record<string, { status: string; reason: string | null }> }>;
  unsupported_optional_inputs: string[];
}
export const getCapabilities = (season: number, defense: string, snapshot_id: string) =>
  call<Capabilities>(`/v1/capabilities?season=${season}&defense=${defense}&snapshot_id=${snapshot_id}`);

export interface Metrics {
  expected_epa: number | null;
  expected_yards: number | null;
  success_probability: number | null;
  explosive_probability: number | null;
  sack_probability: number | null;
  turnover_probability: number | null;
  first_down_or_td_probability: number | null;
  epa_prediction_interval: { quantity: string; lower: number; upper: number; nominal_coverage: number; method: string; calibration_artifact_id: string } | null;
  missing_reasons: Record<string, string>;
}
export interface Candidate {
  candidate_id: string;
  label: string;
  rank: number | null;
  utility_epa_units: number | null;
  metrics: Metrics;
  support: { observed_play_count: number; effective_sample_size: number | null; scope: string };
  ensemble_epa_standard_deviation: number | null;
  reason_codes: string[];
}
export interface RecommendationResponse {
  schema_version: string;
  request_id: string;
  is_synthetic_example: boolean;
  example_notice: string | null;
  status: Status;
  reason_codes: string[];
  mode: Mode;
  cutoff_at: string;
  snapshot_id: string;
  model_bundle_id: string;
  outcome_cohort_id: string;
  recommended_candidate_id: string | null;
  candidates: Candidate[];
  defense_expectation: { status: "predicted" | "unavailable"; coverage_probabilities: Record<string, number> | null;
    blitz_probability: number | null; expected_box_count: number | null; reason_codes: string[] };
  data_evidence: { eligible_defense_play_count: number; effective_sample_size: number | null; latest_eligible_event_at: string | null;
    last_ingested_at: string; availability_basis: string; source_ids: string[]; unavailable_inputs: string[]; reason_codes: string[] };
  model_evidence: { evaluation_report_id: string | null; calibration_artifact_id: string | null; ranking_policy_id: string;
    ensemble_members: number; ranking_passed_policy: boolean; reason_codes: string[] };
  explanation: { engine: string; text: string };
}
export interface RecommendationRequest {
  schema_version: "1.0.0";
  request_id: string;
  scenario_kind: "hypothetical";
  mode: Mode;
  season: number;
  offense_team: string;
  defense_team: string;
  game_id: null;
  cutoff_at: string;
  snapshot_id: string;
  model_bundle_id: string;
  situation: { quarter: number; seconds_remaining_in_quarter: number; down: number; yards_to_go: number; yardline_100: number;
    score_differential: number; goal_to_go: boolean };
  candidate_ids: string[];
  utility: { policy: "expected_epa" };
}
export const postRecommendation = (req: RecommendationRequest, signal?: AbortSignal) =>
  call<RecommendationResponse>("/v1/recommendations", { method: "POST", body: JSON.stringify(req), signal });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const getModelCard = (bundleId: string) => call<any>(`/v1/models/${bundleId}/card`);

export const REASON_TEXT: Record<string, string> = {
  BROAD_CANDIDATES_ONLY: "Broad designed-rush vs dropback comparison only",
  COVERAGE_LABELS_UNAVAILABLE: "Coverage labels unavailable for this snapshot",
  INTERVAL_NOT_CALIBRATED: "No calibrated EPA interval",
  METRIC_NOT_ESTIMATED: "Metric not estimated by this bundle",
  LOW_SUPPORT: "Too few comparable historical snaps",
  LOW_OVERLAP: "Candidate rarely chosen in this situation",
  OUT_OF_DISTRIBUTION: "Situation outside the supported range",
  UNSTABLE_RANKING: "Margin smaller than policy minimum or ensemble spread",
  MISSING_REQUIRED_CAPABILITY: "Required data capability not available",
  MODEL_UNAVAILABLE: "No compatible model for this mode",
  SNAPSHOT_UNAVAILABLE: "Snapshot unavailable",
  STALE_SOURCE: "Latest observation older than freshness policy",
  UNKNOWN_PUBLICATION_TIME: "Publication time unknown",
  PRIOR_DOMINATED: "No current-season observations; prior history dominates",
  NO_ELIGIBLE_CANDIDATE: "No eligible candidate",
};

export const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "—" : `${(x * 100).toFixed(d)}%`);
export const epa = (x: number | null | undefined) => (x === null || x === undefined ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(3)}`);
export const fmtDate = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";
