from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("COORDINATOR_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("COORDINATOR_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CONFIGS_DIR = REPO_ROOT / "configs"
CONTRACTS_DIR = REPO_ROOT / "contracts"

RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
BUNDLES_DIR = ARTIFACTS_DIR / "bundles"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
DEPLOYMENTS_DIR = ARTIFACTS_DIR / "deployments"
PREDICTION_LOG_DIR = ARTIFACTS_DIR / "prediction_log"

SCHEMA_VERSION = "canonical_v1"
TAXONOMY_VERSION = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"
RESPONSE_CONTRACT_VERSION = "1.0.0"
PRIMARY_OUTCOME_COHORT = "reg_scrimmage_penalty_free_v1"

TEAMS = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)
LEAGUE_AVERAGE = "LEAGUE_AVERAGE"
