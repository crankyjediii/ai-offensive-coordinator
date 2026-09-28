# artifacts/

Generated, git-ignored outputs. Only this README is tracked.

- `bundles/<model_bundle_id>/` — immutable model bundles (`bundle.json` + digest-verified weights).
- `experiments/<experiment_id>/` — experiment ledgers: `report.json` (all attempted models, splits, decision) and `metrics.csv`.
- `deployments/` — immutable deployment records and the atomic `active.json` pointer (holds `previous` for rollback).
- `prediction_log/` — forward prediction log (JSONL, no client IPs).
- `logs/` — training job output.
