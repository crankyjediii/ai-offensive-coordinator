# Source register

Primary references linked from the numbered documents, plus the data assets the implementation actually retrieved. Provider pages were checked on 2026-09-27; retrieved assets are recorded with SHA-256 in each snapshot's `manifest.json`.

## Data assets used by the implementation

| Dataset | Release URL pattern | License / terms | Required credit | Redistribution decision |
|---|---|---|---|---|
| Play-by-play | `github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet` | [nflverse-data license](https://github.com/nflverse/nflverse-data/blob/master/LICENSE) | nflverse / nflfastR | Not redistributed; raw files stay under ignored `data/`. |
| Schedules | `.../schedules/games.parquet` | same | nflverse | Not redistributed. |
| FTN charting | `.../ftn_charting/ftn_charting_{season}.parquet` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | FTN Data via nflverse | Aggregates shown in the UI carry attribution; raw rows not redistributed. |
| Participation | `.../pbp_participation/pbp_participation_{season}.parquet` | CC BY-SA 4.0 ([legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)) | NFL NextGenStats via nflverse (pre-2023); FTN Data via nflverse (2023+) | Same as above. Trained weights: review before any public release (12 §Data rights). |

## Provider documentation

- [nflreadpy loader reference](https://nflreadpy.nflverse.com/api/load_functions/) and [participation loader source](https://github.com/nflverse/nflreadpy/blob/main/src/nflreadpy/load_participation.py)
- [nflverse availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
- [Machine-readable PBP dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_pbp.csv)
- [FTN charting loader](https://nflreadr.nflverse.com/reference/load_ftn_charting.html) and [dictionary](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)
- [Participation loader and attribution](https://nflreadr.nflverse.com/reference/load_participation.html) and [dictionary](https://nflreadr.nflverse.com/articles/dictionary_participation.html)
- [Depth-chart dictionary](https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html), [NGS summary dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)

## Methods

- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Lakshminarayanan et al., Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Tibshirani et al., Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)
- [Dudík, Langford and Li, Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)
- [Chernozhukov et al., Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060)
- [Stan user guide, hierarchical regression](https://mc-stan.org/docs/stan-users-guide/regression.html#hierarchical-regression)

## Engineering

- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/), [DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency)
- [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/), [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/), [JSON Schema 2020-12](https://json-schema.org/draft/2020-12)
- [MLflow model registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/), [Next.js deployment](https://nextjs.org/docs/app/getting-started/deploying)
- [W3C WCAG 2.2](https://www.w3.org/TR/WCAG22/), [target size (minimum)](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum)
