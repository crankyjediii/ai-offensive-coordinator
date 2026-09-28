"""Source adapters: raw assets -> canonical silver tables with lineage and availability times."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from coordinator import config
from coordinator.data.ingest import SourceAsset, asset_path, existing_assets
from coordinator.taxonomy.labels import action_expr, cohort_exprs, outcome_exprs
from coordinator.taxonomy.vocab import COVERAGE_SOURCE_MAP, ROUTE_MAP

UTC_DT = pl.Datetime("us", "UTC")


def load_policy() -> dict:
    return yaml.safe_load((config.CONFIGS_DIR / "availability_policy.yaml").read_text())


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _lineage(df: pl.DataFrame, asset: SourceAsset, key_cols: list[str]) -> pl.DataFrame:
    ingested = _ts(asset.retrieved_at)
    published = _ts(asset.source_published_at)
    archived_known = max(x for x in (ingested, published) if x is not None)
    return df.with_columns(
        pl.lit(asset.source_asset_id).alias("record_version_id"),
        pl.lit(asset.source_asset_id).alias("source_asset_id"),
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in key_cols], separator=":").alias("source_row_key"),
        pl.lit(ingested).cast(UTC_DT).alias("ingested_at"),
        pl.lit(published).cast(UTC_DT).alias("source_available_at"),
        pl.lit(asset.source_available_basis).alias("source_available_basis"),
        pl.lit(archived_known).cast(UTC_DT).alias("known_at_archived"),
    )


def build_games(asset: SourceAsset) -> pl.DataFrame:
    policy = load_policy()
    raw = pl.read_parquet(asset_path(asset))
    kickoff_local = pl.concat_str([pl.col("gameday"), pl.col("gametime").fill_null("13:00")], separator=" ")
    df = raw.select(
        "game_id",
        pl.col("season").cast(pl.Int16),
        pl.when(pl.col("game_type") == "REG").then(pl.lit("REG")).otherwise(pl.lit("POST")).alias("season_type"),
        pl.col("week").cast(pl.Int16),
        pl.col("home_team").alias("home_team_id"),
        pl.col("away_team").alias("away_team_id"),
        kickoff_local.str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
        .alias("kickoff_at"),
        pl.col("result").is_not_null().alias("completed"),
        pl.col("home_score"),
        pl.col("away_score"),
        pl.col("home_coach"),
        pl.col("away_coach"),
    )
    hours = policy["game_duration_hours"]
    df = df.with_columns(
        pl.when(pl.col("completed"))
        .then(pl.col("kickoff_at") + pl.duration(hours=hours))
        .otherwise(None)
        .alias("completed_at_est"),
        pl.lit("game_end_estimated").alias("event_time_precision"),
    )
    return _lineage(df, asset, ["game_id"])


def build_plays(assets: list[SourceAsset], games: pl.DataFrame) -> pl.DataFrame:
    policy = load_policy()["sources"]["pbp"]
    frames = []
    for asset in assets:
        raw = pl.read_parquet(asset_path(asset))
        action, reason = action_expr()
        df = raw.with_columns(
            pl.col("posteam").alias("offense_id"),
            pl.col("defteam").alias("defense_id"),
            action.alias("action"),
            reason.alias("action_reason"),
        )
        df = df.with_columns(outcome_exprs()).with_columns(cohort_exprs())
        df = df.select(
            "game_id",
            pl.col("play_id").cast(pl.Int64),
            pl.col("drive").cast(pl.Int32).alias("drive_id"),
            pl.col("order_sequence").cast(pl.Int32).alias("play_order"),
            pl.col("season").cast(pl.Int16),
            "season_type",
            pl.col("week").cast(pl.Int16),
            "offense_id",
            "defense_id",
            pl.col("qtr").cast(pl.Int8).alias("quarter"),
            pl.col("quarter_seconds_remaining").cast(pl.Int32).alias("seconds_remaining"),
            pl.col("half_seconds_remaining").cast(pl.Int32).alias("half_seconds_remaining"),
            pl.col("down").cast(pl.Int8),
            pl.col("ydstogo").cast(pl.Int16).alias("distance"),
            pl.col("yardline_100").cast(pl.Int16).alias("yards_to_goal"),
            (pl.col("goal_to_go").cast(pl.Float64) == 1).alias("goal_to_go"),
            pl.col("score_differential").cast(pl.Int16).alias("offense_score_margin"),
            pl.col("posteam_timeouts_remaining").cast(pl.Int8).alias("offense_timeouts"),
            pl.col("defteam_timeouts_remaining").cast(pl.Int8).alias("defense_timeouts"),
            pl.col("play_type").alias("play_type_raw"),
            "action",
            "action_reason",
            (pl.col("penalty").cast(pl.Float64) == 1).alias("penalty_flag"),
            pl.col("shotgun").cast(pl.Float64).alias("shotgun_pbp"),
            "y_epa", "y_success", "y_yards", "y_explosive", "y_sack", "y_turnover",
            "y_first_down_or_td", "y_failed_fourth_down",
            "is_scrimmage_eligible",
            "in_reg_scrimmage_penalty_free_v1",
            # Retrospective descriptors: never decision-time inputs.
            pl.col("run_location").alias("retro_run_location"),
            pl.col("run_gap").alias("retro_run_gap"),
            pl.col("pass_location").alias("retro_pass_location"),
            pl.col("air_yards").cast(pl.Float64).alias("retro_air_yards"),
        )
        frames.append(_lineage(df, asset, ["game_id", "play_id"]))
    plays = pl.concat(frames, how="vertical_relaxed")
    plays = plays.join(
        games.select("game_id", "kickoff_at", "completed_at_est"), on="game_id", how="left"
    )
    return plays.with_columns(
        pl.col("completed_at_est").alias("event_at"),
        (pl.col("completed_at_est") + pl.duration(hours=policy["delay_after_game_end_hours"])).alias(
            "known_at_reconstructed"
        ),
    )


def _null_zero(col: str, alias: str) -> pl.Expr:
    return pl.when(pl.col(col) <= 0).then(None).otherwise(pl.col(col)).cast(pl.Int8).alias(alias)


def build_charting(assets: list[SourceAsset], plays: pl.DataFrame) -> pl.DataFrame:
    """FTN charting. The source encodes 'not charted' as 0/'0'; those become null, never zero."""
    policy = load_policy()["sources"]["ftn_charting"]
    frames = []
    for asset in assets:
        raw = pl.read_parquet(asset_path(asset))
        df = raw.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("nflverse_play_id").cast(pl.Int64).alias("play_id"),
            pl.col("starting_hash").replace_strict({"L": "left", "M": "middle", "R": "right"}, default=None).alias("hash"),
            pl.col("qb_location").replace_strict(
                {"S": "shotgun", "U": "under_center", "P": "pistol"}, default=None
            ).alias("qb_alignment"),
            _null_zero("n_defense_box", "box_count"),
            pl.when((pl.col("n_pass_rushers") <= 0) | (pl.col("n_pass_rushers") > 11)).then(None)
            .otherwise(pl.col("n_pass_rushers")).cast(pl.Int8).alias("pass_rusher_count"),
            pl.when((pl.col("n_pass_rushers") <= 0) | (pl.col("n_pass_rushers") > 11)).then(None)
            .otherwise(pl.col("n_blitzers")).cast(pl.Int8).alias("blitzer_count"),
            (pl.col("n_pass_rushers") > 11).alias("rusher_count_quarantined"),
            pl.col("is_motion").alias("motion_observed"),
            pl.col("is_play_action").alias("play_action_observed"),
            pl.col("is_screen_pass").alias("screen_observed"),
            pl.col("is_rpo").alias("rpo_observed"),
            pl.col("date_pulled").cast(UTC_DT).alias("provider_pulled_at"),
        )
        frames.append(_lineage(df, asset, ["game_id", "play_id"]))
    ch = pl.concat(frames, how="vertical_relaxed")
    scrimmage = plays.select("game_id", "play_id", "is_scrimmage_eligible", "action", "completed_at_est")
    ch = ch.join(scrimmage, on=["game_id", "play_id"], how="left")
    # Attributes on non-scrimmage rows are not applicable; rush counts only apply to dropbacks.
    ok = pl.col("is_scrimmage_eligible").fill_null(False)
    ch = ch.with_columns(
        *[pl.when(ok).then(pl.col(c)).otherwise(None).alias(c)
          for c in ("hash", "qb_alignment", "box_count", "motion_observed", "play_action_observed",
                    "screen_observed", "rpo_observed")],
        *[pl.when(ok & (pl.col("action") == "dropback")).then(pl.col(c)).otherwise(None).alias(c)
          for c in ("pass_rusher_count", "blitzer_count")],
        (pl.col("completed_at_est") + pl.duration(hours=policy["delay_after_game_end_hours"])).alias(
            "known_at_reconstructed"
        ),
    )
    return ch.drop("is_scrimmage_eligible", "action", "completed_at_est")


PERSONNEL_POS = {"RB": "rb", "FB": "rb", "TE": "te", "WR": "wr"}


def _position_count(col: str, pos: str) -> pl.Expr:
    return (
        pl.col(col).str.extract(rf"(\d+) {pos}\b", 1).cast(pl.Int8).fill_null(0)
    )


def build_participation(assets: list[SourceAsset], plays: pl.DataFrame) -> pl.DataFrame:
    policy = load_policy()["sources"]["participation"]
    md = policy["available_month_day_after_season"]
    frames = []
    for asset in assets:
        raw = pl.read_parquet(asset_path(asset))
        season = asset.season or 0
        era = "ngs" if season < 2023 else "ftn"
        cov_raw = pl.when(pl.col("defense_coverage_type") == "").then(None).otherwise(pl.col("defense_coverage_type"))
        mz_raw = pl.when(pl.col("defense_man_zone_type") == "").then(None).otherwise(pl.col("defense_man_zone_type"))
        charted = cov_raw.is_not_null()
        df = raw.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("play_id").cast(pl.Int64),
            cov_raw.alias("coverage_raw"),
            (pl.when(charted).then(cov_raw.replace_strict(COVERAGE_SOURCE_MAP, default="UNMAPPED")).otherwise(None)
             if era == "ftn" else pl.lit(None, pl.Utf8)).alias("coverage"),
            (mz_raw if era == "ftn" else pl.lit(None, pl.Utf8)).alias("man_zone"),
            pl.when(charted).then(pl.col("was_pressure")).otherwise(None).alias("pressure_observed"),
            pl.when(pl.col("route").is_in(list(ROUTE_MAP))).then(pl.col("route").replace_strict(ROUTE_MAP, default=None)).otherwise(None).alias("retro_route"),
            pl.col("offense_personnel").alias("offense_personnel_raw"),
            pl.col("defense_personnel").alias("defense_personnel_raw"),
            pl.col("offense_formation").alias("offense_formation_raw"),
            pl.lit(era).alias("participation_era"),
        ).with_columns(
            (_position_count("offense_personnel_raw", "RB") + _position_count("offense_personnel_raw", "FB")).alias("off_rb"),
            _position_count("offense_personnel_raw", "TE").alias("off_te"),
            _position_count("offense_personnel_raw", "WR").alias("off_wr"),
            (_position_count("defense_personnel_raw", "CB") + _position_count("defense_personnel_raw", "FS")
             + _position_count("defense_personnel_raw", "SS") + _position_count("defense_personnel_raw", "DB")
             + _position_count("defense_personnel_raw", "S")).alias("def_db"),
        ).with_columns(
            pl.when(pl.col("offense_personnel_raw").is_null()).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("off_rb", "off_te", "off_wr")
        ).with_columns(
            pl.when(pl.col("defense_personnel_raw").is_null()).then(None).otherwise(pl.col("def_db")).alias("def_db")
        )
        available = datetime.fromisoformat(f"{season + 1}-{md}T00:00:00+00:00")
        df = df.with_columns(pl.lit(available).cast(UTC_DT).alias("known_at_reconstructed"))
        frames.append(_lineage(df, asset, ["game_id", "play_id"]))
    part = pl.concat(frames, how="vertical_relaxed")
    scrimmage = plays.select("game_id", "play_id", "is_scrimmage_eligible")
    part = part.join(scrimmage, on=["game_id", "play_id"], how="left")
    ok = pl.col("is_scrimmage_eligible").fill_null(False)
    return part.with_columns(
        pl.when(ok).then(pl.col(c)).otherwise(None).alias(c)
        for c in ("coverage", "man_zone", "pressure_observed", "off_rb", "off_te", "off_wr", "def_db")
    ).drop("is_scrimmage_eligible")


def silver_dir() -> Path:
    return config.SILVER_DIR / config.SCHEMA_VERSION


def build_silver() -> dict[str, int]:
    """Rebuild all canonical tables from every retrieved raw asset version."""
    out = silver_dir()
    out.mkdir(parents=True, exist_ok=True)
    sched = max(existing_assets("schedules"), key=lambda a: a.retrieved_at)
    games = build_games(sched)
    plays = build_plays(existing_assets("pbp"), games)
    charting = build_charting(existing_assets("ftn_charting"), plays)
    participation = build_participation(existing_assets("participation"), plays)
    tables = {"game": games, "play": plays, "charting_observation": charting, "participation_observation": participation}
    for name, df in tables.items():
        df.write_parquet(out / f"{name}.parquet")
    (out / "_built_at.txt").write_text(datetime.now(UTC).isoformat())
    return {k: v.height for k, v in tables.items()}


def read_silver(name: str) -> pl.DataFrame:
    return pl.read_parquet(silver_dir() / f"{name}.parquet")
