"""Registry of approved source datasets and their licensing/availability metadata."""

from __future__ import annotations

from dataclasses import dataclass

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
NFLVERSE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags"


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    provider: str
    release_tag: str
    filename: str  # may contain {season}
    per_season: bool
    license_url: str
    attribution: str
    first_season: int

    def asset_name(self, season: int | None) -> str:
        return self.filename.format(season=season) if self.per_season else self.filename

    def url(self, season: int | None) -> str:
        return f"{NFLVERSE_RELEASES}/{self.release_tag}/{self.asset_name(season)}"


SOURCES: dict[str, SourceSpec] = {
    "pbp": SourceSpec(
        dataset="pbp",
        provider="nflverse",
        release_tag="pbp",
        filename="play_by_play_{season}.parquet",
        per_season=True,
        license_url="https://github.com/nflverse/nflverse-data/blob/master/LICENSE",
        attribution="nflverse play-by-play (nflfastR)",
        first_season=1999,
    ),
    "schedules": SourceSpec(
        dataset="schedules",
        provider="nflverse",
        release_tag="schedules",
        filename="games.parquet",
        per_season=False,
        license_url="https://github.com/nflverse/nflverse-data/blob/master/LICENSE",
        attribution="nflverse schedules (Lee Sharpe)",
        first_season=1999,
    ),
    "ftn_charting": SourceSpec(
        dataset="ftn_charting",
        provider="ftn_via_nflverse",
        release_tag="ftn_charting",
        filename="ftn_charting_{season}.parquet",
        per_season=True,
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="FTN Data via nflverse (CC BY-SA 4.0)",
        first_season=2022,
    ),
    "participation": SourceSpec(
        dataset="participation",
        provider="nflverse",
        release_tag="pbp_participation",
        filename="pbp_participation_{season}.parquet",
        per_season=True,
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="NFL NextGenStats via nflverse (pre-2023); FTN Data via nflverse (2023+); CC BY-SA 4.0",
        first_season=2016,
    ),
}
