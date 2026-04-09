"""
data_fetch.py — Pull WNBA player stats from nba_api and cache locally.

Three endpoints are merged on PLAYER_ID:
  - leaguedashplayerstats      : box score counting stats
  - leaguedashplayerbiostats   : height, draft year (used to derive WNBA experience)
  - leaguedashplayeradvancedstats : usage, true shooting, PIE

A derived column WNBA_EXPERIENCE is computed as (season_year - DRAFT_YEAR).
Players with a missing or invalid DRAFT_YEAR get NaN and are filtered out
by the processor's min_minutes / dropna step.

Run directly to refresh the cache:
    python -m src.data_fetch           # use cached file if it exists
    python -m src.data_fetch --refresh # force a fresh API pull
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerBioStats, LeagueDashPlayerStats

from src.utils import clean_name, ensure_dirs, load_config

# nba_api column subsets we actually need from each endpoint
BOX_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3_PCT",
]
BIO_COLS = [
    "PLAYER_ID", "PLAYER_HEIGHT_INCHES", "DRAFT_YEAR",
]
ADV_COLS = [
    "PLAYER_ID", "USG_PCT", "TS_PCT", "PIE",
]


def _fetch_box(league_id: str, season: str) -> pd.DataFrame:
    print(f"  Fetching box score stats (season={season})...")
    time.sleep(1)  # be polite to the API
    resp = LeagueDashPlayerStats(
        league_id_nullable=league_id,
        season=season,
        per_mode_detailed="Totals",
    )
    df = resp.get_data_frames()[0]
    return df[[c for c in BOX_COLS if c in df.columns]]


def _fetch_bio(league_id: str, season: str) -> pd.DataFrame:
    print("  Fetching bio stats (height, draft year)...")
    time.sleep(1)
    resp = LeagueDashPlayerBioStats(
        league_id=league_id,
        season=season,
    )
    df = resp.get_data_frames()[0]
    return df[[c for c in BIO_COLS if c in df.columns]]


def _fetch_advanced(league_id: str, season: str) -> pd.DataFrame:
    print("  Fetching advanced stats (USG%, TS%, PIE)...")
    time.sleep(1)
    # LeagueDashPlayerStats with measure_type_detailed_defense="Advanced" gives USG_PCT, TS_PCT, PIE
    resp = LeagueDashPlayerStats(
        league_id_nullable=league_id,
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    df = resp.get_data_frames()[0]
    return df[[c for c in ADV_COLS if c in df.columns]]


def fetch_and_cache(config: dict, refresh: bool = False) -> pd.DataFrame:
    """
    Pull all three endpoints and merge into one DataFrame.
    Writes to config['data']['raw_path']. Returns the merged DataFrame.
    If the cache file already exists and refresh=False, loads from disk instead.
    """
    ensure_dirs(config)
    cache_path = Path(config["data"]["raw_path"])
    league_id = config["data"]["league_id"]
    season = config["data"]["season"]

    if cache_path.exists() and not refresh:
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Fetching fresh data from nba_api (WNBA {season})...")
    box = _fetch_box(league_id, season)
    bio = _fetch_bio(league_id, season)
    adv = _fetch_advanced(league_id, season)

    df = box.merge(bio, on="PLAYER_ID", how="left")
    df = df.merge(adv, on="PLAYER_ID", how="left")

    # Derive years of WNBA experience from draft year.
    # DRAFT_YEAR of 0 means undrafted/unknown — treat as NaN.
    season_year = int(season)
    df["DRAFT_YEAR"] = pd.to_numeric(df["DRAFT_YEAR"], errors="coerce").replace(0, pd.NA)
    df["WNBA_EXPERIENCE"] = season_year - df["DRAFT_YEAR"]

    # Add a clean name column for reliable downstream joining
    df["name_clean"] = df["PLAYER_NAME"].apply(clean_name)

    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} players to {cache_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a fresh pull from nba_api, overwriting the cache.",
    )
    args = parser.parse_args()

    cfg = load_config()
    df = fetch_and_cache(cfg, refresh=args.refresh)
    print(df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS"]].head(10))
