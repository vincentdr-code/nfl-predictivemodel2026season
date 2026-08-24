"""
Ingest game schedules and play-by-play data from nfl_data_py.
"""
import os
import pandas as pd
import nfl_data_py as nfl
from pathlib import Path


def get_schedules(seasons, cache_dir="data/raw"):
    """
    Pull NFL schedules for given seasons. Cache locally to avoid re-downloading.

    Args:
        seasons: list of season years (e.g., [2002, 2003, ...])
        cache_dir: directory to cache parquet files

    Returns:
        DataFrame with columns: game_id, season, week, home_team, away_team,
                               home_score, away_score, roof, surface, stadium,
                               lat, lon, kickoff_time
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "schedules.parquet")

    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    print(f"Pulling schedules for seasons {seasons[0]}–{seasons[-1]}...")
    schedules = nfl.import_schedules(seasons)

    # Rename/select columns for consistency
    schedules = schedules.rename(columns={
        "week": "week",
        "season": "season",
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
    })

    # Keep only outdoor/dome info if available, else set to unknown
    if "roof" not in schedules.columns:
        schedules["roof"] = "unknown"

    if "surface" not in schedules.columns:
        schedules["surface"] = "unknown"

    if "stadium" not in schedules.columns:
        schedules["stadium"] = "unknown"

    # Lat/lon for stadiums (nfl_data_py may or may not include these)
    if "lat" not in schedules.columns:
        schedules["lat"] = None
    if "lon" not in schedules.columns:
        schedules["lon"] = None

    if "kickoff_time" not in schedules.columns:
        schedules["kickoff_time"] = None

    schedules.to_parquet(cache_file)
    print(f"Cached schedules to {cache_file}")

    return schedules


def get_pbp(seasons, cache_dir="data/raw"):
    """
    Pull play-by-play data for EPA and advanced stats. Cache locally.

    Args:
        seasons: list of season years
        cache_dir: directory to cache parquet files

    Returns:
        DataFrame with PBP columns including EPA, epa, home_team, away_team, etc.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "pbp.parquet")

    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    print(f"Pulling play-by-play for seasons {seasons[0]}–{seasons[-1]}...")
    pbp = nfl.import_pbp_data(seasons)

    pbp.to_parquet(cache_file)
    print(f"Cached PBP to {cache_file}")

    return pbp


def aggregate_pbp_to_game(pbp):
    """
    Aggregate play-by-play to game level: compute EPA per play and team.

    Args:
        pbp: DataFrame from get_pbp()

    Returns:
        DataFrame with game_id, home_team, away_team,
                     home_epa_per_play, away_epa_per_play, etc.
    """
    # Group by game and team; sum EPA
    game_stats = pbp.groupby(['game_id', 'posteam']).agg({
        'epa': 'sum',
        'play_type': 'count',  # number of plays
    }).reset_index()

    game_stats.columns = ['game_id', 'team', 'total_epa', 'play_count']
    game_stats['epa_per_play'] = game_stats['total_epa'] / game_stats['play_count']

    return game_stats


if __name__ == "__main__":
    # Quick test
    seasons = list(range(2020, 2026))
    scheds = get_schedules(seasons)
    print(f"Schedules shape: {scheds.shape}")
    print(scheds.head())
