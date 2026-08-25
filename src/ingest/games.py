"""
Ingest game schedules and play-by-play data from nfl_data_py.

The schedules table already includes weather (temp, wind), QB names,
rest days, and divisional flags — no separate joins needed for those.
"""
import os
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from pathlib import Path


SCHEDULE_KEEP_COLS = [
    'game_id', 'season', 'game_type', 'week', 'gameday', 'gametime',
    'home_team', 'away_team', 'home_score', 'away_score',
    'result', 'total', 'overtime',
    'home_rest', 'away_rest',
    'spread_line', 'total_line',
    'div_game', 'roof', 'surface', 'stadium',
    'temp', 'wind',
    'home_qb_name', 'away_qb_name',
    'home_coach', 'away_coach',
]


def get_schedules(seasons, cache_dir="data/raw", refresh=False):
    """
    Pull NFL schedules for given seasons. Cache locally.

    Args:
        seasons: list of season years
        cache_dir: directory to cache parquet
        refresh: if True, re-download even if cache exists

    Returns:
        DataFrame filtered to regular-season games with final scores
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "schedules.parquet")

    if os.path.exists(cache_file) and not refresh:
        return pd.read_parquet(cache_file)

    print(f"Pulling schedules for seasons {seasons[0]}-{seasons[-1]}...")
    schedules = nfl.import_schedules(list(seasons))

    # Keep only useful columns that exist
    keep = [c for c in SCHEDULE_KEEP_COLS if c in schedules.columns]
    schedules = schedules[keep].copy()

    # Regular season only; keep BOTH played and upcoming games so the pipeline
    # can generate predictions for the current season's remaining weeks.
    schedules = schedules[schedules['game_type'] == 'REG'].copy()

    played = schedules['home_score'].notna() & schedules['away_score'].notna()
    schedules['played'] = played.astype(int)
    schedules['home_win'] = np.where(played,
                                     (schedules['home_score'] > schedules['away_score']).astype(int),
                                     np.nan)
    schedules['margin'] = np.where(played,
                                   schedules['home_score'] - schedules['away_score'],
                                   np.nan)

    schedules.to_parquet(cache_file)
    print(f"  Cached {len(schedules)} regular-season games to {cache_file}")

    return schedules


PBP_KEEP_COLS = [
    'game_id', 'season', 'week', 'posteam', 'defteam',
    'play_type', 'epa', 'yards_gained',
    'third_down_converted', 'third_down_failed',
    'yardline_100', 'touchdown', 'field_goal_result',
    'fumble_lost', 'interception', 'pass', 'rush',
    'success', 'first_down',
    'passer_player_name',  # for QB ratings
    'qb_dropback',
]


def _get_pbp_one_season(season, cache_dir, max_retries=3):
    """
    Download PBP for a single season with retries. Cache per season.
    Returns empty DataFrame if the season's data doesn't exist yet (404).
    """
    per_season_dir = os.path.join(cache_dir, "pbp_by_season")
    Path(per_season_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(per_season_dir, f"pbp_{season}.parquet")

    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            df = nfl.import_pbp_data([season], columns=PBP_KEEP_COLS, downcast=True)
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            last_err = e
            msg = str(e)
            # 404 = season not published yet; don't retry, don't fail
            if '404' in msg or 'Not Found' in msg or (
                isinstance(e, NameError) and "'Error' is not defined" in msg
            ):
                print(f"    season {season}: no data available yet (404). Skipping.")
                empty = pd.DataFrame(columns=PBP_KEEP_COLS)
                empty.to_parquet(cache_file)  # cache empty so we don't retry every run
                return empty
            print(f"    season {season} attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")

    raise RuntimeError(f"Failed to download PBP for season {season} after {max_retries} attempts") from last_err


def get_pbp(seasons, cache_dir="data/raw", refresh=False):
    """
    Pull play-by-play, one season at a time (network-robust). Cache per season
    and also as a combined parquet.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "pbp.parquet")

    if os.path.exists(cache_file) and not refresh:
        return pd.read_parquet(cache_file)

    print(f"Pulling play-by-play for {len(seasons)} seasons (per-season, cached)...")
    frames = []
    for season in seasons:
        print(f"  season {season}...", end=" ", flush=True)
        df = _get_pbp_one_season(season, cache_dir)
        print(f"{len(df):,} plays")
        frames.append(df)

    pbp = pd.concat(frames, ignore_index=True)
    pbp.to_parquet(cache_file)
    print(f"  Cached {len(pbp):,} plays to {cache_file}")

    return pbp


def aggregate_pbp_to_team_game(pbp):
    """
    Aggregate PBP to (game_id, team) level for offense.
    Turnovers are the sum of fumble_lost + interception.

    Args:
        pbp: DataFrame from get_pbp()

    Returns:
        DataFrame with columns:
            game_id, team, plays, off_epa_per_play, third_down_pct,
            turnovers, success_rate
    """
    # Filter to plays with a possession team (offense/defense on the field)
    offense = pbp[pbp['posteam'].notna() & (pbp['play_type'].isin(['pass', 'run']))].copy()

    off_stats = offense.groupby(['game_id', 'posteam']).agg(
        plays=('epa', 'count'),
        off_epa_sum=('epa', 'sum'),
        third_conv=('third_down_converted', 'sum'),
        third_fail=('third_down_failed', 'sum'),
        fumbles=('fumble_lost', 'sum'),
        ints=('interception', 'sum'),
        success=('success', 'sum'),
    ).reset_index().rename(columns={'posteam': 'team'})

    off_stats['off_epa_per_play'] = off_stats['off_epa_sum'] / off_stats['plays'].clip(lower=1)
    third_attempts = off_stats['third_conv'] + off_stats['third_fail']
    off_stats['third_down_pct'] = np.where(
        third_attempts > 0,
        off_stats['third_conv'] / third_attempts.clip(lower=1),
        0.0
    )
    off_stats['turnovers'] = off_stats['fumbles'].fillna(0) + off_stats['ints'].fillna(0)
    off_stats['success_rate'] = off_stats['success'] / off_stats['plays'].clip(lower=1)

    return off_stats[[
        'game_id', 'team', 'plays', 'off_epa_per_play',
        'third_down_pct', 'turnovers', 'success_rate'
    ]]


def build_game_team_table(schedules, pbp):
    """
    Explode schedule into (game_id, team, is_home, opponent, points_for, points_against)
    and merge in offensive stats. Defensive stats = opponent's offensive stats.

    Returns:
        DataFrame with one row per team per game.
    """
    off_stats = aggregate_pbp_to_team_game(pbp)

    # Long format: one row per team per game
    home_rows = schedules[['game_id', 'season', 'week', 'gameday',
                           'home_team', 'away_team',
                           'home_score', 'away_score']].copy()
    home_rows.columns = ['game_id', 'season', 'week', 'gameday',
                         'team', 'opponent', 'points_for', 'points_against']
    home_rows['is_home'] = 1

    away_rows = schedules[['game_id', 'season', 'week', 'gameday',
                           'away_team', 'home_team',
                           'away_score', 'home_score']].copy()
    away_rows.columns = ['game_id', 'season', 'week', 'gameday',
                         'team', 'opponent', 'points_for', 'points_against']
    away_rows['is_home'] = 0

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    # Merge offensive stats for this team
    long = long.merge(off_stats, on=['game_id', 'team'], how='left')

    # Defensive stats = opponent's offensive stats
    opp_stats = off_stats.rename(columns={
        'team': 'opponent',
        'off_epa_per_play': 'def_epa_per_play_allowed',
        'third_down_pct': 'def_third_down_pct_allowed',
        'turnovers': 'takeaways',
        'success_rate': 'def_success_rate_allowed',
        'plays': 'def_plays',
    })
    long = long.merge(
        opp_stats[['game_id', 'opponent', 'def_epa_per_play_allowed',
                   'def_third_down_pct_allowed', 'takeaways',
                   'def_success_rate_allowed']],
        on=['game_id', 'opponent'], how='left'
    )

    return long.sort_values(['season', 'week', 'game_id', 'is_home'],
                            ascending=[True, True, True, False]).reset_index(drop=True)


if __name__ == "__main__":
    seasons = list(range(2023, 2026))
    scheds = get_schedules(seasons)
    print(f"Schedules: {scheds.shape}")
    print(scheds.head(3).to_string())
