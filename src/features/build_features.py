"""
Build the feature matrix for model training.

STRICT RULE: features for a game at (season, week) may ONLY use data
from games with earlier (season, week) than the target game. No same-week
data, no future data. This is enforced by using `.shift(1)` on rolling
windows computed after chronological sorting per team.
"""
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.ingest import games as ingest_games
from src.models.elo import EloRating
from src.models.qb_ratings import build_qb_ratings_history, match_qb_to_schedule


ROLLING_STATS = [
    'points_for', 'points_against',
    'off_epa_per_play', 'def_epa_per_play_allowed',
    'third_down_pct', 'def_third_down_pct_allowed',
    'turnovers', 'takeaways',
    'success_rate', 'def_success_rate_allowed',
]


def add_rolling_team_features(team_game, windows=(4,)):
    """
    For each (team, season), add trailing-window rolling means for the
    stats in ROLLING_STATS. Uses .shift(1) so the current game is excluded.

    Returns team_game with new columns like `off_epa_per_play_r4`,
    `off_epa_per_play_std` (season-to-date mean).
    """
    df = team_game.sort_values(['team', 'season', 'week', 'game_id']).copy()

    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        grouped = df.groupby(['team', 'season'])[stat]
        shifted = grouped.shift(1)  # Exclude current game
        for w in windows:
            df[f'{stat}_r{w}'] = shifted.groupby([df['team'], df['season']]).transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
        # Season-to-date (expanding mean, excluding current game)
        df[f'{stat}_std'] = shifted.groupby([df['team'], df['season']]).transform(
            lambda s: s.expanding().mean()
        )

    return df


def add_prior_season_baselines(team_game):
    """
    For each (team, season), add the team's prior-season mean of key stats.
    Used to seed early-season features when rolling windows are empty.
    """
    df = team_game.copy()

    season_means = df.groupby(['team', 'season'])[ROLLING_STATS].mean().reset_index()
    season_means.columns = ['team', 'season'] + [f'{s}_prior' for s in ROLLING_STATS]
    season_means['season'] = season_means['season'] + 1  # shift so prior season lines up

    df = df.merge(season_means, on=['team', 'season'], how='left')

    # For any rolling column that's NaN (early season), fall back to prior-season mean
    for stat in ROLLING_STATS:
        for suffix in ['_r4', '_std']:
            col = f'{stat}{suffix}'
            prior = f'{stat}_prior'
            if col in df.columns and prior in df.columns:
                df[col] = df[col].fillna(df[prior])

    return df


def compute_elo_features(schedules, config):
    """
    Play through all games chronologically, recording PRE-game Elo for
    every game (no leakage).

    Returns DataFrame indexed by game_id with home_elo, away_elo, elo_diff_hfa,
    elo_win_prob_home columns.
    """
    elo = EloRating(**config['elo'])
    games = schedules.sort_values(['season', 'week', 'gameday']).reset_index(drop=True)

    rows = []
    prev_season = None
    for _, g in games.iterrows():
        if prev_season is not None and int(g['season']) != prev_season:
            elo.revert_season()

        home_pre = elo.get(g['home_team'])
        away_pre = elo.get(g['away_team'])
        elo_diff = (home_pre + elo.home_advantage) - away_pre
        win_prob = elo.get_win_probability(elo_diff)

        rows.append({
            'game_id': g['game_id'],
            'home_elo': home_pre,
            'away_elo': away_pre,
            'elo_diff_hfa': elo_diff,
            'elo_win_prob_home': win_prob,
        })

        # Update ratings only for played games
        if pd.notna(g['home_score']) and pd.notna(g['away_score']):
            elo.process_game(g['home_team'], g['away_team'],
                             g['home_score'], g['away_score'])
        prev_season = int(g['season'])

    return pd.DataFrame(rows)


def build_feature_matrix(schedules, team_game, config):
    """
    Build the per-game feature matrix by joining home-team and away-team
    rolling features + Elo features + situational schedule features.

    Returns a DataFrame with one row per game and columns suitable for
    LightGBM training. Target columns: home_win, margin, total.
    """
    windows = config['features']['rolling_windows']

    print("  Adding rolling team features...")
    tg = add_rolling_team_features(team_game, windows=windows)

    print("  Adding prior-season baselines...")
    tg = add_prior_season_baselines(tg)

    feature_cols = [c for c in tg.columns
                    if any(c.startswith(s) and (c.endswith('_r4') or c.endswith('_std'))
                           for s in ROLLING_STATS)]

    print(f"  {len(feature_cols)} team-level feature columns")

    # Pivot home and away sides
    home_tg = tg[tg['is_home'] == 1][['game_id', 'team'] + feature_cols].copy()
    home_tg = home_tg.rename(columns={c: f'home_{c}' for c in feature_cols})
    home_tg = home_tg.rename(columns={'team': 'home_team'})

    away_tg = tg[tg['is_home'] == 0][['game_id', 'team'] + feature_cols].copy()
    away_tg = away_tg.rename(columns={c: f'away_{c}' for c in feature_cols})
    away_tg = away_tg.rename(columns={'team': 'away_team'})

    # Join to schedule
    schedule_cols = ['game_id', 'season', 'week', 'gameday',
                     'home_team', 'away_team',
                     'home_score', 'away_score', 'home_win', 'margin',
                     'total', 'spread_line', 'total_line',
                     'home_rest', 'away_rest', 'div_game',
                     'roof', 'temp', 'wind',
                     'home_qb_name', 'away_qb_name',
                     # Injury flags (added in v2)
                     'home_qb_out', 'away_qb_out',
                     'home_qb_questionable', 'away_qb_questionable',
                     'home_key_out', 'away_key_out']
    available = [c for c in schedule_cols if c in schedules.columns]
    feats = schedules[available].copy()

    feats = feats.merge(home_tg, on=['game_id', 'home_team'], how='left')
    feats = feats.merge(away_tg, on=['game_id', 'away_team'], how='left')

    # Add Elo features
    print("  Computing Elo features chronologically...")
    elo_df = compute_elo_features(schedules, config)
    feats = feats.merge(elo_df, on='game_id', how='left')

    # Add QB rating features
    processed = config['data']['processed_dir']
    qb_path = Path(processed) / "qb_ratings.parquet"
    if qb_path.exists():
        print("  Merging QB ratings...")
        qb_ratings = pd.read_parquet(qb_path)
        feats = match_qb_to_schedule(feats, qb_ratings)
    else:
        print("  (skip) qb_ratings.parquet not built yet; run src.models.qb_ratings first")
        feats['home_qb_rating'] = 0.0
        feats['away_qb_rating'] = 0.0
        feats['qb_rating_diff'] = 0.0

    # Derived features
    feats['rest_diff'] = feats['home_rest'].fillna(7) - feats['away_rest'].fillna(7)
    feats['is_dome'] = feats['roof'].isin(['dome', 'closed']).astype(int)
    feats['is_outdoor'] = feats['roof'].isin(['outdoors', 'open']).astype(int)
    feats['temp'] = feats['temp'].fillna(70)  # median indoor temp proxy
    feats['wind'] = feats['wind'].fillna(0)
    feats['bad_weather'] = ((feats['wind'] >= config['features']['bad_weather_wind_threshold_mph'])
                            & (feats['is_outdoor'] == 1)).astype(int)
    feats['div_game'] = feats['div_game'].fillna(0).astype(int)

    return feats


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    processed = config['data']['processed_dir']

    print("Loading data...")
    schedules = pd.read_csv(f"{processed}/games_master.csv")
    team_game = pd.read_parquet(f"{processed}/team_game_stats.parquet")
    print(f"  {len(schedules):,} games, {len(team_game):,} team-game rows")

    print("Building feature matrix...")
    feats = build_feature_matrix(schedules, team_game, config)

    output = f"{processed}/features.parquet"
    feats.to_parquet(output)
    print(f"[OK] Wrote {len(feats):,} rows x {len(feats.columns)} cols to {output}")

    print("\nSample:")
    print(feats.head(3).to_string())

    return feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
