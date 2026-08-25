"""
Power rankings from current Elo ratings, with week-over-week movement.

Also computes a small historical Elo timeseries per team for sparklines.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.models.elo import EloRating


def compute_power_rankings(schedules, config, through_week=None, season=None):
    """
    Play through games chronologically and snapshot post-game Elo.
    Returns per-team snapshots at end of each week.

    If season/through_week given, returns rankings AT end of that specific week.
    Otherwise returns full timeseries.
    """
    elo_config = config['elo']
    elo = EloRating(**elo_config)

    played = schedules[schedules['home_score'].notna() & schedules['away_score'].notna()].copy()
    played = played.sort_values(['season', 'week', 'gameday']).reset_index(drop=True)

    snapshots = []  # (season, week, team, elo)
    prev_season = None

    for _, g in played.iterrows():
        s = int(g['season'])
        if prev_season is not None and s != prev_season:
            elo.revert_season()
        elo.process_game(g['home_team'], g['away_team'], g['home_score'], g['away_score'])
        prev_season = s

        # Snapshot after this game
        for team, rating in elo.ratings.items():
            snapshots.append({'season': s, 'week': int(g['week']), 'team': team, 'elo': float(rating)})

    df = pd.DataFrame(snapshots)
    # For each (season, week, team), take the last snapshot (end-of-week rating)
    latest_per_week = df.groupby(['season', 'week', 'team'])['elo'].last().reset_index()

    if season is not None and through_week is not None:
        end_week = int(through_week)
        this_week = latest_per_week[(latest_per_week['season'] == season)
                                     & (latest_per_week['week'] == end_week)]
        prev = latest_per_week[(latest_per_week['season'] == season)
                                & (latest_per_week['week'] == end_week - 1)]

        this_week = this_week.sort_values('elo', ascending=False).reset_index(drop=True)
        this_week['rank'] = this_week.index + 1

        # Prior week's ranks for movement
        prev_ranked = prev.sort_values('elo', ascending=False).reset_index(drop=True)
        prev_ranked['prev_rank'] = prev_ranked.index + 1
        this_week = this_week.merge(prev_ranked[['team', 'prev_rank']], on='team', how='left')
        this_week['rank_change'] = this_week['prev_rank'] - this_week['rank']  # positive = moved up

        return this_week, latest_per_week
    return None, latest_per_week


def latest_snapshot(schedules, config):
    """Return the most recent power rankings (end of latest played week)."""
    played = schedules[schedules['home_score'].notna()]
    if len(played) == 0:
        return None, None
    latest_season = int(played['season'].max())
    latest_week = int(played[played['season'] == latest_season]['week'].max())
    return compute_power_rankings(schedules, config, through_week=latest_week, season=latest_season)


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    processed = config['data']['processed_dir']
    outputs = config['data']['outputs_dir']

    print("Loading schedules...")
    schedules = pd.read_csv(f"{processed}/games_master.csv")

    rankings, history = latest_snapshot(schedules, config)
    if rankings is None:
        print("No played games yet.")
        return

    rankings['name'] = rankings['team']
    print("\nPower Rankings — End of latest played week")
    print(f"{'Rank':<5} {'Chg':<5} {'Team':<5} {'Elo':<7}")
    print("-" * 25)
    for _, r in rankings.iterrows():
        chg = int(r['rank_change']) if pd.notna(r['rank_change']) else 0
        chg_str = f"{'+' if chg > 0 else ''}{chg}" if chg != 0 else "—"
        print(f"{int(r['rank']):<5} {chg_str:<5} {r['team']:<5} {r['elo']:<7.1f}")

    Path(outputs).mkdir(parents=True, exist_ok=True)
    rankings.to_csv(f"{outputs}/power_rankings.csv", index=False)
    history.to_parquet(f"{outputs}/elo_history.parquet")
    print(f"\n[OK] Saved power_rankings.csv + elo_history.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
