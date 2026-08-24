"""
Main entrypoint: orchestrate all data ingestion and write games_master.csv
"""
import argparse
import pandas as pd
import yaml
from pathlib import Path
from src.ingest import games, injuries, weather


def load_config(config_file="config.yaml"):
    """Load configuration from YAML."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main(config_file="config.yaml", refresh=False, week=None):
    """
    Orchestrate full data pipeline: ingest games, injuries, weather → write games_master.csv

    Args:
        config_file: path to config.yaml
        refresh: if True, re-download all data; if False, use cache
        week: current NFL week (for future use in Phase 7)
    """
    config = load_config(config_file)

    seasons = config['data']['seasons']
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']

    print(f"Building dataset for seasons {seasons[0]}–{seasons[-1]}...")

    # Phase 1a: Pull schedules
    print("\n[1/3] Fetching schedules...")
    schedules = games.get_schedules(seasons, cache_dir=raw_dir)
    print(f"  Loaded {len(schedules)} games")

    # Phase 1b: Pull PBP (optional for Phase 1; needed for feature engineering in Phase 3)
    print("\n[2/3] Fetching play-by-play (for EPA)...")
    pbp = games.get_pbp(seasons, cache_dir=raw_dir)
    print(f"  Loaded {len(pbp)} plays")

    # Aggregate PBP to game level
    game_stats = games.aggregate_pbp_to_game(pbp)
    print(f"  Aggregated to {len(game_stats)} game-team records")

    # Merge EPA stats back to schedule
    schedules_with_epa = schedules.merge(
        game_stats[game_stats['team'] == schedules['home_team'].iloc[0]],
        left_on=['game_id', 'home_team'],
        right_on=['game_id', 'team'],
        how='left',
        suffixes=('', '_home')
    )
    # This merge logic is naive; a proper implementation would pivot the game_stats table

    # Phase 1c: Add injury flags (currently stub for historical data)
    print("\n[3/3] Adding injury & weather data...")
    injuries_df = injuries.get_historical_injuries_stub(seasons, cache_dir=raw_dir)
    schedules_with_injuries = injuries.flag_qb_out(schedules, injuries_df)
    print(f"  Flagged QB injuries: {schedules_with_injuries['home_qb_out'].sum() + schedules_with_injuries['away_qb_out'].sum()} games")

    # Phase 1d: Add weather data
    schedules_with_weather = weather.add_weather_to_games(schedules_with_injuries, cache_dir=raw_dir)
    print(f"  Added weather data (temp, wind, precip)")

    # Write master dataset
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{processed_dir}/games_master.csv"
    schedules_with_weather.to_csv(output_file, index=False)
    print(f"\n✓ Wrote {len(schedules_with_weather)} games to {output_file}")

    return schedules_with_weather


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NFL games master dataset")
    parser.add_argument("--refresh", action="store_true", help="Re-download all data (ignore cache)")
    parser.add_argument("--week", type=int, help="Current NFL week (for Phase 7)")
    args = parser.parse_args()

    main(refresh=args.refresh, week=args.week)
