"""
Main entrypoint: orchestrate all data ingestion and write games_master.csv
+ game_team_stats.parquet.
"""
import argparse
import pandas as pd
import yaml
from pathlib import Path
from src.ingest import games


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main(config_file="config.yaml", refresh=False, week=None):
    config = load_config(config_file)

    seasons = config['data']['seasons']
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    print(f"Building dataset for seasons {seasons[0]}-{seasons[-1]}...")

    # 1. Schedules (already includes weather, QB names, rest, div flag)
    print("\n[1/3] Fetching schedules...")
    schedules = games.get_schedules(seasons, cache_dir=raw_dir, refresh=refresh)
    print(f"  {len(schedules):,} regular-season games")

    # 2. Play-by-play (for EPA / third downs / turnovers / success rate)
    print("\n[2/3] Fetching play-by-play...")
    pbp = games.get_pbp(seasons, cache_dir=raw_dir, refresh=refresh)
    print(f"  {len(pbp):,} plays loaded")

    # 3. Build team-game table (offense + defense stats per team per game)
    print("\n[3/3] Aggregating PBP to team-game level...")
    team_game = games.build_game_team_table(schedules, pbp)
    print(f"  {len(team_game):,} team-game rows")

    # Write outputs
    master_path = f"{processed_dir}/games_master.csv"
    schedules.to_csv(master_path, index=False)
    print(f"\n[OK] Wrote {len(schedules):,} games to {master_path}")

    team_path = f"{processed_dir}/team_game_stats.parquet"
    team_game.to_parquet(team_path)
    print(f"[OK] Wrote {len(team_game):,} team-game rows to {team_path}")

    return schedules, team_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NFL games master dataset")
    parser.add_argument("--refresh", action="store_true", help="Re-download all data")
    parser.add_argument("--week", type=int, help="Current NFL week (for Phase 7)")
    args = parser.parse_args()

    main(refresh=args.refresh, week=args.week)
