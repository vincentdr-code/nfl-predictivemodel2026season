"""
Ingest injury data from ESPN API.
"""
import os
import json
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime


def pull_current_injuries(cache_dir="data/raw/injuries"):
    """
    Pull current-week injury reports from ESPN.
    Used for live predictions; not applicable to historical backtest.

    Args:
        cache_dir: directory to cache JSON responses

    Returns:
        DataFrame with columns: team, player, position, status
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Warning: Failed to fetch current injuries: {e}")
        return pd.DataFrame(columns=['team', 'player', 'position', 'status'])

    # Parse ESPN injury data structure
    injuries = []
    for article in data.get('articles', []):
        story = article.get('story', '')
        team = article.get('team', {}).get('abbreviation', '')

        # Naive parse: look for patterns in story text
        # ESPN format often: "Player (Position, Status)" in the story
        if story and team:
            # This is a simplified parse; ESPN's structure is complex
            lines = story.split('\n')
            for line in lines:
                if '(' in line and ')' in line:
                    try:
                        player_part, detail_part = line.split('(')
                        player = player_part.strip()
                        detail = detail_part.rstrip(')').strip()
                        injuries.append({
                            'team': team,
                            'player': player,
                            'detail': detail,
                            'timestamp': datetime.now().isoformat()
                        })
                    except:
                        pass

    # Cache raw response
    cache_file = os.path.join(cache_dir, f"injuries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(cache_file, 'w') as f:
        json.dump(data, f)

    df = pd.DataFrame(injuries)
    return df if len(df) > 0 else pd.DataFrame(columns=['team', 'player', 'detail', 'timestamp'])


def get_historical_injuries_stub(seasons, cache_dir="data/raw/injuries"):
    """
    Placeholder for historical injury data.
    In production, this would pull archived injury reports (e.g., from a sports DB).
    For now, returns empty DataFrame to avoid breaking the pipeline.

    Args:
        seasons: list of season years
        cache_dir: cache directory

    Returns:
        DataFrame with columns: game_id, season, week, team, player, position, status
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Stub: no historical injury data source readily available
    # (ESPN's API is current-week only; historical would need a subscription service)
    print("Warning: Historical injury data not available. Using stub (empty DataFrame).")
    return pd.DataFrame(columns=[
        'game_id', 'season', 'week', 'team', 'player', 'position', 'status'
    ])


def flag_qb_out(schedule_df, injuries_df):
    """
    Add binary flag columns: home_qb_out, away_qb_out

    Args:
        schedule_df: DataFrame with game_id, season, week, home_team, away_team
        injuries_df: DataFrame with season, week, team, player, position, status

    Returns:
        schedule_df with added columns home_qb_out, away_qb_out (1/0)
    """
    result = schedule_df.copy()
    result['home_qb_out'] = 0
    result['away_qb_out'] = 0

    # Filter injuries to only QBs with status = 'Out' or 'IR'
    qb_outs = injuries_df[
        (injuries_df.get('position', '') == 'QB') &
        (injuries_df.get('status', '').isin(['Out', 'IR']))
    ]

    # Mark games where home QB is out
    for _, injury in qb_outs.iterrows():
        team = injury.get('team', '')
        season = injury.get('season')
        week = injury.get('week')

        home_mask = (result['home_team'] == team) & (result['season'] == season) & (result['week'] == week)
        away_mask = (result['away_team'] == team) & (result['season'] == season) & (result['week'] == week)

        result.loc[home_mask, 'home_qb_out'] = 1
        result.loc[away_mask, 'away_qb_out'] = 1

    return result


if __name__ == "__main__":
    # Quick test: try to pull current injuries
    injuries = pull_current_injuries()
    print(f"Current injuries: {len(injuries)} records")
    print(injuries.head())
