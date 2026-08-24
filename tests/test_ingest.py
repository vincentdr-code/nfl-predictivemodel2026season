"""
Tests for Phase 1 data ingestion.
"""
import pytest
import pandas as pd
import os
from pathlib import Path


def test_games_master_exists():
    """Verify games_master.csv was created and has expected structure."""
    games_master_path = "data/processed/games_master.csv"

    assert os.path.exists(games_master_path), f"{games_master_path} does not exist"

    df = pd.read_csv(games_master_path)
    assert len(df) > 0, "games_master.csv is empty"

    # Check expected columns
    expected_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_games_master_row_count():
    """Verify games_master.csv has approximately 6k games (32 teams * 17 weeks/season * 24 seasons)."""
    games_master_path = "data/processed/games_master.csv"
    df = pd.read_csv(games_master_path)

    # ~6,144 games expected for 2002-2025 (32 teams, 17 weeks/season, 24 seasons / 2 to avoid double-counting)
    expected_min = 5500
    expected_max = 7000

    assert expected_min <= len(df) <= expected_max, \
        f"Unexpected row count: {len(df)}. Expected ~6k."


def test_games_master_no_missing_scores():
    """Verify home_score and away_score have no NaNs."""
    games_master_path = "data/processed/games_master.csv"
    df = pd.read_csv(games_master_path)

    assert df['home_score'].notna().all(), "home_score has NaN values"
    assert df['away_score'].notna().all(), "away_score has NaN values"


def test_games_master_unique_game_ids():
    """Verify all game_ids are unique."""
    games_master_path = "data/processed/games_master.csv"
    df = pd.read_csv(games_master_path)

    assert df['game_id'].nunique() == len(df), \
        f"Duplicate game_ids found: {len(df) - df['game_id'].nunique()} duplicates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
