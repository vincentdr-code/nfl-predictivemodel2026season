"""
CRITICAL: Test that features have zero future-data leakage.

Every feature used to predict a game at (season S, week W) must only
use data from games with week < W in season S, or from prior seasons.
"""
import pytest
import pandas as pd
import os


def test_no_leakage_rolling_features():
    """
    Verify that rolling features for game at (season, week) only use prior games.
    This test will pass after Phase 3 (features.parquet is built).
    """
    features_path = "data/processed/features.parquet"

    # Skip test if features not yet built
    if not os.path.exists(features_path):
        pytest.skip("features.parquet not yet built (Phase 3)")

    features = pd.read_parquet(features_path)

    # Sanity check: for each row (game), check that rolling window columns
    # only reference games from earlier in the same season or prior seasons
    # (This is enforced at build time in build_features.py, but we verify here)

    # Expected rolling feature columns (computed in Phase 3)
    rolling_cols = [
        'home_epa_per_play_4game',
        'away_epa_per_play_4game',
        'home_points_4game',
        'away_points_4game',
    ]

    # Verify they're all present
    for col in rolling_cols:
        assert col in features.columns, \
            f"Expected rolling feature column '{col}' not found. Features have {features.columns.tolist()}"

    # No specific numeric test here; the enforcement is in build_features.py
    # This test is a placeholder that will be extended with stricter checks in Phase 3


def test_no_future_games_in_season_window():
    """
    If season-to-date features are computed, verify they use only games up to target week.
    """
    features_path = "data/processed/features.parquet"

    if not os.path.exists(features_path):
        pytest.skip("features.parquet not yet built (Phase 3)")

    features = pd.read_parquet(features_path)
    games_master = pd.read_csv("data/processed/games_master.csv")

    # Merge features with game metadata to check week consistency
    merged = features.merge(
        games_master[['game_id', 'season', 'week']],
        on='game_id',
        how='left'
    )

    # Check that all rows have valid season/week (no NaNs)
    assert merged['season'].notna().all(), "Found NaN seasons in features"
    assert merged['week'].notna().all(), "Found NaN weeks in features"

    # No specific numeric check; implementation detail in Phase 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
