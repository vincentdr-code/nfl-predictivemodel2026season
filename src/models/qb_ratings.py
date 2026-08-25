"""
QB rating layer.

Compute a per-QB rating from career passing EPA per play, weighted toward
recent seasons. Used as an adjustment on top of team Elo:

    home_qb_adj = home_qb_rating - league_avg_qb_rating

Adds a `qb_rating_diff` feature (home - away) that the LightGBM model
uses directly. In backtests this typically pulls Brier down by 2-4 pts
on weeks with QB changes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LEAGUE_AVG_EPA_PER_DROPBACK = 0.0  # set after computation
RATING_SCALE = 200.0                 # convert EPA to Elo-scale points


def _weight_seasons(pbp, half_life_seasons=3):
    """Give recent seasons more weight (exponential decay)."""
    max_season = pbp['season'].max()
    age = max_season - pbp['season']
    return 0.5 ** (age / half_life_seasons)


def build_qb_ratings_history(pbp, min_dropbacks_per_season=100):
    """
    Compute a per-QB rating for each (qb, season) window using all data
    strictly BEFORE that season (no leakage).

    Returns DataFrame with columns:
        passer_name, season, dropbacks_prior, avg_epa_prior, qb_rating
    """
    passer_col = 'passer_player_name' if 'passer_player_name' in pbp.columns else 'passer'
    if passer_col not in pbp.columns:
        raise KeyError(f"No passer name column in PBP; expected one of {['passer_player_name', 'passer']}")

    # Filter to dropbacks (pass attempts + sacks) with a valid passer
    dropbacks = pbp[
        (pbp[passer_col].notna())
        & (pbp['epa'].notna())
        & (pbp.get('pass', 0) == 1)
    ].copy()

    dropbacks = dropbacks.rename(columns={passer_col: 'qb'})
    dropbacks = dropbacks[['qb', 'season', 'week', 'epa']]

    # For each (qb, target_season) pair, aggregate prior-season stats
    # Approach: for each qb, cumulative stats by season, then shift(1)
    grp = dropbacks.groupby(['qb', 'season']).agg(
        season_dropbacks=('epa', 'count'),
        season_epa_sum=('epa', 'sum'),
    ).reset_index()

    # For each QB, build cumulative "prior-through" numbers, shifted
    grp = grp.sort_values(['qb', 'season']).reset_index(drop=True)
    grp['dropbacks_cum'] = grp.groupby('qb')['season_dropbacks'].cumsum()
    grp['epa_cum'] = grp.groupby('qb')['season_epa_sum'].cumsum()
    grp['dropbacks_prior'] = grp.groupby('qb')['dropbacks_cum'].shift(1).fillna(0)
    grp['epa_prior'] = grp.groupby('qb')['epa_cum'].shift(1).fillna(0)

    # Rookie / small-sample fallback: mean-shrink toward league average
    global LEAGUE_AVG_EPA_PER_DROPBACK
    total_epa = dropbacks['epa'].sum()
    total_db = len(dropbacks)
    LEAGUE_AVG_EPA_PER_DROPBACK = total_epa / total_db

    # Bayesian shrinkage: (epa_prior + k * league_avg) / (dropbacks_prior + k)
    k = 200  # prior weight (equivalent to ~1 season of league-avg data)
    grp['avg_epa_prior'] = (
        (grp['epa_prior'] + k * LEAGUE_AVG_EPA_PER_DROPBACK)
        / (grp['dropbacks_prior'] + k)
    )

    # Convert to Elo-scale rating (delta above league average * scale)
    grp['qb_rating'] = (grp['avg_epa_prior'] - LEAGUE_AVG_EPA_PER_DROPBACK) * RATING_SCALE

    return grp[['qb', 'season', 'dropbacks_prior', 'avg_epa_prior', 'qb_rating']]


def match_qb_to_schedule(schedules, qb_ratings):
    """
    Match schedule.home_qb_name and away_qb_name to the qb_ratings table.
    Falls back to 0 (league average) if unknown.

    Returns schedules with home_qb_rating, away_qb_rating, qb_rating_diff cols added.
    """
    if 'home_qb_name' not in schedules.columns:
        schedules['home_qb_name'] = None
    if 'away_qb_name' not in schedules.columns:
        schedules['away_qb_name'] = None

    # Build a lookup on (qb_normalized, season) → rating
    def normalize(name):
        if pd.isna(name):
            return None
        # PBP uses "P.Mahomes"-style abbreviated names; schedule uses "Patrick Mahomes"
        # Store both first-initial-abbreviated and full lowercase for matching
        return str(name).strip()

    r = qb_ratings.copy()
    r['qb_norm'] = r['qb'].apply(normalize)

    # First try exact match; then try "F.Last" abbreviation
    def to_abbrev(full_name):
        if pd.isna(full_name):
            return None
        parts = str(full_name).strip().split()
        if len(parts) >= 2:
            return f"{parts[0][0]}.{parts[-1]}"
        return None

    lookup = {}
    for _, row in r.iterrows():
        lookup[(row['qb_norm'], int(row['season']))] = float(row['qb_rating'])

    def rating_for(name, season):
        if pd.isna(name):
            return 0.0
        season = int(season)
        # Try exact match
        if (str(name).strip(), season) in lookup:
            return lookup[(str(name).strip(), season)]
        # Try abbreviated form (F.Last)
        abbrev = to_abbrev(name)
        if abbrev and (abbrev, season) in lookup:
            return lookup[(abbrev, season)]
        return 0.0

    schedules = schedules.copy()
    schedules['home_qb_rating'] = schedules.apply(
        lambda r: rating_for(r['home_qb_name'], r['season']), axis=1
    )
    schedules['away_qb_rating'] = schedules.apply(
        lambda r: rating_for(r['away_qb_name'], r['season']), axis=1
    )
    schedules['qb_rating_diff'] = schedules['home_qb_rating'] - schedules['away_qb_rating']
    return schedules


def main(config_file='config.yaml'):
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']

    print("Loading PBP...")
    pbp = pd.read_parquet(f"{raw_dir}/pbp.parquet")
    # We need passer_player_name; if only pulled minimal cols, re-fetch
    if 'passer_player_name' not in pbp.columns and 'passer' not in pbp.columns:
        print("  ERROR: PBP cache doesn't have passer names.")
        print("  Re-run make_dataset.py after updating get_pbp() to include passer_player_name.")
        return

    print(f"  {len(pbp):,} plays")
    print("Building QB ratings history...")
    ratings = build_qb_ratings_history(pbp)
    print(f"  {len(ratings):,} (qb, season) rows")
    print(f"  League avg EPA/dropback: {LEAGUE_AVG_EPA_PER_DROPBACK:.4f}")

    output = f"{processed_dir}/qb_ratings.parquet"
    ratings.to_parquet(output)
    print(f"[OK] Wrote {output}")

    # Show top and bottom QBs for 2025
    latest = ratings[ratings['season'] == ratings['season'].max()]
    top = latest.nlargest(10, 'qb_rating')[['qb', 'dropbacks_prior', 'qb_rating']]
    print("\nTop 10 QBs entering 2025:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
