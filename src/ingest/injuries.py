"""
Ingest historical injury reports via nfl_data_py.

Available 2010+. Contains per-team per-week injury reports with columns:
    season, week, team, position, full_name, report_status
Statuses: 'Out', 'Doubtful', 'Questionable', 'Probable', None

We use these to build per-game flags:
    home_qb_out, away_qb_out
    home_key_players_out, away_key_players_out (count of RB/WR/TE/LT/DE Out)
"""
import os
from pathlib import Path

import pandas as pd
import nfl_data_py as nfl


KEY_POSITIONS = {'RB', 'WR', 'TE', 'LT', 'RT', 'LG', 'RG', 'C', 'DE', 'DT', 'LB', 'CB', 'S'}


def get_injuries(seasons, cache_dir="data/raw", refresh=False):
    """
    Pull per-week injury reports (2010+). Fetch season-by-season so a
    missing/future season 404 doesn't kill the whole pull.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, "injuries.parquet")

    if os.path.exists(cache_file) and not refresh:
        return pd.read_parquet(cache_file)

    print(f"Pulling injuries for {len(seasons)} seasons (per-season, 404-tolerant)...")
    supported = [s for s in seasons if s >= 2010]
    if not supported:
        return pd.DataFrame(columns=['season', 'week', 'team', 'position', 'full_name', 'report_status'])

    keep = ['season', 'week', 'team', 'position', 'full_name', 'report_status', 'report_primary_injury']
    frames = []
    for s in supported:
        try:
            inj_s = nfl.import_injuries([s])
            if len(inj_s):
                inj_s = inj_s[[c for c in keep if c in inj_s.columns]].copy()
                inj_s = inj_s[inj_s['week'].notna()]
                if len(inj_s):
                    inj_s['week'] = inj_s['week'].astype(int)
                    inj_s['season'] = inj_s['season'].astype(int)
                    frames.append(inj_s)
                    print(f"  season {s}: {len(inj_s):,} records")
        except Exception as e:
            msg = str(e)
            if '404' in msg or 'Not Found' in msg or (
                isinstance(e, NameError) and "'Error' is not defined" in msg
            ):
                print(f"  season {s}: not published yet, skipping")
            else:
                print(f"  season {s}: {type(e).__name__}: {e}")

    if not frames:
        return pd.DataFrame(columns=keep)

    inj = pd.concat(frames, ignore_index=True)
    inj.to_parquet(cache_file)
    print(f"  Cached {len(inj):,} injury rows total to {cache_file}")
    return inj


def build_injury_flags(schedules, injuries):
    """
    For each game, compute:
        home_qb_out (1/0): starting QB has report_status in {'Out'}
        away_qb_out (1/0)
        home_qb_questionable (1/0)  — QB is Questionable/Doubtful
        away_qb_questionable (1/0)
        home_key_out (int): count of key-position players marked Out
        away_key_out (int)

    Returns schedules with those columns added.
    """
    result = schedules.copy()
    result['home_qb_out'] = 0
    result['away_qb_out'] = 0
    result['home_qb_questionable'] = 0
    result['away_qb_questionable'] = 0
    result['home_key_out'] = 0
    result['away_key_out'] = 0

    if len(injuries) == 0:
        return result

    # QB outs per (season, week, team)
    qb_out = injuries[(injuries['position'] == 'QB')
                      & (injuries['report_status'].isin(['Out', 'IR', 'Injured Reserve']))]
    qb_out_set = set(zip(qb_out['season'].astype(int), qb_out['week'].astype(int), qb_out['team']))

    qb_q = injuries[(injuries['position'] == 'QB')
                    & (injuries['report_status'].isin(['Questionable', 'Doubtful']))]
    qb_q_set = set(zip(qb_q['season'].astype(int), qb_q['week'].astype(int), qb_q['team']))

    # Key players out counts
    key_out = injuries[injuries['position'].isin(KEY_POSITIONS)
                       & injuries['report_status'].isin(['Out', 'IR', 'Injured Reserve'])]
    key_counts = key_out.groupby(['season', 'week', 'team']).size().to_dict()

    for idx, row in result.iterrows():
        s = int(row['season']); w = int(row['week'])
        h = row['home_team']; a = row['away_team']
        if (s, w, h) in qb_out_set: result.loc[idx, 'home_qb_out'] = 1
        if (s, w, a) in qb_out_set: result.loc[idx, 'away_qb_out'] = 1
        if (s, w, h) in qb_q_set: result.loc[idx, 'home_qb_questionable'] = 1
        if (s, w, a) in qb_q_set: result.loc[idx, 'away_qb_questionable'] = 1
        result.loc[idx, 'home_key_out'] = int(key_counts.get((s, w, h), 0))
        result.loc[idx, 'away_key_out'] = int(key_counts.get((s, w, a), 0))

    return result


if __name__ == "__main__":
    seasons = list(range(2010, 2026))
    inj = get_injuries(seasons)
    print(f"\nTotal injury records: {len(inj):,}")
    print(f"Unique seasons: {sorted(inj['season'].unique())}")
    print(f"\nBy status:")
    print(inj['report_status'].value_counts())
