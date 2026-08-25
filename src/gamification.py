"""
Pick 'Em + achievements storage backed by SQLite.

Single-user (per-browser-session) mode: uses a browser-cookie user_id and
persists to a local SQLite file. Multi-user is a future upgrade.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd


DB_PATH = Path("data/processed/gamification.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    user_id     TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    pick_team   TEXT NOT NULL,      -- team code the user picked to win
    pick_ats    TEXT,                -- team code covered ATS (optional)
    pick_ou     TEXT,                -- 'over' / 'under' (optional)
    confidence  INTEGER,             -- 1-16 for confidence pool (optional)
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, game_id)
);

CREATE TABLE IF NOT EXISTS achievements (
    user_id     TEXT NOT NULL,
    achievement TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER,
    earned_at   TEXT NOT NULL,
    detail      TEXT,
    PRIMARY KEY (user_id, achievement, season, week)
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_pick(user_id: str, game_id: str, season: int, week: int,
              pick_team: str, pick_ats: Optional[str] = None,
              pick_ou: Optional[str] = None, confidence: Optional[int] = None):
    """Insert or replace a pick."""
    from datetime import datetime
    with _conn() as conn:
        conn.execute(
            """INSERT INTO picks (user_id, game_id, season, week, pick_team,
                                   pick_ats, pick_ou, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (user_id, game_id) DO UPDATE SET
                   pick_team = excluded.pick_team,
                   pick_ats  = excluded.pick_ats,
                   pick_ou   = excluded.pick_ou,
                   confidence = excluded.confidence,
                   created_at = excluded.created_at""",
            (user_id, game_id, season, week, pick_team,
             pick_ats, pick_ou, confidence, datetime.utcnow().isoformat())
        )


def get_picks(user_id: str, season: int, week: Optional[int] = None) -> pd.DataFrame:
    """Load picks for a user; optionally filter to one week."""
    with _conn() as conn:
        if week is not None:
            df = pd.read_sql_query(
                "SELECT * FROM picks WHERE user_id=? AND season=? AND week=?",
                conn, params=(user_id, season, week))
        else:
            df = pd.read_sql_query(
                "SELECT * FROM picks WHERE user_id=? AND season=?",
                conn, params=(user_id, season))
    return df


def score_week(user_id: str, season: int, week: int,
               model_preds: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    """
    Score a week's picks against actual outcomes and against the model.

    Args:
        user_id
        season, week
        model_preds: DataFrame with game_id, home_team, away_team, pred_home_win_prob
        schedules:   DataFrame with game_id, home_team, away_team, home_score, away_score

    Returns dict with keys:
        user_correct, user_total, user_pct,
        model_correct, model_total, model_pct,
        beat_model (bool), perfect_week (bool)
    """
    picks = get_picks(user_id, season, week)
    played = schedules[(schedules['season'] == season) & (schedules['week'] == week)
                       & schedules['home_score'].notna()].copy()
    played['actual_winner'] = played.apply(
        lambda r: r['home_team'] if r['home_score'] > r['away_score'] else r['away_team'], axis=1
    )
    played = played[['game_id', 'home_team', 'away_team', 'actual_winner']]

    if len(played) == 0:
        return {'games_played': 0}

    # User picks joined with actuals
    merged = played.merge(picks[['game_id', 'pick_team']], on='game_id', how='left')
    merged['user_correct'] = (merged['pick_team'] == merged['actual_winner']).astype(int)
    user_correct = int(merged['user_correct'].sum())
    user_total = int(merged['pick_team'].notna().sum())

    # Model picks
    mp = model_preds[model_preds['game_id'].isin(played['game_id'])]
    mp = mp.merge(played[['game_id', 'actual_winner']], on='game_id', how='inner')
    mp['model_pick'] = mp.apply(
        lambda r: r['home_team'] if r['pred_home_win_prob'] > 0.5 else r['away_team'], axis=1
    )
    mp['model_correct'] = (mp['model_pick'] == mp['actual_winner']).astype(int)
    model_correct = int(mp['model_correct'].sum())
    model_total = len(mp)

    return {
        'games_played': int(len(played)),
        'user_correct': user_correct,
        'user_total': user_total,
        'user_pct': (user_correct / user_total * 100) if user_total else 0,
        'model_correct': model_correct,
        'model_total': model_total,
        'model_pct': (model_correct / model_total * 100) if model_total else 0,
        'beat_model': user_correct > model_correct,
        'perfect_week': user_correct == user_total and user_total > 0,
    }


def score_season(user_id: str, season: int,
                 model_preds: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per week, user vs model records.
    """
    played_weeks = sorted(schedules[(schedules['season'] == season)
                                     & schedules['home_score'].notna()]['week'].unique())
    rows = []
    for wk in played_weeks:
        r = score_week(user_id, season, wk, model_preds, schedules)
        r['week'] = int(wk)
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------- Achievements ----------------

ACHIEVEMENTS = {
    'first_pick': {
        'name': 'First Down',
        'desc': 'Made your first pick',
    },
    'perfect_week': {
        'name': 'Perfect Week',
        'desc': 'Correctly picked every game in a week',
    },
    'beat_model': {
        'name': 'Beat The Model',
        'desc': 'Scored higher than the model in a week',
    },
    'upset_caller': {
        'name': 'Upset Caller',
        'desc': 'Correctly picked an underdog under 35%',
    },
    'streak_5': {
        'name': 'On A Roll',
        'desc': 'Won 5 picks in a row',
    },
    'streak_10': {
        'name': 'Unstoppable',
        'desc': 'Won 10 picks in a row',
    },
    'season_beat_model': {
        'name': 'Coach Of The Year',
        'desc': 'Finish a season with a higher win % than the model',
    },
}


def grant_achievement(user_id: str, achievement: str, season: int,
                      week: Optional[int] = None, detail: str = ""):
    """Grant an achievement if not already earned."""
    from datetime import datetime
    with _conn() as conn:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO achievements
                   (user_id, achievement, season, week, earned_at, detail)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, achievement, season, week or 0,
                 datetime.utcnow().isoformat(), detail)
            )
        except sqlite3.Error:
            pass


def get_achievements(user_id: str) -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM achievements WHERE user_id=? ORDER BY earned_at DESC",
            conn, params=(user_id,))


def check_and_grant_all(user_id: str, season: int, week: int,
                        model_preds: pd.DataFrame, schedules: pd.DataFrame):
    """After a week completes, check every achievement condition."""
    score = score_week(user_id, season, week, model_preds, schedules)
    if score.get('games_played', 0) == 0:
        return

    if score.get('perfect_week'):
        grant_achievement(user_id, 'perfect_week', season, week,
                          detail=f"Week {week}: {score['user_correct']}/{score['user_total']}")

    if score.get('beat_model'):
        grant_achievement(user_id, 'beat_model', season, week,
                          detail=f"Week {week}: user {score['user_correct']} vs model {score['model_correct']}")

    # Upset caller: any pick where model gave the pick team < 35% and user was right
    picks = get_picks(user_id, season, week)
    played = schedules[(schedules['season'] == season) & (schedules['week'] == week)
                       & schedules['home_score'].notna()].copy()
    for _, p in picks.iterrows():
        game = played[played['game_id'] == p['game_id']]
        if len(game) == 0:
            continue
        game = game.iloc[0]
        winner = game['home_team'] if game['home_score'] > game['away_score'] else game['away_team']
        if p['pick_team'] != winner:
            continue
        mp_row = model_preds[model_preds['game_id'] == p['game_id']]
        if len(mp_row) == 0:
            continue
        mp_row = mp_row.iloc[0]
        model_prob_for_pick = (mp_row['pred_home_win_prob']
                               if p['pick_team'] == game['home_team']
                               else 1 - mp_row['pred_home_win_prob'])
        if model_prob_for_pick < 0.35:
            grant_achievement(user_id, 'upset_caller', season, week,
                              detail=f"{p['pick_team']} ({model_prob_for_pick*100:.0f}% model odds)")
