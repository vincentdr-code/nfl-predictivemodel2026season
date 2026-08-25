"""
Pick 'Em + achievements storage.

Persistence:
- If TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set (typical for
  Streamlit Cloud), uses Turso's HTTP API — pure-Python via `requests`,
  no compiled deps.
- Otherwise falls back to a local SQLite file for dev.

Turso is libSQL (SQLite fork) so the schema and SQL are identical.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


DB_PATH = Path("data/processed/gamification.db")

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)

# Convert libsql://foo.turso.io -> https://foo.turso.io for the HTTP API
if USE_TURSO and TURSO_URL.startswith("libsql://"):
    TURSO_HTTP_URL = "https://" + TURSO_URL[len("libsql://"):]
else:
    TURSO_HTTP_URL = TURSO_URL


SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS picks (
        user_id     TEXT NOT NULL,
        game_id     TEXT NOT NULL,
        season      INTEGER NOT NULL,
        week        INTEGER NOT NULL,
        pick_team   TEXT NOT NULL,
        pick_ats    TEXT,
        pick_ou     TEXT,
        confidence  INTEGER,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (user_id, game_id)
    )""",
    """CREATE TABLE IF NOT EXISTS achievements (
        user_id     TEXT NOT NULL,
        achievement TEXT NOT NULL,
        season      INTEGER NOT NULL,
        week        INTEGER,
        earned_at   TEXT NOT NULL,
        detail      TEXT,
        PRIMARY KEY (user_id, achievement, season, week)
    )""",
]


# ---------------- Turso HTTP client ----------------

def _turso_arg(value):
    """Convert a Python value to Turso's arg format."""
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    return {"type": "text", "value": str(value)}


def _turso_row_value(v):
    """Convert a Turso response value back to Python."""
    if v is None:
        return None
    if isinstance(v, dict):
        t = v.get("type")
        val = v.get("value")
        if t == "null":
            return None
        if t == "integer":
            return int(val) if val is not None else None
        if t == "float":
            return float(val) if val is not None else None
        return val  # text/blob returned as-is
    return v


def _turso_execute(statements):
    """
    Execute a batch of statements against Turso via the HTTP pipeline API.
    `statements` is a list of (sql, args) tuples.
    Returns list of result dicts, one per statement, with keys 'cols', 'rows'.
    """
    requests_body = []
    for sql, args in statements:
        stmt = {"sql": sql}
        if args:
            stmt["args"] = [_turso_arg(a) for a in args]
        requests_body.append({"type": "execute", "stmt": stmt})
    requests_body.append({"type": "close"})

    resp = requests.post(
        f"{TURSO_HTTP_URL}/v2/pipeline",
        headers={"Authorization": f"Bearer {TURSO_TOKEN}"},
        json={"requests": requests_body},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    out = []
    for r in data.get("results", []):
        if r.get("type") != "ok":
            err = r.get("error", {})
            raise RuntimeError(f"Turso error: {err.get('message', err)}")
        response = r.get("response", {})
        if response.get("type") == "execute":
            result = response.get("result", {})
            cols = [c["name"] for c in result.get("cols", [])]
            raw_rows = result.get("rows", [])
            rows = [[_turso_row_value(v) for v in row] for row in raw_rows]
            out.append({"cols": cols, "rows": rows})
        else:
            out.append({"cols": [], "rows": []})
    return out


# ---------------- Storage-agnostic executor ----------------

_schema_initialized = False


def _ensure_schema():
    """Run CREATE TABLE IF NOT EXISTS once per process."""
    global _schema_initialized
    if _schema_initialized:
        return
    if USE_TURSO:
        _turso_execute([(sql, ()) for sql in SCHEMA_STATEMENTS])
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            for sql in SCHEMA_STATEMENTS:
                conn.execute(sql)
            conn.commit()
    _schema_initialized = True


def _exec(sql, params=()):
    """Execute a write (INSERT/UPDATE/DELETE)."""
    _ensure_schema()
    if USE_TURSO:
        _turso_execute([(sql, params)])
    else:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(sql, params)
            conn.commit()


def _query(sql, params=()) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame."""
    _ensure_schema()
    if USE_TURSO:
        result = _turso_execute([(sql, params)])[0]
        return pd.DataFrame(result["rows"], columns=result["cols"])
    else:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(sql, conn, params=params)


# ---------------- Public API ----------------

def save_pick(user_id: str, game_id: str, season: int, week: int,
              pick_team: str, pick_ats: Optional[str] = None,
              pick_ou: Optional[str] = None, confidence: Optional[int] = None):
    _exec(
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
    if week is not None:
        return _query(
            "SELECT * FROM picks WHERE user_id=? AND season=? AND week=?",
            (user_id, season, week))
    return _query(
        "SELECT * FROM picks WHERE user_id=? AND season=?",
        (user_id, season))


def score_week(user_id: str, season: int, week: int,
               model_preds: pd.DataFrame, schedules: pd.DataFrame) -> dict:
    picks = get_picks(user_id, season, week)
    played = schedules[(schedules['season'] == season) & (schedules['week'] == week)
                       & schedules['home_score'].notna()].copy()
    if len(played) == 0:
        return {'games_played': 0}
    played['actual_winner'] = played.apply(
        lambda r: r['home_team'] if r['home_score'] > r['away_score'] else r['away_team'], axis=1
    )
    played = played[['game_id', 'home_team', 'away_team', 'actual_winner']]

    merged = played.merge(picks[['game_id', 'pick_team']], on='game_id', how='left')
    merged['user_correct'] = (merged['pick_team'] == merged['actual_winner']).astype(int)
    user_correct = int(merged['user_correct'].sum())
    user_total = int(merged['pick_team'].notna().sum())

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
    'first_pick':         {'name': 'First Down',        'desc': 'Made your first pick'},
    'perfect_week':       {'name': 'Perfect Week',      'desc': 'Correctly picked every game in a week'},
    'beat_model':         {'name': 'Beat The Model',    'desc': 'Scored higher than the model in a week'},
    'upset_caller':       {'name': 'Upset Caller',      'desc': 'Correctly picked an underdog under 35%'},
    'streak_5':           {'name': 'On A Roll',         'desc': 'Won 5 picks in a row'},
    'streak_10':          {'name': 'Unstoppable',       'desc': 'Won 10 picks in a row'},
    'season_beat_model':  {'name': 'Coach Of The Year', 'desc': 'Finish a season with a higher win % than the model'},
}


def grant_achievement(user_id: str, achievement: str, season: int,
                      week: Optional[int] = None, detail: str = ""):
    _exec(
        """INSERT OR IGNORE INTO achievements
           (user_id, achievement, season, week, earned_at, detail)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, achievement, season, week or 0,
         datetime.utcnow().isoformat(), detail)
    )


def get_achievements(user_id: str) -> pd.DataFrame:
    return _query(
        "SELECT * FROM achievements WHERE user_id=? ORDER BY earned_at DESC",
        (user_id,))


def check_and_grant_all(user_id: str, season: int, week: int,
                        model_preds: pd.DataFrame, schedules: pd.DataFrame):
    score = score_week(user_id, season, week, model_preds, schedules)
    if score.get('games_played', 0) == 0:
        return

    if score.get('perfect_week'):
        grant_achievement(user_id, 'perfect_week', season, week,
                          detail=f"Week {week}: {score['user_correct']}/{score['user_total']}")
    if score.get('beat_model'):
        grant_achievement(user_id, 'beat_model', season, week,
                          detail=f"Week {week}: user {score['user_correct']} vs model {score['model_correct']}")

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
        model_prob = (mp_row['pred_home_win_prob']
                      if p['pick_team'] == game['home_team']
                      else 1 - mp_row['pred_home_win_prob'])
        if model_prob < 0.35:
            grant_achievement(user_id, 'upset_caller', season, week,
                              detail=f"{p['pick_team']} ({model_prob*100:.0f}% model odds)")
