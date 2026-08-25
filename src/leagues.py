"""
League management for Pick 'Em: invite codes, membership, per-league scoring.

Storage uses the same _exec / _query helpers as gamification.py, so it
inherits Turso-or-local automatically.
"""
import random
import string
from datetime import datetime
from typing import Optional

import pandas as pd

from src.gamification import _exec, _query, _ensure_schema


LEAGUE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS leagues (
        league_id   TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        tiebreaker  TEXT NOT NULL DEFAULT 'confidence',
        season      INTEGER NOT NULL,
        created_at  TEXT NOT NULL,
        created_by  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS league_members (
        league_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        joined_at   TEXT NOT NULL,
        PRIMARY KEY (league_id, user_id)
    )""",
    """CREATE TABLE IF NOT EXISTS league_picks (
        league_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        game_id     TEXT NOT NULL,
        season      INTEGER NOT NULL,
        week        INTEGER NOT NULL,
        pick_team   TEXT NOT NULL,
        confidence  INTEGER,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (league_id, user_id, game_id)
    )""",
]


_league_schema_initialized = False


def _ensure_league_schema():
    global _league_schema_initialized
    if _league_schema_initialized:
        return
    _ensure_schema()  # base schema first (achievements, picks)
    for sql in LEAGUE_SCHEMA:
        _exec(sql)
    _league_schema_initialized = True


# ---------------- League CRUD ----------------

def _generate_code(length: int = 6) -> str:
    """Generate a random invite code like NFL-8K3M."""
    # Avoid ambiguous chars: 0/O, 1/I/L
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(length))


def create_league(name: str, created_by: str, season: int,
                  tiebreaker: str = "confidence") -> str:
    """Create a new league and return its invite code."""
    _ensure_league_schema()
    # Retry generation up to 20 times if collision
    for _ in range(20):
        code = _generate_code()
        try:
            _exec(
                """INSERT INTO leagues (league_id, name, tiebreaker, season, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (code, name.strip(), tiebreaker, int(season),
                 datetime.utcnow().isoformat(), created_by.strip())
            )
            # Creator auto-joins
            _exec(
                """INSERT INTO league_members (league_id, user_id, joined_at)
                   VALUES (?, ?, ?)""",
                (code, created_by.strip(), datetime.utcnow().isoformat())
            )
            return code
        except Exception as e:
            if "UNIQUE" in str(e) or "PRIMARY KEY" in str(e):
                continue
            raise
    raise RuntimeError("Could not generate a unique league code after 20 tries")


def join_league(league_id: str, user_id: str) -> bool:
    """Join an existing league. Returns True if joined, False if league doesn't exist."""
    _ensure_league_schema()
    league = get_league(league_id)
    if league is None:
        return False
    _exec(
        """INSERT OR IGNORE INTO league_members (league_id, user_id, joined_at)
           VALUES (?, ?, ?)""",
        (league_id, user_id.strip(), datetime.utcnow().isoformat())
    )
    return True


def get_league(league_id: str) -> Optional[dict]:
    """Fetch a league by code. Returns None if not found."""
    _ensure_league_schema()
    df = _query("SELECT * FROM leagues WHERE league_id=?", (league_id.strip().upper(),))
    if len(df) == 0:
        return None
    return df.iloc[0].to_dict()


def get_user_leagues(user_id: str) -> pd.DataFrame:
    """Return the leagues a user has joined."""
    _ensure_league_schema()
    return _query(
        """SELECT l.league_id, l.name, l.season, l.tiebreaker, l.created_at, l.created_by,
                  (SELECT COUNT(*) FROM league_members lm WHERE lm.league_id = l.league_id) AS member_count
           FROM leagues l
           JOIN league_members m ON m.league_id = l.league_id
           WHERE m.user_id = ?
           ORDER BY l.created_at DESC""",
        (user_id.strip(),)
    )


def get_league_members(league_id: str) -> pd.DataFrame:
    """List members of a league."""
    _ensure_league_schema()
    return _query(
        "SELECT user_id, joined_at FROM league_members WHERE league_id=? ORDER BY joined_at",
        (league_id.strip().upper(),)
    )


# ---------------- Picks (league-scoped) ----------------

def save_pick(league_id: str, user_id: str, game_id: str, season: int, week: int,
              pick_team: str, confidence: Optional[int] = None):
    """Insert or replace a league pick."""
    _ensure_league_schema()
    _exec(
        """INSERT INTO league_picks
             (league_id, user_id, game_id, season, week, pick_team, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (league_id, user_id, game_id) DO UPDATE SET
             pick_team = excluded.pick_team,
             confidence = excluded.confidence,
             created_at = excluded.created_at""",
        (league_id, user_id.strip(), game_id, int(season), int(week),
         pick_team, confidence, datetime.utcnow().isoformat())
    )


def get_picks(league_id: str, user_id: Optional[str] = None,
              season: Optional[int] = None, week: Optional[int] = None) -> pd.DataFrame:
    """Fetch picks for a league; optionally filter by user, season, and/or week."""
    _ensure_league_schema()
    q = "SELECT * FROM league_picks WHERE league_id=?"
    args = [league_id]
    if user_id is not None:
        q += " AND user_id=?"; args.append(user_id.strip())
    if season is not None:
        q += " AND season=?"; args.append(int(season))
    if week is not None:
        q += " AND week=?"; args.append(int(week))
    return _query(q, tuple(args))


def save_picks_batch(league_id: str, user_id: str, season: int, week: int,
                     picks_dict: dict):
    """
    picks_dict: {game_id: {'pick_team': ..., 'confidence': ...}}
    Writes all in one call for efficiency.
    """
    for game_id, entry in picks_dict.items():
        save_pick(league_id, user_id, game_id, season, week,
                  entry['pick_team'], entry.get('confidence'))


# ---------------- Scoring ----------------

def score_league_week(league_id: str, season: int, week: int,
                      model_preds: pd.DataFrame, schedules: pd.DataFrame,
                      tiebreaker: str = "confidence") -> pd.DataFrame:
    """
    Score a league for a specific week. Returns per-user DataFrame with:
        user_id, correct, total, confidence_pts, points_possible, beat_model
    """
    played = schedules[(schedules['season'] == season) & (schedules['week'] == week)
                       & schedules['home_score'].notna()].copy()
    if len(played) == 0:
        return pd.DataFrame()
    played['actual_winner'] = played.apply(
        lambda r: r['home_team'] if r['home_score'] > r['away_score'] else r['away_team'], axis=1
    )

    picks = get_picks(league_id, season=season, week=week)
    if len(picks) == 0:
        return pd.DataFrame()

    merged = picks.merge(played[['game_id', 'actual_winner']], on='game_id', how='inner')
    merged['correct'] = (merged['pick_team'] == merged['actual_winner']).astype(int)
    merged['pts'] = merged['correct'] * merged['confidence'].fillna(1).astype(int)

    scored = merged.groupby('user_id').agg(
        correct=('correct', 'sum'),
        total=('game_id', 'count'),
        confidence_pts=('pts', 'sum'),
        points_possible=('confidence', lambda s: int(s.fillna(1).sum())),
    ).reset_index()

    # Model score for comparison
    mp = model_preds[model_preds['game_id'].isin(played['game_id'])].merge(
        played[['game_id', 'actual_winner']], on='game_id')
    if len(mp):
        mp['model_pick'] = mp.apply(
            lambda r: r['home_team'] if r['pred_home_win_prob'] > 0.5 else r['away_team'], axis=1)
        mp['model_correct'] = (mp['model_pick'] == mp['actual_winner']).astype(int)
        # Model gets "average confidence" per game as its notional score
        n = len(mp)
        model_pts = int(mp['model_correct'].sum() * (n + 1) / 2)  # avg conf value = (n+1)/2
        scored['beat_model'] = scored['confidence_pts'] > model_pts
    else:
        scored['beat_model'] = False

    return scored.sort_values('confidence_pts', ascending=False).reset_index(drop=True)


def score_league_season(league_id: str, season: int,
                        model_preds: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Season-long leaderboard for a league. Returns per-user DataFrame with:
        user_id, correct, total, pct, confidence_pts, weeks_scored,
        beat_model_wks, best_week_pts
    """
    played_weeks = sorted(schedules[(schedules['season'] == season)
                                     & schedules['home_score'].notna()]['week'].unique())
    if not played_weeks:
        return pd.DataFrame()

    all_weekly = []
    for wk in played_weeks:
        wk_scored = score_league_week(league_id, season, int(wk), model_preds, schedules)
        if len(wk_scored):
            wk_scored['week'] = int(wk)
            all_weekly.append(wk_scored)

    if not all_weekly:
        return pd.DataFrame()

    weekly = pd.concat(all_weekly, ignore_index=True)
    agg = weekly.groupby('user_id').agg(
        correct=('correct', 'sum'),
        total=('total', 'sum'),
        confidence_pts=('confidence_pts', 'sum'),
        weeks_scored=('week', 'count'),
        beat_model_wks=('beat_model', 'sum'),
        best_week_pts=('confidence_pts', 'max'),
    ).reset_index()
    agg['pct'] = (agg['correct'] / agg['total'] * 100).round(1)
    return agg.sort_values('confidence_pts', ascending=False).reset_index(drop=True)
