"""
Kickoff datetime helpers — used to freeze picks at game time.
"""
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def kickoff_utc(gameday, gametime) -> Optional[datetime]:
    """
    Parse (gameday, gametime) from nfl_data_py schedule to a UTC datetime.
    gameday: 'YYYY-MM-DD' string
    gametime: 'HH:MM' string in Eastern time (NFL default)
    Returns None if either is missing/malformed.
    """
    if not gameday or not gametime:
        return None
    try:
        gday = str(gameday)[:10]
        gtime = str(gametime)
        # gametime may include seconds; keep first 5 chars
        gtime = gtime[:5] if len(gtime) >= 5 else gtime
        dt_et = datetime.strptime(f"{gday} {gtime}", "%Y-%m-%d %H:%M")
        return dt_et.replace(tzinfo=ET).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def is_locked(gameday, gametime) -> bool:
    """True if kickoff has passed (or if kickoff is unknown, be conservative)."""
    ko = kickoff_utc(gameday, gametime)
    if ko is None:
        # Unknown kickoff → treat as unlocked (be forgiving)
        return False
    return datetime.now(timezone.utc) >= ko


def kickoff_display(gameday, gametime) -> str:
    """Pretty-print kickoff time in Eastern time. Cross-platform (no %-I)."""
    ko = kickoff_utc(gameday, gametime)
    if ko is None:
        return ""
    local = ko.astimezone(ET)
    hour_12 = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    return f"{local.strftime('%a %b')} {local.day} · {hour_12}:{local.minute:02d} {ampm} ET"
