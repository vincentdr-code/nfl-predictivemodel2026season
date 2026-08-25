"""
NFL Predictor Dashboard — v2.

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import yaml
import altair as alt

# Promote Streamlit secrets to env vars BEFORE importing gamification so
# it picks up the Turso credentials on startup.
try:
    if 'TURSO_DATABASE_URL' in st.secrets:
        os.environ['TURSO_DATABASE_URL'] = str(st.secrets['TURSO_DATABASE_URL'])
    if 'TURSO_AUTH_TOKEN' in st.secrets:
        os.environ['TURSO_AUTH_TOKEN'] = str(st.secrets['TURSO_AUTH_TOKEN'])
except (FileNotFoundError, KeyError, AttributeError):
    pass  # no secrets configured; gamification falls back to local SQLite

from src.teams import TEAMS, logo_url, team_name, short_name, primary_color
from src import gamification as game
from src import leagues
from src.kickoff import kickoff_utc, is_locked, kickoff_display
from src.simulate.playoff_bracket import compute_bracket_probs


st.set_page_config(page_title="NFL Predictor · 2026", layout="wide", initial_sidebar_state="collapsed")


# ------------------------------ styling ------------------------------

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800;900&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #0A0A0B; --surface: #111114; --surface-2: #16171C;
        --border: #1F2028; --border-2: #2A2C36;
        --text: #F2F2F2; --text-2: #A1A1A6; --text-3: #6E6E76;
        --accent: #E11D48; --accent-dim: #7A0F27;
        --success: #10B981; --warn: #F59E0B; --info: #3B82F6; --gold: #EAB308;
    }
    html, body, [class*="css"], .stApp, .main {
        font-family: 'Geist', -apple-system, sans-serif;
        color: var(--text); background: var(--bg);
        font-feature-settings: "ss01", "ss02";
    }
    .main .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* Masthead */
    .masthead { border-bottom: 1px solid var(--border); padding-bottom: 1.25rem;
                margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: flex-end; }
    .masthead-title { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }
    .masthead-title .accent { color: var(--accent); }
    .masthead-meta { text-align: right; font-family: 'Geist Mono', monospace;
                     font-size: 0.68rem; color: var(--text-3);
                     text-transform: uppercase; letter-spacing: 0.1em; line-height: 1.5; }

    /* Profile strip */
    .profile-strip {
        display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
        padding: 0.75rem 0; margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
        font-family: 'Geist Mono', monospace; font-size: 0.72rem;
    }
    .profile-strip .who {
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em;
        margin-right: 0.5rem;
    }
    .profile-strip .name { color: var(--text); font-weight: 600; letter-spacing: 0; }
    .profile-strip .record { color: var(--text-2); margin-left: auto; }
    .profile-strip .record .win { color: var(--success); font-weight: 600; }
    .profile-strip .record .loss { color: var(--accent); font-weight: 600; }
    .badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 3px 8px; border-radius: 12px;
        background: var(--surface-2); border: 1px solid var(--border-2);
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
        color: var(--gold);
    }
    .badge svg { width: 10px; height: 10px; }

    /* Eyebrow */
    .eyebrow { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.18em;
               color: var(--text-3); font-weight: 500; margin: 1.75rem 0 0.75rem 0;
               display: flex; align-items: baseline; justify-content: space-between; }
    .eyebrow .count { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2); }

    /* Widgets */
    .stSelectbox label, .stTextInput label, .stRadio label {
        color: var(--text-3); font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stTextInput input {
        background: var(--surface); border-color: var(--border); border-radius: 4px;
    }
    .stTextInput input { color: var(--text); font-family: 'Geist Mono', monospace; }

    /* Buttons */
    .stButton button {
        background: var(--accent); color: white; border: none;
        border-radius: 4px; padding: 0.5rem 1.25rem;
        font-family: 'Geist', sans-serif; font-weight: 600; font-size: 0.85rem;
        letter-spacing: -0.01em;
        transition: transform 0.1s;
    }
    .stButton button:hover { background: var(--accent-dim); }
    .stButton button:active { transform: translateY(1px); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border: none;
        border-bottom: 2px solid transparent; border-radius: 0;
        padding: 0.75rem 1.25rem; font-weight: 500;
        color: var(--text-3); letter-spacing: -0.01em;
    }
    .stTabs [aria-selected="true"] { color: var(--text); border-bottom-color: var(--accent); }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

    /* Game row */
    .game-row { display: grid; grid-template-columns: 1fr auto 1fr;
                align-items: center; gap: 1.5rem;
                padding: 1.25rem 0; border-bottom: 1px solid var(--border); }
    .game-row.lock { background: linear-gradient(90deg, rgba(16,185,129,0.03) 0%, transparent 40%); }
    .game-row.upset { background: linear-gradient(90deg, rgba(245,158,11,0.04) 0%, transparent 40%); }
    .game-row.user-picked-correct {
        background: linear-gradient(90deg, rgba(16,185,129,0.08) 0%, transparent 60%);
        border-left: 3px solid var(--success); padding-left: calc(0 - 3px);
    }
    .game-row.user-picked-wrong {
        background: linear-gradient(90deg, rgba(225,29,72,0.06) 0%, transparent 60%);
        border-left: 3px solid var(--accent); padding-left: calc(0 - 3px);
    }
    .team-side { display: flex; align-items: center; gap: 0.9rem; min-width: 0; }
    .team-side.away { justify-content: flex-end; text-align: right; }
    .team-mark img { width: 44px; height: 44px; object-fit: contain;
                     filter: drop-shadow(0 0 12px rgba(0,0,0,0.3)); }
    .team-info { min-width: 0; }
    .team-info .city { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
                       color: var(--text-3); font-weight: 500; line-height: 1; margin-bottom: 0.2rem; }
    .team-info .name { font-size: 1.1rem; font-weight: 700; line-height: 1;
                       letter-spacing: -0.02em; color: var(--text); }
    .team-info .record { font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                         color: var(--text-3); margin-top: 0.2rem; }
    .center-scores { display: flex; align-items: baseline; gap: 0.75rem;
                     font-family: 'Geist Mono', monospace; font-weight: 600; }
    .center-scores .away, .center-scores .home {
        font-size: 1.75rem; letter-spacing: -0.02em; min-width: 2.4ch; text-align: center;
    }
    .center-scores .sep { color: var(--text-3); font-size: 1.1rem; font-weight: 400; }
    .center-scores .winner { color: var(--text); }
    .center-scores .loser  { color: var(--text-3); }

    .game-meta { display: grid; grid-template-columns: 1fr auto 1fr;
                 align-items: center; gap: 1.5rem; padding: 0 0 0.5rem 0;
                 font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2); }
    .game-meta .pct { font-weight: 500; }
    .game-meta .pct.away { text-align: right; }
    .prob-track { width: 100%; height: 3px; background: var(--border);
                  border-radius: 1.5px; overflow: hidden; display: flex; }
    .prob-track .away-fill, .prob-track .home-fill { height: 100%; }
    .stats-line { display: flex; gap: 1rem; padding: 0.5rem 0 0 0;
                  font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                  color: var(--text-3); text-transform: uppercase; letter-spacing: 0.08em;
                  justify-content: center; flex-wrap: wrap; }
    .stats-line strong { color: var(--text); font-weight: 500; }

    .tag { display: inline-block; font-family: 'Geist Mono', monospace;
           font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.12em;
           padding: 2px 6px; border-radius: 2px;
           background: var(--surface); color: var(--text-2); border: 1px solid var(--border); }
    .tag.lock    { background: rgba(16,185,129,0.12); color: var(--success); border-color: rgba(16,185,129,0.25); }
    .tag.strong  { background: rgba(59,130,246,0.10); color: var(--info);    border-color: rgba(59,130,246,0.22); }
    .tag.tossup  { background: rgba(245,158,11,0.10); color: var(--warn);    border-color: rgba(245,158,11,0.22); }
    .tag.upset   { background: rgba(225,29,72,0.10);  color: var(--accent);  border-color: rgba(225,29,72,0.25); }
    .tag.final   { background: rgba(255,255,255,0.03); color: var(--text-3); }
    .tag.pick    { background: rgba(59,130,246,0.10); color: var(--info); border-color: rgba(59,130,246,0.22); }

    /* Pick UI */
    .pick-card {
        padding: 1rem 1.25rem; border: 1px solid var(--border);
        border-radius: 6px; background: var(--surface);
        margin-bottom: 0.75rem;
    }
    .pick-choice {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;
        margin-top: 0.75rem;
    }
    .pick-btn {
        padding: 0.75rem; border-radius: 4px;
        background: var(--surface-2); border: 1px solid var(--border-2);
        display: flex; align-items: center; gap: 0.6rem;
        cursor: pointer; transition: all 0.15s;
    }
    .pick-btn.selected { border-color: var(--info); background: rgba(59,130,246,0.1); }
    .pick-btn.selected-correct { border-color: var(--success); background: rgba(16,185,129,0.1); }
    .pick-btn.selected-wrong { border-color: var(--accent); background: rgba(225,29,72,0.1); }
    .pick-btn img { width: 32px; height: 32px; }
    .pick-btn .info { display: flex; flex-direction: column; }
    .pick-btn .info .n { font-weight: 600; font-size: 0.9rem; }
    .pick-btn .info .p { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-3); }

    /* Standings */
    .standings { border-top: 1px solid var(--border-2); }
    .standings-header { display: grid; grid-template-columns: 32px 32px 1fr 44px 60px 100px;
                        gap: 0.75rem; padding: 0.5rem;
                        font-family: 'Geist Mono', monospace; font-size: 0.65rem;
                        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em;
                        border-bottom: 1px solid var(--border); font-weight: 500; }
    .standings-header .right { text-align: right; }
    .standings-row { display: grid; grid-template-columns: 32px 32px 1fr 44px 60px 100px;
                     gap: 0.75rem; padding: 0.65rem 0.5rem;
                     align-items: center; border-bottom: 1px solid var(--border); }
    .standings-row:hover { background: rgba(255,255,255,0.02); }
    .standings-row.playoff { background: linear-gradient(90deg, rgba(225,29,72,0.05) 0%, transparent 30%); }
    .standings-row.focus { background: linear-gradient(90deg, rgba(59,130,246,0.10) 0%, rgba(59,130,246,0.02) 60%, transparent 100%);
                           border-left: 3px solid var(--info); padding-left: calc(0.5rem - 3px); }
    .standings-row .rank { font-family: 'Geist Mono', monospace; font-size: 0.75rem;
                           color: var(--text-3); font-weight: 500; }
    .standings-row .rank.playoff { color: var(--accent); font-weight: 700; }
    .standings-row .team-mark img { width: 24px; height: 24px; }
    .standings-row .team-name { font-size: 0.95rem; font-weight: 500; letter-spacing: -0.01em; }
    .standings-row .team-name .city { color: var(--text-3); font-weight: 400; margin-right: 0.35rem; }
    .standings-row .num { font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums;
                          font-size: 0.85rem; text-align: right; }
    .odds-cell { position: relative; text-align: right; padding-bottom: 6px; }
    .odds-cell .val { font-family: 'Geist Mono', monospace; font-weight: 600; font-size: 0.9rem; }
    .odds-cell .bar { position: absolute; bottom: 0; right: 0; height: 2px;
                      background: var(--accent); opacity: 0.6; max-width: 100%; }

    /* Power rankings */
    .pr-row { display: grid; grid-template-columns: 32px 40px 32px 1fr 70px 60px;
              gap: 0.75rem; padding: 0.6rem 0.5rem;
              align-items: center; border-bottom: 1px solid var(--border); }
    .pr-row .rank { font-family: 'Geist Mono', monospace; font-size: 0.85rem; font-weight: 600; }
    .pr-row .move { font-family: 'Geist Mono', monospace; font-size: 0.75rem; }
    .pr-row .move.up { color: var(--success); }
    .pr-row .move.down { color: var(--accent); }
    .pr-row .move.same { color: var(--text-3); }
    .pr-row .team-mark img { width: 26px; height: 26px; }
    .pr-row .team-name { font-size: 0.95rem; font-weight: 500; }
    .pr-row .num { font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums;
                   font-size: 0.85rem; text-align: right; }
    .pr-row .rec { color: var(--text-3); }

    /* KPI strip */
    .kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr);
                 gap: 2.5rem; padding: 1.5rem 0 2rem;
                 border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
    .kpi .label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em;
                  color: var(--text-3); font-weight: 500; }
    .kpi .value { font-family: 'Geist Mono', monospace; font-size: 1.9rem; font-weight: 700;
                  letter-spacing: -0.02em; line-height: 1.1; margin-top: 0.35rem; color: var(--text); }
    .kpi .value .unit { font-size: 0.9rem; color: var(--text-3); font-weight: 400; margin-left: 0.15rem; }
    .kpi .sub { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2);
                margin-top: 0.3rem; }

    /* Team focus */
    .team-header { display: grid; grid-template-columns: 100px 1fr auto; gap: 1.5rem;
                   align-items: center; padding: 1.5rem 0;
                   border-bottom: 1px solid var(--border); }
    .team-header .mark img { width: 96px; height: 96px; object-fit: contain; }
    .team-header .name-block .city { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em;
                                     color: var(--text-3); font-weight: 500; }
    .team-header .name-block .name { font-size: 2.25rem; font-weight: 800;
                                     letter-spacing: -0.03em; line-height: 1; }
    .team-header .name-block .div { font-family: 'Geist Mono', monospace; font-size: 0.75rem;
                                    color: var(--text-2); margin-top: 0.4rem; letter-spacing: 0.05em; }
    .team-header .status-badge { font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                                 padding: 0.5rem 0.9rem; border-radius: 3px;
                                 text-transform: uppercase; letter-spacing: 0.12em;
                                 font-weight: 600; border: 1px solid; }
    .status-clinched { background: rgba(16,185,129,0.15); color: var(--success); border-color: rgba(16,185,129,0.35); }
    .status-likely   { background: rgba(16,185,129,0.08); color: var(--success); border-color: rgba(16,185,129,0.2); }
    .status-hunt     { background: rgba(59,130,246,0.10); color: var(--info);    border-color: rgba(59,130,246,0.22); }
    .status-bubble   { background: rgba(245,158,11,0.10); color: var(--warn);    border-color: rgba(245,158,11,0.22); }
    .status-longshot { background: rgba(225,29,72,0.08);  color: var(--accent);  border-color: rgba(225,29,72,0.2); }
    .status-out      { background: rgba(255,255,255,0.03); color: var(--text-3); border-color: var(--border); }

    /* Race meter */
    .race-meter { margin: 1rem 0 2rem; padding: 1rem 0; }
    .race-label { display: flex; justify-content: space-between;
                  font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                  color: var(--text-3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
    .race-track { position: relative; height: 24px; background: var(--surface);
                  border: 1px solid var(--border); border-radius: 3px; }
    .race-fill { height: 100%; border-radius: 3px 0 0 3px; transition: width 0.6s; }
    .race-marker { position: absolute; top: -6px; bottom: -6px; width: 2px;
                   background: var(--text-3); opacity: 0.6; }
    .race-marker .mlabel { position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
                           margin-top: 6px; font-family: 'Geist Mono', monospace;
                           font-size: 0.6rem; color: var(--text-3);
                           text-transform: uppercase; letter-spacing: 0.1em; white-space: nowrap; }

    /* Schedule mini */
    .schedule-row { display: grid; grid-template-columns: 32px 32px 1fr 90px 70px 60px;
                    gap: 0.75rem; padding: 0.7rem 0.5rem;
                    align-items: center; border-bottom: 1px solid var(--border); }
    .schedule-row .wk { font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                        color: var(--text-3); font-weight: 500; }
    .schedule-row .side { font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                          color: var(--text-3); text-align: center; }
    .schedule-row .opp { font-size: 0.9rem; font-weight: 500; letter-spacing: -0.01em; }
    .schedule-row .win-bar-cell { position: relative; height: 20px; background: var(--border);
                                   border-radius: 2px; overflow: hidden; }
    .schedule-row .win-bar { height: 100%; }
    .schedule-row .win-pct, .schedule-row .spread {
        font-family: 'Geist Mono', monospace; font-size: 0.8rem; text-align: right;
    }
    .schedule-row .win-pct { font-weight: 600; }
    .schedule-row .spread { color: var(--text-3); }

    /* Bracket */
    .bracket-round {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .bracket-slot {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 4px; padding: 0.6rem 0.8rem;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .bracket-slot .seed { font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                          color: var(--text-3); min-width: 20px; }
    .bracket-slot img { width: 22px; height: 22px; }
    .bracket-slot .name { font-size: 0.85rem; font-weight: 600; }
    .bracket-slot .prob { margin-left: auto; font-family: 'Geist Mono', monospace;
                          font-size: 0.72rem; color: var(--text-2); }

    /* Empty */
    .empty-state { text-align: center; padding: 3rem 1rem; color: var(--text-3);
                   font-family: 'Geist Mono', monospace; font-size: 0.85rem; letter-spacing: 0.05em; }
    .stCode pre { background: var(--surface) !important; border: 1px solid var(--border) !important;
                  border-radius: 4px !important; font-family: 'Geist Mono', monospace !important;
                  font-size: 0.8rem !important; }

    /* ---------- Pick 'Em v2 (vertical mobile-first cards) ---------- */
    .pick-card-v2 {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }
    .pick-card-v2.locked { opacity: 0.7; background: var(--bg); }
    .pick-card-v2 .header-row {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.75rem;
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.1em;
    }
    .pick-card-v2 .header-row .kickoff { color: var(--text-2); }
    .pick-card-v2 .header-row .lock-tag {
        background: rgba(225,29,72,0.1); color: var(--accent);
        padding: 2px 6px; border-radius: 3px; font-weight: 600;
    }
    .pick-card-v2 .matchup {
        display: grid; grid-template-columns: 1fr auto 1fr;
        align-items: center; gap: 0.75rem; margin-bottom: 0.6rem;
    }
    .pick-card-v2 .side { display: flex; align-items: center; gap: 0.6rem; }
    .pick-card-v2 .side.right { justify-content: flex-end; text-align: right; }
    .pick-card-v2 .side img { width: 36px; height: 36px; object-fit: contain; }
    .pick-card-v2 .side .info .team-abbr {
        font-weight: 700; font-size: 1rem; letter-spacing: -0.02em;
    }
    .pick-card-v2 .side .info .team-city {
        font-family: 'Geist Mono', monospace; font-size: 0.65rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.1em;
    }
    .pick-card-v2 .vs { color: var(--text-3); font-family: 'Geist Mono', monospace;
                        font-size: 0.75rem; }
    .pick-card-v2 .model-hint {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0.7rem; margin-top: 0.5rem;
        background: rgba(59,130,246,0.06); border-radius: 6px;
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-2);
    }
    .pick-card-v2 .model-hint strong { color: var(--info); font-weight: 600; }
    .pick-card-v2 .final-row {
        text-align: center; padding: 0.4rem;
        font-family: 'Geist Mono', monospace; font-size: 0.85rem;
        color: var(--text-2); font-weight: 500;
    }
    .pick-card-v2 .final-row .winner { color: var(--text); font-weight: 700; }
    .pick-card-v2 .final-row.correct { color: var(--success); }
    .pick-card-v2 .final-row.incorrect { color: var(--accent); }

    /* Sticky save bar on mobile */
    .save-bar {
        position: sticky; bottom: 0;
        background: linear-gradient(180deg, rgba(10,10,11,0) 0%, var(--bg) 30%, var(--bg) 100%);
        padding: 1.5rem 0 1rem;
        margin: 1rem -1rem 0;
        border-top: 1px solid var(--border);
        z-index: 10;
    }
    .unsaved-pill {
        display: inline-block;
        padding: 4px 10px; border-radius: 12px;
        background: rgba(245,158,11,0.12); color: var(--warn);
        border: 1px solid rgba(245,158,11,0.25);
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
        margin-left: 0.5rem;
    }

    /* Leaderboard */
    .lb-row {
        display: grid; grid-template-columns: 30px 1fr 60px 60px 60px 60px;
        gap: 0.5rem; padding: 0.75rem 0.5rem;
        align-items: center; border-bottom: 1px solid var(--border);
    }
    .lb-row.you { background: linear-gradient(90deg, rgba(59,130,246,0.1) 0%, transparent 60%);
                  border-left: 3px solid var(--info); padding-left: calc(0.5rem - 3px); }
    .lb-row.leader .rank { color: var(--gold); font-weight: 700; }
    .lb-row .rank {
        font-family: 'Geist Mono', monospace; font-size: 0.9rem;
        color: var(--text-3); font-weight: 600;
    }
    .lb-row .username { font-size: 0.95rem; font-weight: 600; letter-spacing: -0.01em; }
    .lb-row .username .subtitle {
        display: block; font-family: 'Geist Mono', monospace; font-size: 0.65rem;
        color: var(--text-3); font-weight: 400; letter-spacing: 0.05em;
        text-transform: uppercase; margin-top: 2px;
    }
    .lb-row .num {
        font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums;
        font-size: 0.9rem; text-align: right;
    }
    .lb-row .num.strong { font-weight: 700; color: var(--text); }
    .lb-row .num.dim { color: var(--text-3); }

    .lb-header {
        display: grid; grid-template-columns: 30px 1fr 60px 60px 60px 60px;
        gap: 0.5rem; padding: 0.5rem;
        font-family: 'Geist Mono', monospace; font-size: 0.65rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em;
        border-bottom: 1px solid var(--border-2); font-weight: 500;
    }
    .lb-header .right { text-align: right; }

    /* League chip */
    .league-chip {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.4rem 0.75rem; border-radius: 999px;
        background: var(--surface); border: 1px solid var(--border);
        font-family: 'Geist Mono', monospace; font-size: 0.72rem;
        color: var(--text-2);
    }
    .league-chip .code {
        font-weight: 700; color: var(--accent); letter-spacing: 0.05em;
    }
    .league-chip .name { color: var(--text); font-weight: 500; letter-spacing: -0.01em; }

    /* ---------- Mobile breakpoints ---------- */
    @media (max-width: 768px) {
        .main .block-container { padding-top: 1rem; padding-left: 0.75rem; padding-right: 0.75rem; }
        .masthead { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
        .masthead-title { font-size: 1.4rem; }
        .masthead-meta { text-align: left; font-size: 0.6rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.6rem 0.75rem; font-size: 0.85rem; }
        .kpi-strip { grid-template-columns: repeat(2, 1fr); gap: 1.25rem; padding: 1rem 0 1.25rem; }
        .kpi .value { font-size: 1.4rem; }
        .game-row {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
        .game-row .team-side.away { justify-content: flex-start; text-align: left; flex-direction: row; }
        .game-row .team-side.away .team-mark { order: -1; }
        .center-scores { justify-content: center; }
        .standings-header, .standings-row {
            grid-template-columns: 26px 26px 1fr 36px 50px 72px;
            gap: 0.4rem;
            font-size: 0.75rem;
        }
        .standings-row .team-name { font-size: 0.85rem; }
        .standings-row .team-name .city { display: none; }
        .lb-header, .lb-row {
            grid-template-columns: 24px 1fr 50px 50px 50px 55px;
            gap: 0.4rem;
        }
        .lb-row .username { font-size: 0.85rem; }
        .lb-row .num { font-size: 0.8rem; }
        .pick-card-v2 { padding: 0.85rem 1rem; }
        .pick-card-v2 .side img { width: 32px; height: 32px; }
        .pick-card-v2 .side .info .team-abbr { font-size: 0.9rem; }
        .team-header { grid-template-columns: 60px 1fr; gap: 1rem; padding: 1rem 0; }
        .team-header .mark img { width: 56px; height: 56px; }
        .team-header .name-block .name { font-size: 1.5rem; }
        .team-header .status-badge { grid-column: 1 / span 2; text-align: center; }
    }

    /* Make Streamlit tap targets bigger on mobile */
    @media (max-width: 640px) {
        .stButton button { padding: 0.75rem 1rem; min-height: 44px; font-size: 0.95rem; }
        .stSelectbox [data-baseweb="select"] { min-height: 44px; }
    }
</style>
"""
st.html(CUSTOM_CSS)


# ------------------------------ data ------------------------------

@st.cache_data
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_features():
    p = Path("data/processed/features.parquet")
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_schedules():
    p = Path("data/processed/games_master.csv")
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_backtest_reports(outputs_dir):
    reports = {}
    for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin',
                 'backtest_total', 'backtest_scores', 'backtest_vs_vegas']:
        p = Path(outputs_dir) / f"{name}.txt"
        if p.exists():
            reports[name] = p.read_text(encoding='utf-8', errors='replace')
    return reports

@st.cache_data
def load_vegas_detail(outputs_dir):
    p = Path(outputs_dir) / "backtest_vs_vegas.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_model(models_dir, name):
    p = Path(models_dir) / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def compute_records(schedules, season, upto_week):
    wl = defaultdict(lambda: [0, 0])
    played = schedules[(schedules['season'] == season)
                       & (schedules['week'] < upto_week)
                       & (schedules['home_score'].notna())]
    for _, g in played.iterrows():
        h, a = g['home_team'], g['away_team']
        if g['home_score'] > g['away_score']:
            wl[h][0] += 1; wl[a][1] += 1
        else:
            wl[a][0] += 1; wl[h][1] += 1
    return {t: f"{w}-{l}" for t, (w, l) in wl.items()}

@st.cache_data
def predict_season(_feats, season, _wp, _mg, _tot):
    subset = _feats[_feats['season'] == season].copy()
    subset['pred_home_win_prob'] = _wp['model'].predict_proba(subset[_wp['features']].fillna(0))[:, 1]
    subset['pred_margin'] = _mg['model'].predict(subset[_mg['features']].fillna(0))
    subset['pred_total'] = _tot['model'].predict(subset[_tot['features']].fillna(0))
    subset['pred_home_score'] = ((subset['pred_margin'] + subset['pred_total']) / 2).round(0).astype(int)
    subset['pred_away_score'] = ((subset['pred_total'] - subset['pred_margin']) / 2).round(0).astype(int)
    keep = ['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team',
            'home_score', 'away_score',
            'pred_home_win_prob', 'pred_margin', 'pred_total',
            'pred_home_score', 'pred_away_score']
    if 'gametime' in subset.columns:
        keep.insert(4, 'gametime')
    return subset[keep].sort_values(['week', 'gameday'])


# ------------------------------ components ------------------------------

def confidence_tag(p_home):
    p = max(p_home, 1 - p_home)
    if p >= 0.75: return "lock", "Lock"
    if p >= 0.63: return "strong", "Strong"
    if p <= 0.55: return "tossup", "Toss-Up"
    return "", "Lean"


def game_row(row, records, user_pick=None):
    """user_pick: team code the user picked, or None."""
    home, away = row['home_team'], row['away_team']
    home_meta = TEAMS.get(home, {}); away_meta = TEAMS.get(away, {})
    p_home = float(row['pred_home_win_prob']); p_away = 1 - p_home
    ph = int(round(p_home * 100)); pa = 100 - ph
    played = pd.notna(row.get('home_score'))
    if played:
        hs, as_ = int(row['home_score']), int(row['away_score'])
        home_win = hs > as_
        winner = home if home_win else away
        tag_html = '<span class="tag final">Final</span>'
        if user_pick:
            row_cls = 'user-picked-correct' if user_pick == winner else 'user-picked-wrong'
        else:
            row_cls = ''
    else:
        hs, as_ = int(row['pred_home_score']), int(row['pred_away_score'])
        home_win = hs > as_
        cc, ct = confidence_tag(p_home)
        tag_html = f'<span class="tag {cc}">{ct}</span>'
        row_cls = 'lock' if cc == 'lock' else ('upset' if cc == 'tossup' else '')

    pick_tag = f'<span class="tag pick">You: {short_name(user_pick)}</span>' if user_pick else ''
    aw_cls = 'winner' if not home_win else 'loser'
    hw_cls = 'winner' if home_win else 'loser'
    date_str = str(row['gameday'])[:10] if pd.notna(row.get('gameday')) else ''
    hr = records.get(home, '0-0'); ar = records.get(away, '0-0')
    hc = primary_color(home); ac = primary_color(away)

    return (
        f'<div class="game-row {row_cls}">'
          '<div class="team-side away">'
            f'<div class="team-info"><div class="city">{away_meta.get("city", away)}</div>'
            f'<div class="name">{short_name(away)}</div><div class="record">{ar}</div></div>'
            f'<div class="team-mark"><img src="{logo_url(away)}"/></div>'
          '</div>'
          '<div class="center-scores">'
            f'<div class="away {aw_cls}">{as_}</div>'
            '<div class="sep">·</div>'
            f'<div class="home {hw_cls}">{hs}</div>'
          '</div>'
          '<div class="team-side">'
            f'<div class="team-mark"><img src="{logo_url(home)}"/></div>'
            f'<div class="team-info"><div class="city">{home_meta.get("city", home)}</div>'
            f'<div class="name">{short_name(home)}</div><div class="record">{hr}</div></div>'
          '</div>'
        '</div>'
        '<div class="game-meta">'
          f'<div class="pct away">{short_name(away).upper()} {pa}%</div>'
          '<div class="prob-track">'
            f'<div class="away-fill" style="width:{pa}%; background:{ac};"></div>'
            f'<div class="home-fill" style="width:{ph}%; background:{hc};"></div>'
          '</div>'
          f'<div class="pct">{short_name(home).upper()} {ph}%</div>'
        '</div>'
        '<div class="stats-line">'
          f'{tag_html}{pick_tag}'
          f'<span>SPREAD <strong>{row["pred_margin"]:+.1f}</strong></span>'
          f'<span>TOTAL <strong>{row["pred_total"]:.1f}</strong></span>'
          f'{f"<span>{date_str}</span>" if date_str else ""}'
        '</div>'
    )


def kpi(label, value, unit="", sub=""):
    unit_span = f'<span class="unit">{unit}</span>' if unit else ''
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}{unit_span}</div>'
            f'<div class="sub">{sub}</div></div>')


def standings_row_html(rank, row, focus_team=None, top_seeds=7):
    team = row['team']
    meta = TEAMS.get(team, {})
    is_playoff = rank <= top_seeds
    is_focus = (team == focus_team)
    odds = float(row['playoff_odds']); bar = int(odds * 100)
    cls = ['standings-row']
    if is_playoff: cls.append('playoff')
    if is_focus:   cls.append('focus')
    rank_cls = ' playoff' if is_playoff else ''
    return (
        f'<div class="{" ".join(cls)}">'
        f'<div class="rank{rank_cls}">{rank}</div>'
        f'<div class="team-mark"><img src="{logo_url(team)}"/></div>'
        f'<div class="team-name"><span class="city">{meta.get("city", "")}</span>{meta.get("name", team)}</div>'
        f'<div class="num">{int(row["current_wins"])}</div>'
        f'<div class="num">{row["sim_wins_mean"]:.1f}</div>'
        f'<div class="odds-cell"><div class="val">{odds*100:.1f}%</div>'
        f'<div class="bar" style="width:{bar}%;"></div></div></div>'
    )


def race_meter(odds):
    pct = int(round(odds * 100))
    color = "var(--success)" if odds >= 0.75 else "var(--warn)" if odds >= 0.5 else "var(--accent)"
    return (
        '<div class="race-meter">'
        f'<div class="race-label"><span>Playoff Race</span><span>{pct}% odds</span></div>'
        '<div class="race-track">'
        f'<div class="race-fill" style="width:{pct}%; background:{color};"></div>'
        '<div class="race-marker" style="left:50%;"><span class="mlabel">50% cutoff</span></div>'
        '<div class="race-marker" style="left:75%;"><span class="mlabel">75% likely</span></div>'
        '</div></div>'
    )


def playoff_status(odds):
    if odds >= 0.95:  return "status-clinched", "Virtual Lock"
    if odds >= 0.75:  return "status-likely",   "Likely"
    if odds >= 0.50:  return "status-hunt",     "In The Hunt"
    if odds >= 0.20:  return "status-bubble",   "Bubble"
    if odds >= 0.05:  return "status-longshot", "Long Shot"
    return "status-out", "On The Outside"


def pick_card_v2(game_row, user_pick=None, model_pick_team=None,
                 model_pick_prob=None, confidence=None, played=False):
    """
    Vertical, mobile-first game card for the Pick 'Em tab. Renders the
    matchup + model hint + a lock indicator. The actual pick control
    (radio/select) is rendered by the caller (Streamlit widgets can't
    live inside our HTML).
    """
    home = game_row['home_team']; away = game_row['away_team']
    home_meta = TEAMS.get(home, {}); away_meta = TEAMS.get(away, {})
    locked = is_locked(game_row.get('gameday'), game_row.get('gametime'))
    ko_str = kickoff_display(game_row.get('gameday'), game_row.get('gametime'))
    lock_html = '<span class="lock-tag">Locked</span>' if locked else ''
    conf_txt = f'Conf: {confidence}' if confidence else 'No confidence set'

    if model_pick_team and model_pick_prob is not None:
        model_line = (f'<div class="model-hint"><span>Model pick</span>'
                      f'<span><strong>{short_name(model_pick_team)}</strong> '
                      f'{model_pick_prob*100:.0f}%</span></div>')
    else:
        model_line = ''

    final_html = ''
    if played:
        hs, as_ = int(game_row['home_score']), int(game_row['away_score'])
        winner = home if hs > as_ else away
        cls = ''
        if user_pick:
            cls = 'correct' if user_pick == winner else 'incorrect'
        result = f"{short_name(away)} {as_} · <span class='winner'>{short_name(winner)}</span> · {short_name(home)} {hs}"
        final_html = f'<div class="final-row {cls}">{result}</div>'

    return (
        f'<div class="pick-card-v2{" locked" if locked else ""}">'
          f'<div class="header-row"><span class="kickoff">{ko_str}</span>'
          f'<span>{conf_txt}{lock_html}</span></div>'
          '<div class="matchup">'
            '<div class="side">'
              f'<img src="{logo_url(away)}"/>'
              f'<div class="info"><div class="team-abbr">{short_name(away)}</div>'
              f'<div class="team-city">{away_meta.get("city", "")}</div></div>'
            '</div>'
            '<div class="vs">@</div>'
            '<div class="side right">'
              f'<div class="info"><div class="team-abbr">{short_name(home)}</div>'
              f'<div class="team-city">{home_meta.get("city", "")}</div></div>'
              f'<img src="{logo_url(home)}"/>'
            '</div>'
          '</div>'
          f'{model_line}'
          f'{final_html}'
        '</div>'
    )


def league_chip(league_id, name):
    return (f'<div class="league-chip"><span class="code">{league_id}</span>'
            f'<span class="name">{name}</span></div>')


def bracket_slot(seed, team, prob):
    meta = TEAMS.get(team, {})
    return (
        '<div class="bracket-slot">'
        f'<span class="seed">#{seed}</span>'
        f'<img src="{logo_url(team)}"/>'
        f'<span class="name">{meta.get("name", team)}</span>'
        f'<span class="prob">{prob*100:.0f}%</span>'
        '</div>'
    )


# ------------------------------ main ------------------------------

def main():
    config = load_config()
    models_dir = config['data']['models_dir']
    outputs_dir = config['data']['outputs_dir']

    feats = load_features()
    schedules = load_schedules()

    # Header
    st.html(
        '<div class="masthead">'
        '<div class="masthead-title">NFL <span class="accent">Predictor</span></div>'
        '<div class="masthead-meta">Model v2 · Elo + QB + LGBM<br>'
        '2026 season · Live projections + Pick \'Em</div>'
        '</div>'
    )

    # Session-persistent user ID (simple honor-system multi-user)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ""

    # Profile strip if user is "logged in"
    if st.session_state.user_id:
        user_id = st.session_state.user_id
        badges = game.get_achievements(user_id)
        badges_html = "".join(
            f'<span class="badge">{game.ACHIEVEMENTS.get(b["achievement"], {}).get("name", b["achievement"])}</span>'
            for _, b in badges.iterrows()
        )
        st.html(
            '<div class="profile-strip">'
            '<span class="who">User</span>'
            f'<span class="name">{user_id}</span>'
            f'{badges_html}'
            f'<span class="record">{len(badges)} achievement{"s" if len(badges) != 1 else ""}</span>'
            '</div>'
        )

    if feats is None or schedules is None:
        st.error("Data not built. Run `python make_dataset.py` and `python -m src.features.build_features`.")
        return
    winprob_m = load_model(models_dir, 'winprob_model')
    margin_m = load_model(models_dir, 'margin_model')
    total_m = load_model(models_dir, 'total_model')
    if not all([winprob_m, margin_m, total_m]):
        st.warning("Models not trained.")
        return

    tabs = st.tabs(["Weekly Games", "Pick 'Em", "Team Focus", "Season Simulator", "Model Backtest"])
    tab_games, tab_pick, tab_focus, tab_sim, tab_back = tabs

    # ============================== TAB: Weekly Games ==============================
    with tab_games:
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 2, 1.5])
        with c1:
            seasons = sorted(feats['season'].unique())
            season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="wk_season")
        with c2:
            weeks = sorted(feats[feats['season'] == season]['week'].unique())
            week = st.selectbox("Week", weeks, key="wk_week")
        with c3:
            team_options = ["All Teams"] + sorted(TEAMS.keys())
            team_filter = st.selectbox("Team Filter", team_options, key="wk_team",
                                        format_func=lambda t: t if t == "All Teams" else f"{t} — {team_name(t)}")
        with c4:
            sort_mode = st.selectbox("Sort", ["Kickoff", "Confidence High", "Confidence Low", "Upset Risk"], key="wk_sort")

        preds = predict_season(feats, season, winprob_m, margin_m, total_m)
        preds = preds[preds['week'] == week]

        if team_filter != "All Teams":
            preds = preds[(preds['home_team'] == team_filter) | (preds['away_team'] == team_filter)]
        if sort_mode == "Confidence High":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: max(p, 1-p))).sort_values('_c', ascending=False)
        elif sort_mode == "Confidence Low":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: max(p, 1-p))).sort_values('_c', ascending=True)
        elif sort_mode == "Upset Risk":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: min(p, 1-p))).sort_values('_c', ascending=False)

        records = compute_records(schedules, season, week)

        # Load user picks for this week if available
        user_picks_map = {}
        if st.session_state.user_id:
            up = game.get_picks(st.session_state.user_id, season, week)
            user_picks_map = dict(zip(up['game_id'], up['pick_team']))

        if len(preds) == 0:
            st.html('<div class="empty-state">No games match the current filters.</div>')
        else:
            filter_desc = "" if team_filter == "All Teams" else f" · {short_name(team_filter)} only"
            st.html(f'<div class="eyebrow"><span>{season} · Week {week}{filter_desc}</span>'
                    f'<span class="count">{len(preds)} games</span></div>')
            st.html("".join(game_row(g, records, user_pick=user_picks_map.get(g['game_id']))
                            for _, g in preds.iterrows()))

    # ============================== TAB: Pick 'Em (League) ==============================
    with tab_pick:
        # ---------- Auth (username) ----------
        c1, c2 = st.columns([2, 3])
        with c1:
            uid = st.text_input("Your username", value=st.session_state.user_id,
                                key="pick_uid", placeholder="e.g. danny")
            if uid.strip() != st.session_state.user_id:
                st.session_state.user_id = uid.strip()
                st.session_state.pop('current_league_id', None)
                st.rerun()

        if not st.session_state.user_id:
            st.html('<div class="empty-state">Enter a username above to see your leagues and make picks.</div>')
        else:
            user_id = st.session_state.user_id

            # ---------- League selection ----------
            user_leagues = leagues.get_user_leagues(user_id)
            has_leagues = len(user_leagues) > 0

            lc1, lc2 = st.columns([3, 2])
            with lc1:
                if has_leagues:
                    league_options = user_leagues['league_id'].tolist()
                    current_lid = st.session_state.get('current_league_id')
                    if current_lid not in league_options:
                        current_lid = league_options[0]
                        st.session_state.current_league_id = current_lid
                    league_labels = {row['league_id']: f"{row['name']}  ({row['league_id']})  ·  {int(row['member_count'])} members"
                                      for _, row in user_leagues.iterrows()}
                    picked = st.selectbox("League", league_options,
                                          index=league_options.index(current_lid),
                                          format_func=lambda x: league_labels.get(x, x),
                                          key="league_picker")
                    if picked != current_lid:
                        st.session_state.current_league_id = picked
                        st.rerun()
                else:
                    st.info("You aren't in any leagues yet. Create one, or join a friend's below.")

            with lc2:
                action = st.radio("Manage", ["Play", "Create League", "Join League"],
                                   horizontal=True, key="league_action",
                                   label_visibility="collapsed")

            # ---------- Create/Join flow ----------
            if action == "Create League":
                with st.form("create_league", clear_on_submit=True):
                    new_name = st.text_input("League name", placeholder="e.g. Danny's Office Pool")
                    create_season = st.selectbox("Season",
                                                  sorted(feats['season'].unique(), reverse=True))
                    submitted = st.form_submit_button("Create league", type="primary")
                    if submitted:
                        if not new_name.strip():
                            st.error("Give the league a name.")
                        else:
                            try:
                                new_code = leagues.create_league(
                                    new_name.strip(), user_id, int(create_season), 'confidence')
                                st.success(f"Created **{new_name}**. Invite code: **{new_code}** — share it with your friends.")
                                st.session_state.current_league_id = new_code
                                st.session_state.league_action = "Play"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Couldn't create: {e}")

            elif action == "Join League":
                with st.form("join_league", clear_on_submit=True):
                    code_input = st.text_input("Invite code", placeholder="e.g. K3M8P2",
                                                max_chars=8)
                    submitted = st.form_submit_button("Join league", type="primary")
                    if submitted:
                        code = code_input.strip().upper()
                        if not code:
                            st.error("Enter an invite code.")
                        elif leagues.join_league(code, user_id):
                            st.success(f"Joined league {code}.")
                            st.session_state.current_league_id = code
                            st.session_state.league_action = "Play"
                            st.rerun()
                        else:
                            st.error(f"League {code} not found. Double-check the code.")

            elif action == "Play" and has_leagues:
                current_lid = st.session_state.current_league_id
                league = leagues.get_league(current_lid)
                league_season = int(league['season'])

                # Season/week selectors
                sc1, sc2, sc3 = st.columns([1, 1, 2])
                with sc1:
                    pick_season = st.selectbox("Season",
                                                sorted(feats['season'].unique()),
                                                index=list(sorted(feats['season'].unique())).index(league_season)
                                                if league_season in list(feats['season'].unique())
                                                else len(feats['season'].unique()) - 1,
                                                key="pick_season")
                with sc2:
                    pick_weeks = sorted(feats[feats['season'] == pick_season]['week'].unique())
                    pick_week = st.selectbox("Week", pick_weeks, key="pick_week")

                # League chip + tabs
                st.html(f'<div style="margin: 0.5rem 0;">{league_chip(current_lid, league["name"])}</div>')

                pick_tab, board_tab, mem_tab = st.tabs(["Make Picks", "Leaderboard", "Members"])

                all_preds = predict_season(feats, pick_season, winprob_m, margin_m, total_m)

                # ---------- Sub-tab: MAKE PICKS ----------
                with pick_tab:
                    week_preds = all_preds[all_preds['week'] == pick_week].reset_index(drop=True)
                    if len(week_preds) == 0:
                        st.html(f'<div class="empty-state">No games for Week {pick_week}.</div>')
                    else:
                        n_games = len(week_preds)

                        # Load saved picks from DB into initial session state
                        saved_picks_df = leagues.get_picks(current_lid, user_id=user_id,
                                                            season=pick_season, week=pick_week)
                        saved = {r['game_id']: {'pick_team': r['pick_team'],
                                                'confidence': int(r['confidence']) if pd.notna(r.get('confidence')) else None}
                                 for _, r in saved_picks_df.iterrows()}

                        # Session-state pending picks (per user/league/week)
                        state_key = f"pending_picks_{current_lid}_{pick_season}_{pick_week}"
                        if state_key not in st.session_state:
                            st.session_state[state_key] = dict(saved)  # start with saved

                        pending = st.session_state[state_key]

                        # Count unsaved
                        unsaved_count = sum(1 for gid, p in pending.items()
                                             if p != saved.get(gid))
                        unsaved_html = (f'<span class="unsaved-pill">{unsaved_count} unsaved</span>'
                                        if unsaved_count else '')

                        st.html(f'<div class="eyebrow"><span>Week {pick_week} Picks · Confidence 1-{n_games}{unsaved_html}</span>'
                                f'<span class="count">{n_games} games</span></div>')

                        # Render each game
                        for _, g in week_preds.iterrows():
                            gid = g['game_id']
                            home, away = g['home_team'], g['away_team']
                            p_home = float(g['pred_home_win_prob'])
                            model_pick = home if p_home > 0.5 else away
                            model_prob = max(p_home, 1 - p_home)
                            played = pd.notna(g.get('home_score'))
                            locked = played or is_locked(g.get('gameday'), g.get('gametime'))
                            cur = pending.get(gid, {})

                            with st.container():
                                st.html(pick_card_v2(
                                    g, user_pick=cur.get('pick_team'),
                                    model_pick_team=model_pick, model_pick_prob=model_prob,
                                    confidence=cur.get('confidence'), played=played,
                                ))
                                cA, cB = st.columns([3, 1])
                                with cA:
                                    options = [away, home]
                                    idx = options.index(cur.get('pick_team')) if cur.get('pick_team') in options else 0
                                    pick = st.radio(
                                        f"Winner · {short_name(away)} @ {short_name(home)}",
                                        options=options, index=idx,
                                        format_func=lambda t: f"{short_name(t)} — {team_name(t)}",
                                        key=f"pick_{gid}", horizontal=True,
                                        disabled=locked, label_visibility="collapsed",
                                    )
                                with cB:
                                    conf = st.number_input(
                                        "Conf", min_value=1, max_value=n_games,
                                        value=cur.get('confidence') or n_games,
                                        step=1, key=f"conf_{gid}",
                                        disabled=locked, label_visibility="collapsed",
                                    )
                                # Update pending
                                if not locked:
                                    pending[gid] = {'pick_team': pick, 'confidence': int(conf)}

                        # ---------- Save bar (sticky-ish on mobile via CSS) ----------
                        st.html('<div class="save-bar"></div>')
                        used_confidences = [p['confidence'] for p in pending.values() if p.get('confidence')]
                        dupes = [c for c in used_confidences if used_confidences.count(c) > 1]
                        b1, b2 = st.columns([1, 3])
                        with b1:
                            save_clicked = st.button("Save Picks", type="primary", use_container_width=True)
                        with b2:
                            if dupes:
                                st.warning(f"Confidence values must be unique. Duplicates: {sorted(set(dupes))}")
                            elif unsaved_count == 0:
                                st.caption("All picks saved.")
                            else:
                                st.caption(f"{unsaved_count} pick(s) not yet saved.")

                        if save_clicked:
                            if dupes:
                                st.error("Fix duplicate confidence values before saving.")
                            else:
                                to_save = {gid: p for gid, p in pending.items()
                                            if p != saved.get(gid) and not is_locked(
                                                week_preds[week_preds['game_id'] == gid].iloc[0].get('gameday'),
                                                week_preds[week_preds['game_id'] == gid].iloc[0].get('gametime'))}
                                if to_save:
                                    leagues.save_picks_batch(current_lid, user_id, int(pick_season), int(pick_week), to_save)
                                    game.grant_achievement(user_id, 'first_pick', int(pick_season))
                                st.success(f"Saved {len(to_save)} pick(s).")
                                st.rerun()

                # ---------- Sub-tab: LEADERBOARD ----------
                with board_tab:
                    lb = leagues.score_league_season(current_lid, pick_season, all_preds, schedules)
                    if lb.empty:
                        st.html('<div class="empty-state">No completed games yet this season. Come back after Week 1.</div>')
                    else:
                        st.html(f'<div class="eyebrow"><span>Season Standings</span>'
                                f'<span class="count">{len(lb)} players · {int(lb["weeks_scored"].max())} weeks scored</span></div>')
                        header = ('<div class="lb-header">'
                                  '<div>#</div><div>Player</div>'
                                  '<div class="right">Pts</div>'
                                  '<div class="right">W-L</div>'
                                  '<div class="right">%</div>'
                                  '<div class="right">Best Wk</div></div>')
                        rows_html = header
                        for i, row in lb.iterrows():
                            rank = i + 1
                            classes = ['lb-row']
                            if row['user_id'] == user_id: classes.append('you')
                            if rank == 1: classes.append('leader')
                            wl = f"{int(row['correct'])}-{int(row['total']) - int(row['correct'])}"
                            beat_line = f"{int(row['beat_model_wks'])} weeks beat model"
                            rows_html += (
                                f'<div class="{" ".join(classes)}">'
                                f'<div class="rank">{rank}</div>'
                                f'<div class="username">{row["user_id"]}'
                                f'<span class="subtitle">{beat_line}</span></div>'
                                f'<div class="num strong">{int(row["confidence_pts"])}</div>'
                                f'<div class="num">{wl}</div>'
                                f'<div class="num dim">{row["pct"]:.0f}%</div>'
                                f'<div class="num">{int(row["best_week_pts"])}</div>'
                                '</div>'
                            )
                        st.html(rows_html)

                # ---------- Sub-tab: MEMBERS ----------
                with mem_tab:
                    members = leagues.get_league_members(current_lid)
                    st.html(f'<div class="eyebrow"><span>League: {league["name"]}</span>'
                            f'<span class="count">Code: {current_lid} · Share with friends</span></div>')
                    if len(members) == 0:
                        st.info("No members yet — invite someone with the code above.")
                    else:
                        for _, m in members.iterrows():
                            st.markdown(f"- **{m['user_id']}** — joined {str(m['joined_at'])[:10]}")

    # ============================== TAB: Team Focus ==============================
    with tab_focus:
        c1, c2, c3, _ = st.columns([1.5, 1.5, 1.2, 2])
        with c1:
            teams_sorted = sorted(TEAMS.keys())
            default_idx = teams_sorted.index("BUF") if "BUF" in teams_sorted else 0
            focus_team = st.selectbox("Team", teams_sorted, index=default_idx,
                                      format_func=lambda t: f"{t} — {team_name(t)}", key="focus_team")
        with c2:
            compare_opts = ["None (single team view)"] + [t for t in teams_sorted if t != focus_team]
            compare_team = st.selectbox("Compare To", compare_opts,
                                        format_func=lambda t: t if t == "None (single team view)" else f"{t} — {team_name(t)}",
                                        key="compare_team")
            if compare_team == "None (single team view)":
                compare_team = None
        with c3:
            focus_season = st.selectbox("Season", sorted(feats['season'].unique()),
                                        index=len(feats['season'].unique()) - 1, key="focus_season")

        sim_files = sorted(Path(outputs_dir).glob(f"season_sim_{focus_season}_*.csv"))
        sim_df = pd.read_csv(sim_files[-1]) if sim_files else None
        sim_row = sim_df[sim_df['team'] == focus_team].iloc[0] if (sim_df is not None and len(sim_df[sim_df['team'] == focus_team])) else None

        meta = TEAMS.get(focus_team, {})
        div_name = f"{meta.get('conf', '')} {meta.get('div', '')}".strip()
        if sim_row is not None:
            odds = float(sim_row['playoff_odds']); status_cls, status_label = playoff_status(odds)
        else:
            odds = 0; status_cls, status_label = "status-out", "No Sim Data"

        st.html(
            '<div class="team-header">'
            f'<div class="mark"><img src="{logo_url(focus_team)}"/></div>'
            f'<div class="name-block"><div class="city">{meta.get("city", "")}</div>'
            f'<div class="name">{meta.get("name", focus_team)}</div>'
            f'<div class="div">{div_name}</div></div>'
            f'<div class="status-badge {status_cls}">{status_label}</div>'
            '</div>'
        )

        if sim_row is not None:
            conf_teams_df = sim_df[sim_df['conference'] == meta.get('conf', '')].sort_values('playoff_odds', ascending=False).reset_index(drop=True)
            conf_rank = int(conf_teams_df.index[conf_teams_df['team'] == focus_team][0]) + 1
            st.html(
                '<div class="kpi-strip">'
                + kpi("Playoff Odds", f"{odds*100:.1f}", "%", sub=f"AFC/NFC seed #{conf_rank}")
                + kpi("Division Odds", f"{float(sim_row['division_win_odds'])*100:.1f}", "%", sub=div_name)
                + kpi("Projected Wins", f"{sim_row['sim_wins_mean']:.1f}",
                      sub=f"Range: {sim_row['sim_wins_p10']:.0f}–{sim_row['sim_wins_p90']:.0f}")
                + kpi("Current Wins", f"{int(sim_row['current_wins'])}", sub="Regular season")
                + '</div>'
            )
            st.html(race_meter(odds))

        # Comparison mode: show side-by-side
        if compare_team and sim_df is not None:
            comp_row = sim_df[sim_df['team'] == compare_team]
            if len(comp_row):
                comp_row = comp_row.iloc[0]
                st.html(f'<div class="eyebrow"><span>Head to Head · {short_name(focus_team)} vs {short_name(compare_team)}</span></div>')
                col1, col2, col3 = st.columns(3)
                col1.metric(f"Playoff %", f"{odds*100:.1f}%", f"{(odds - comp_row['playoff_odds'])*100:+.1f}")
                col2.metric(f"Proj Wins", f"{sim_row['sim_wins_mean']:.1f}", f"{sim_row['sim_wins_mean'] - comp_row['sim_wins_mean']:+.1f}")
                col3.metric(f"Div Win %", f"{float(sim_row['division_win_odds'])*100:.1f}%", f"{(sim_row['division_win_odds'] - comp_row['division_win_odds'])*100:+.1f}")

        # Full schedule
        all_preds = predict_season(feats, focus_season, winprob_m, margin_m, total_m)
        team_games = all_preds[(all_preds['home_team'] == focus_team) | (all_preds['away_team'] == focus_team)].copy()
        if len(team_games) > 0:
            st.html('<div class="eyebrow"><span>Full Schedule</span><span class="count">Predictions + Actuals</span></div>')
            rows_html = ""
            for _, g in team_games.iterrows():
                is_home = g['home_team'] == focus_team
                opp = g['away_team'] if is_home else g['home_team']
                side_ind = "vs" if is_home else "at"
                tw_prob = g['pred_home_win_prob'] if is_home else (1 - g['pred_home_win_prob'])
                tw_pct = int(round(tw_prob * 100))
                spread = g['pred_margin'] if is_home else -g['pred_margin']
                played = pd.notna(g.get('home_score'))
                if played:
                    my = int(g['home_score']) if is_home else int(g['away_score'])
                    opps = int(g['away_score']) if is_home else int(g['home_score'])
                    res = f'<span style="color:{"var(--success)" if my > opps else "var(--accent)"};font-weight:600;">{"W" if my > opps else "L"} {my}-{opps}</span>'
                    row_cls = "played"
                else:
                    res = f'<span class="win-pct">{tw_pct}%</span>'
                    row_cls = ""
                bar_c = primary_color(focus_team) if tw_prob >= 0.5 else primary_color(opp)
                rows_html += (
                    f'<div class="schedule-row {row_cls}">'
                    f'<div class="wk">W{int(g["week"])}</div>'
                    f'<div class="team-mark"><img src="{logo_url(opp)}"/></div>'
                    f'<div class="opp"><span class="side">{side_ind}</span> {team_name(opp)}</div>'
                    f'<div class="win-bar-cell"><div class="win-bar" style="width:{tw_pct}%; background:{bar_c};"></div></div>'
                    f'<div class="win-pct">{res if played else f"{tw_pct}%"}</div>'
                    f'<div class="spread">{spread:+.1f}</div>'
                    '</div>'
                )
            st.html(rows_html)

    # ============================== TAB: Season Simulator ==============================
    with tab_sim:
        sim_files = sorted(Path(outputs_dir).glob("season_sim_*.csv"))
        if not sim_files:
            st.info("No season simulations found.")
        else:
            c1, c2, _ = st.columns([1.5, 1.5, 3])
            with c1:
                sim_name = st.selectbox("Simulation", [f.name for f in sim_files],
                                        index=len(sim_files) - 1, key="sim_file")
            with c2:
                team_opts = ["None"] + sorted(TEAMS.keys())
                sim_focus = st.selectbox("Highlight Team", team_opts,
                                          format_func=lambda t: t if t == "None" else f"{t} — {team_name(t)}",
                                          key="sim_focus")
            focus_team_sim = None if sim_focus == "None" else sim_focus
            df = pd.read_csv(Path(outputs_dir) / sim_name)

            top_playoff = df.loc[df['playoff_odds'].idxmax()]
            top_div = df.loc[df['division_win_odds'].idxmax()]
            virtual_locks = (df['playoff_odds'] > 0.85).sum()

            st.html(
                '<div class="kpi-strip">'
                + kpi("Simulations", "10,000", sub="Monte Carlo iterations")
                + kpi("Virtual Locks", str(virtual_locks), sub="teams above 85% odds")
                + kpi("Top Playoff Odds", f"{top_playoff['playoff_odds']*100:.1f}", "%", sub=team_name(top_playoff['team']))
                + kpi("Top Division Odds", f"{top_div['division_win_odds']*100:.1f}", "%", sub=team_name(top_div['team']))
                + '</div>'
            )

            afc_col, nfc_col = st.columns(2, gap="large")
            for label, conf_col in [("AFC", afc_col), ("NFC", nfc_col)]:
                with conf_col:
                    st.html(f'<div class="eyebrow"><span>{label} Playoff Picture</span><span class="count">Top 7 seed</span></div>')
                    conf_df = df[df['conference'] == label].sort_values('playoff_odds', ascending=False).reset_index(drop=True)
                    header = ('<div class="standings-header"><div>#</div><div></div><div>Team</div>'
                              '<div class="right">W</div><div class="right">Proj</div>'
                              '<div class="right">Playoff%</div></div>')
                    rows = "".join(standings_row_html(i + 1, r, focus_team=focus_team_sim) for i, r in conf_df.iterrows())
                    st.html('<div class="standings">' + header + rows + '</div>')

            # Playoff Bracket
            st.html('<div class="eyebrow"><span>Playoff Bracket Projection</span><span class="count">Round-by-round odds</span></div>')
            bracket = compute_bracket_probs(str(Path(outputs_dir) / sim_name))
            for conf in ['AFC', 'NFC']:
                st.html(f'<div style="font-family: Geist Mono, monospace; font-size: 0.75rem; '
                        f'color: var(--text-2); text-transform: uppercase; letter-spacing: 0.15em; '
                        f'margin: 1rem 0 0.5rem;">{conf} · Wildcard → Div → Conf → Super Bowl</div>')
                slots_by_round = {
                    'WC': [(s['seed'], s['team'], s['wc_win']) for s in bracket[conf]],
                    'Div': [(s['seed'], s['team'], s['div_win']) for s in bracket[conf]],
                    'Conf': [(s['seed'], s['team'], s['conf_win']) for s in bracket[conf]],
                    'SB': [(s['seed'], s['team'], s['sb_win']) for s in bracket[conf]],
                }
                cols = st.columns(4)
                for i, (round_name, slots) in enumerate(slots_by_round.items()):
                    with cols[i]:
                        st.html(f'<div style="font-family: Geist Mono, monospace; font-size: 0.65rem; '
                                f'color: var(--text-3); text-transform: uppercase; letter-spacing: 0.15em; '
                                f'margin-bottom: 0.5rem;">{round_name}</div>')
                        html = "".join(bracket_slot(s, t, p) for s, t, p in slots)
                        st.html(html)

            # Playoff odds distribution
            st.html('<div class="eyebrow"><span>Playoff Odds Distribution</span><span class="count">All 32 teams</span></div>')
            chart_df = df.sort_values('playoff_odds', ascending=True).copy()
            chart_df['playoff_pct'] = (chart_df['playoff_odds'] * 100).round(1)
            chart_df['is_focus'] = (chart_df['team'] == focus_team_sim) if focus_team_sim else False
            bar = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('playoff_pct:Q', title='Playoff Odds (%)', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('team:N', sort=alt.EncodingSortField(field='playoff_pct', order='ascending'), title=None,
                        axis=alt.Axis(labelFontSize=10)),
                color=alt.condition(
                    alt.datum.is_focus,
                    alt.value('#3B82F6'),
                    alt.Color('conference:N',
                              scale=alt.Scale(domain=['AFC', 'NFC'], range=['#E11D48', '#7A0F27']),
                              legend=alt.Legend(orient='top-right', title=None))
                ),
                tooltip=['team', 'conference', 'division', 'current_wins', 'sim_wins_mean',
                         alt.Tooltip('playoff_pct:Q', title='Playoff %', format='.1f')]
            ).properties(height=560)
            st.altair_chart(bar, use_container_width=True)

            # Power Rankings
            pr_path = Path(outputs_dir) / "power_rankings.csv"
            if pr_path.exists():
                st.html('<div class="eyebrow"><span>Power Rankings</span><span class="count">Elo-based · latest</span></div>')
                pr = pd.read_csv(pr_path).sort_values('elo', ascending=False).reset_index(drop=True)
                pr['rank'] = pr.index + 1
                header = ('<div class="pr-row" style="border-bottom-color: var(--border-2); color: var(--text-3); '
                          'font-family: Geist Mono, monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;">'
                          '<div>#</div><div>Chg</div><div></div><div>Team</div>'
                          '<div style="text-align:right;">Elo</div><div style="text-align:right;">Rec</div></div>')
                rows_html = header
                for _, r in pr.iterrows():
                    team = r['team']
                    meta = TEAMS.get(team, {})
                    chg = int(r.get('rank_change', 0)) if pd.notna(r.get('rank_change')) else 0
                    if chg > 0: move_cls, move_txt = 'up', f'▲{chg}'
                    elif chg < 0: move_cls, move_txt = 'down', f'▼{abs(chg)}'
                    else: move_cls, move_txt = 'same', '—'
                    rec = ""  # optional field
                    rows_html += (
                        f'<div class="pr-row">'
                        f'<div class="rank">{int(r["rank"])}</div>'
                        f'<div class="move {move_cls}">{move_txt}</div>'
                        f'<div><img src="{logo_url(team)}" style="width:26px;height:26px;"/></div>'
                        f'<div class="team-name">{meta.get("name", team)}</div>'
                        f'<div class="num">{r["elo"]:.0f}</div>'
                        f'<div class="num rec">{rec}</div>'
                        '</div>'
                    )
                st.html(rows_html)

    # ============================== TAB: Model Backtest ==============================
    with tab_back:
        reports = load_backtest_reports(outputs_dir)
        if not reports:
            st.info("No backtest reports yet.")
            return

        def extract(text, key):
            for line in text.splitlines():
                if line.strip().startswith(key):
                    try: return line.split(':', 1)[1].strip().split()[0]
                    except: return "—"
            return "—"

        elo_bs = extract(reports.get('backtest_elo', ''), "Brier score")
        lgb_bs = extract(reports.get('backtest_winprob', ''), "Brier score")
        margin_mae = extract(reports.get('backtest_margin', ''), "MAE")
        home_mae = extract(reports.get('backtest_scores', ''), "MAE home_score")
        try: delta_sub = f"Delta {float(lgb_bs) - float(elo_bs):+.4f} vs Elo"
        except: delta_sub = ""

        st.html(
            '<div class="kpi-strip">'
            + kpi("Elo Brier", elo_bs, sub="Baseline · 2010-2024")
            + kpi("LightGBM Brier", lgb_bs, sub=delta_sub)
            + kpi("Margin MAE", margin_mae, "pts", sub="vs 11.41 predict-zero")
            + kpi("Home Score MAE", home_mae, "pts", sub="per-team")
            + '</div>'
        )

        # Vegas ATS block
        if 'backtest_vs_vegas' in reports:
            vegas_df = load_vegas_detail(outputs_dir)
            if vegas_df is not None:
                total = len(vegas_df)
                m_pct = vegas_df['ats_correct'].mean() * 100
                v_pct = vegas_df['vegas_ats_correct'].mean() * 100
                disagree_ats = vegas_df[vegas_df['agrees_with_book'] == 0]['ats_correct'].mean() * 100

                st.html('<div class="eyebrow"><span>Against The Spread (ATS)</span><span class="count">vs Vegas closing lines</span></div>')
                st.html(
                    '<div class="kpi-strip">'
                    + kpi("Model ATS", f"{m_pct:.1f}", "%", sub=f"{int(vegas_df['ats_correct'].sum())}-{total - int(vegas_df['ats_correct'].sum())} · Need 52.4%")
                    + kpi("Vegas ATS", f"{v_pct:.1f}", "%", sub="Book's own picks")
                    + kpi("Contrarian ATS", f"{disagree_ats:.1f}", "%", sub="When model disagrees with book")
                    + kpi("Games", f"{total:,}", sub="With closing lines")
                    + '</div>'
                )

        st.html('<div class="eyebrow"><span>Full Backtest Reports</span><span class="count">Walk-forward 2010-2024</span></div>')
        for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin', 'backtest_total',
                     'backtest_scores', 'backtest_vs_vegas']:
            if name in reports:
                title = name.replace('backtest_', '').replace('_', ' ').title()
                with st.expander(title, expanded=(name == 'backtest_elo')):
                    st.code(reports[name], language='text')


if __name__ == "__main__":
    main()
