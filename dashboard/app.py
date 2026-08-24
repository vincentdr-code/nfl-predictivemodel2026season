"""
NFL Predictor Dashboard — Streamlit.

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

# Streamlit only adds the script's directory to sys.path; add project root
# so `from src.teams import ...` resolves on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import yaml
import altair as alt

from src.teams import (
    TEAMS, logo_url, team_name, short_name, primary_color,
)


st.set_page_config(
    page_title="NFL Predictor · 2026",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------ styling ------------------------------

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800;900&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:        #0A0A0B;
        --surface:   #111114;
        --surface-2: #16171C;
        --border:    #1F2028;
        --border-2:  #2A2C36;
        --text:      #F2F2F2;
        --text-2:    #A1A1A6;
        --text-3:    #6E6E76;
        --accent:    #E11D48;
        --accent-dim:#7A0F27;
        --success:   #10B981;
        --warn:      #F59E0B;
        --info:      #3B82F6;
    }

    html, body, [class*="css"], .stApp, .main {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
        background: var(--bg);
        font-feature-settings: "ss01", "ss02", "cv01", "cv02";
    }

    .main .block-container { max-width: 1280px; padding-top: 2.5rem; padding-bottom: 4rem; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* Masthead */
    .masthead {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.5rem; margin-bottom: 2rem;
        display: flex; justify-content: space-between; align-items: flex-end;
    }
    .masthead-title { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }
    .masthead-title .accent { color: var(--accent); }
    .masthead-meta {
        text-align: right; font-family: 'Geist Mono', monospace;
        font-size: 0.7rem; color: var(--text-3);
        text-transform: uppercase; letter-spacing: 0.1em; line-height: 1.4;
    }

    /* Eyebrow labels */
    .eyebrow {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.18em;
        color: var(--text-3); font-weight: 500;
        margin: 2rem 0 0.75rem 0;
        display: flex; align-items: baseline; justify-content: space-between;
    }
    .eyebrow .count { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2); }

    /* Streamlit widget overrides */
    .stSelectbox label { color: var(--text-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--surface); border-color: var(--border); border-radius: 4px;
    }
    .stRadio label { color: var(--text-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .stRadio [role="radiogroup"] { gap: 0.4rem; }
    .stCheckbox label { font-size: 0.85rem; color: var(--text-2); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border: none;
        border-bottom: 2px solid transparent; border-radius: 0;
        padding: 0.75rem 1.25rem; font-weight: 500;
        color: var(--text-3); letter-spacing: -0.01em;
    }
    .stTabs [aria-selected="true"] { background: transparent; color: var(--text); border-bottom-color: var(--accent); }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

    /* Game row */
    .game-row {
        display: grid; grid-template-columns: 1fr auto 1fr;
        align-items: center; gap: 1.5rem;
        padding: 1.25rem 0; border-bottom: 1px solid var(--border);
    }
    .game-row.upset {
        background: linear-gradient(90deg, rgba(245,158,11,0.04) 0%, transparent 40%);
    }
    .game-row.lock {
        background: linear-gradient(90deg, rgba(16,185,129,0.03) 0%, transparent 40%);
    }
    .team-side { display: flex; align-items: center; gap: 0.9rem; min-width: 0; }
    .team-side.away { justify-content: flex-end; text-align: right; }
    .team-mark img { width: 44px; height: 44px; object-fit: contain; filter: drop-shadow(0 0 12px rgba(0,0,0,0.3)); }
    .team-info { min-width: 0; }
    .team-info .city {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--text-3); font-weight: 500; line-height: 1; margin-bottom: 0.2rem;
    }
    .team-info .name {
        font-size: 1.1rem; font-weight: 700; line-height: 1;
        letter-spacing: -0.02em; color: var(--text);
    }
    .team-info .record {
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); margin-top: 0.2rem;
    }
    .center-scores {
        display: flex; align-items: baseline; gap: 0.75rem;
        font-family: 'Geist Mono', monospace; font-weight: 600;
    }
    .center-scores .away, .center-scores .home {
        font-size: 1.75rem; letter-spacing: -0.02em;
        min-width: 2.4ch; text-align: center;
    }
    .center-scores .sep { color: var(--text-3); font-size: 1.1rem; font-weight: 400; }
    .center-scores .winner { color: var(--text); }
    .center-scores .loser  { color: var(--text-3); }

    /* Probability bar under game */
    .game-meta {
        display: grid; grid-template-columns: 1fr auto 1fr;
        align-items: center; gap: 1.5rem; padding: 0 0 0.5rem 0;
        font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2);
    }
    .game-meta .pct { font-weight: 500; letter-spacing: 0.02em; }
    .game-meta .pct.away { text-align: right; }
    .prob-track {
        width: 100%; height: 3px; background: var(--border);
        border-radius: 1.5px; overflow: hidden; display: flex;
    }
    .prob-track .away-fill, .prob-track .home-fill { height: 100%; }
    .stats-line {
        display: flex; gap: 1rem; padding: 0.5rem 0 0 0;
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.08em;
        justify-content: center; flex-wrap: wrap;
    }
    .stats-line strong { color: var(--text); font-weight: 500; }

    /* Confidence tag */
    .tag {
        display: inline-block; font-family: 'Geist Mono', monospace;
        font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.12em;
        padding: 2px 6px; border-radius: 2px;
        background: var(--surface); color: var(--text-2); border: 1px solid var(--border);
    }
    .tag.lock    { background: rgba(16,185,129,0.12); color: var(--success); border-color: rgba(16,185,129,0.25); }
    .tag.strong  { background: rgba(59,130,246,0.10); color: var(--info);    border-color: rgba(59,130,246,0.22); }
    .tag.tossup  { background: rgba(245,158,11,0.10); color: var(--warn);    border-color: rgba(245,158,11,0.22); }
    .tag.upset   { background: rgba(225,29,72,0.10);  color: var(--accent);  border-color: rgba(225,29,72,0.25); }
    .tag.final   { background: rgba(255,255,255,0.03); color: var(--text-3); }

    /* Standings */
    .standings { border-top: 1px solid var(--border-2); }
    .standings-header {
        display: grid; grid-template-columns: 32px 32px 1fr 44px 60px 100px;
        gap: 0.75rem; padding: 0.5rem;
        font-family: 'Geist Mono', monospace; font-size: 0.65rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em;
        border-bottom: 1px solid var(--border); font-weight: 500;
    }
    .standings-header .right { text-align: right; }
    .standings-row {
        display: grid; grid-template-columns: 32px 32px 1fr 44px 60px 100px;
        gap: 0.75rem; padding: 0.65rem 0.5rem;
        align-items: center; border-bottom: 1px solid var(--border);
        transition: background 0.15s;
    }
    .standings-row:hover { background: rgba(255,255,255,0.02); }
    .standings-row.playoff { background: linear-gradient(90deg, rgba(225,29,72,0.05) 0%, transparent 30%); }
    .standings-row.focus   { background: linear-gradient(90deg, rgba(59,130,246,0.10) 0%, rgba(59,130,246,0.02) 60%, transparent 100%);
                             border-left: 3px solid var(--info); padding-left: calc(0.5rem - 3px); }
    .standings-row .rank {
        font-family: 'Geist Mono', monospace; font-size: 0.75rem;
        color: var(--text-3); font-weight: 500;
    }
    .standings-row .rank.playoff { color: var(--accent); font-weight: 700; }
    .standings-row .team-mark img { width: 24px; height: 24px; object-fit: contain; }
    .standings-row .team-name {
        font-size: 0.95rem; font-weight: 500; letter-spacing: -0.01em;
    }
    .standings-row .team-name .city { color: var(--text-3); font-weight: 400; margin-right: 0.35rem; }
    .standings-row .num {
        font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums;
        font-size: 0.85rem; text-align: right;
    }
    .odds-cell { position: relative; text-align: right; padding-bottom: 6px; }
    .odds-cell .val { font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums;
                      font-weight: 600; font-size: 0.9rem; }
    .odds-cell .bar { position: absolute; bottom: 0; right: 0; height: 2px;
                      background: var(--accent); opacity: 0.6; max-width: 100%; }
    .odds-cell .bar.dim { background: var(--text-3); opacity: 0.4; }

    /* KPI strip */
    .kpi-strip {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 2.5rem; padding: 1.5rem 0 2rem;
        border-bottom: 1px solid var(--border); margin-bottom: 1rem;
    }
    .kpi .label {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em;
        color: var(--text-3); font-weight: 500;
    }
    .kpi .value {
        font-family: 'Geist Mono', monospace; font-size: 1.9rem; font-weight: 700;
        letter-spacing: -0.02em; line-height: 1.1;
        margin-top: 0.35rem; color: var(--text);
    }
    .kpi .value .unit { font-size: 0.9rem; color: var(--text-3); font-weight: 400; margin-left: 0.15rem; }
    .kpi .sub { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--text-2);
                margin-top: 0.3rem; letter-spacing: 0.02em; }

    /* Team focus header */
    .team-header {
        display: grid; grid-template-columns: 100px 1fr auto;
        gap: 1.5rem; align-items: center;
        padding: 1.5rem 0; border-bottom: 1px solid var(--border);
    }
    .team-header .mark img { width: 96px; height: 96px; object-fit: contain; }
    .team-header .name-block .city {
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em;
        color: var(--text-3); font-weight: 500;
    }
    .team-header .name-block .name {
        font-size: 2.25rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1;
    }
    .team-header .name-block .div {
        font-family: 'Geist Mono', monospace; font-size: 0.75rem;
        color: var(--text-2); margin-top: 0.4rem; letter-spacing: 0.05em;
    }
    .team-header .status-badge {
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        padding: 0.5rem 0.9rem; border-radius: 3px;
        text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600;
        border: 1px solid;
    }
    .status-clinched { background: rgba(16,185,129,0.15); color: var(--success); border-color: rgba(16,185,129,0.35); }
    .status-likely   { background: rgba(16,185,129,0.08); color: var(--success); border-color: rgba(16,185,129,0.2); }
    .status-hunt     { background: rgba(59,130,246,0.10); color: var(--info);    border-color: rgba(59,130,246,0.22); }
    .status-bubble   { background: rgba(245,158,11,0.10); color: var(--warn);    border-color: rgba(245,158,11,0.22); }
    .status-longshot { background: rgba(225,29,72,0.08);  color: var(--accent);  border-color: rgba(225,29,72,0.2); }
    .status-out      { background: rgba(255,255,255,0.03); color: var(--text-3); border-color: var(--border); }

    /* Playoff race meter */
    .race-meter {
        margin: 1rem 0 2rem;
        padding: 1rem 0;
    }
    .race-label { display: flex; justify-content: space-between;
                  font-family: 'Geist Mono', monospace; font-size: 0.7rem;
                  color: var(--text-3); text-transform: uppercase;
                  letter-spacing: 0.1em; margin-bottom: 0.5rem; }
    .race-track {
        position: relative; height: 24px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 3px; overflow: visible;
    }
    .race-fill {
        height: 100%; border-radius: 3px 0 0 3px;
        transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .race-marker {
        position: absolute; top: -6px; bottom: -6px; width: 2px;
        background: var(--text-3); opacity: 0.6;
    }
    .race-marker .mlabel {
        position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
        margin-top: 6px; font-family: 'Geist Mono', monospace;
        font-size: 0.6rem; color: var(--text-3);
        text-transform: uppercase; letter-spacing: 0.1em; white-space: nowrap;
    }

    /* Schedule mini rows */
    .schedule-row {
        display: grid; grid-template-columns: 32px 32px 1fr 90px 70px 60px;
        gap: 0.75rem; padding: 0.7rem 0.5rem;
        align-items: center; border-bottom: 1px solid var(--border);
    }
    .schedule-row .wk {
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); font-weight: 500;
    }
    .schedule-row .side {
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); text-align: center;
    }
    .schedule-row .opp {
        font-size: 0.9rem; font-weight: 500; letter-spacing: -0.01em;
    }
    .schedule-row .win-bar-cell {
        position: relative; height: 20px;
        background: var(--border); border-radius: 2px; overflow: hidden;
    }
    .schedule-row .win-bar {
        height: 100%; border-radius: 2px;
    }
    .schedule-row .win-pct, .schedule-row .spread {
        font-family: 'Geist Mono', monospace; font-size: 0.8rem;
        font-variant-numeric: tabular-nums; text-align: right;
    }
    .schedule-row .win-pct { font-weight: 600; }
    .schedule-row .spread { color: var(--text-3); }
    .schedule-row.played .side, .schedule-row.played .opp { color: var(--text-3); }

    /* Backtest code */
    .stCode pre {
        background: var(--surface) !important; border: 1px solid var(--border) !important;
        border-radius: 4px !important; font-family: 'Geist Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Filter bar */
    .filter-bar {
        display: flex; gap: 1rem; align-items: flex-end;
        padding: 0.5rem 0 1rem;
    }

    /* Empty */
    .empty-state {
        text-align: center; padding: 3rem 1rem;
        color: var(--text-3); font-family: 'Geist Mono', monospace;
        font-size: 0.85rem; letter-spacing: 0.05em;
    }
</style>
"""
st.html(CUSTOM_CSS)


# ------------------------------ data loading ------------------------------

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
                 'backtest_total', 'backtest_scores']:
        p = Path(outputs_dir) / f"{name}.txt"
        if p.exists():
            reports[name] = p.read_text(encoding='utf-8', errors='replace')
    return reports


@st.cache_resource
def load_model(models_dir, name):
    p = Path(models_dir) / f"{name}.pkl"
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def compute_records(schedules, season, upto_week):
    """Return {team: 'W-L'} through weeks < upto_week in season."""
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


# ------------------------------ prediction ------------------------------

@st.cache_data
def predict_all_games(_feats, season, _winprob_dict, _margin_dict, _total_dict):
    """
    Cache-key hack: _feats is unhashable so we pass it as an underscore arg.
    We rely on season being cached per call.
    """
    subset = _feats[_feats['season'] == season].copy()
    subset['pred_home_win_prob'] = _winprob_dict['model'].predict_proba(
        subset[_winprob_dict['features']].fillna(0))[:, 1]
    subset['pred_margin'] = _margin_dict['model'].predict(subset[_margin_dict['features']].fillna(0))
    subset['pred_total'] = _total_dict['model'].predict(subset[_total_dict['features']].fillna(0))
    subset['pred_home_score'] = ((subset['pred_margin'] + subset['pred_total']) / 2).round(0).astype(int)
    subset['pred_away_score'] = ((subset['pred_total'] - subset['pred_margin']) / 2).round(0).astype(int)
    return subset[['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team',
                   'home_score', 'away_score',
                   'pred_home_win_prob', 'pred_margin', 'pred_total',
                   'pred_home_score', 'pred_away_score']].sort_values(['week', 'gameday'])


# ------------------------------ components ------------------------------

def confidence_tag(p_home):
    """Return (tag_class, tag_text) for a home win prob."""
    p = max(p_home, 1 - p_home)
    if p >= 0.75:
        return "lock", "Lock"
    if p >= 0.63:
        return "strong", "Strong"
    if p <= 0.55:
        return "tossup", "Toss-Up"
    return "", "Lean"


def game_row(row, records):
    home, away = row['home_team'], row['away_team']
    home_meta = TEAMS.get(home, {})
    away_meta = TEAMS.get(away, {})
    home_city = home_meta.get('city', home)
    away_city = away_meta.get('city', away)

    p_home = float(row['pred_home_win_prob'])
    p_away = 1 - p_home
    ph = int(round(p_home * 100))
    pa = 100 - ph

    played = pd.notna(row.get('home_score'))
    if played:
        hs, as_ = int(row['home_score']), int(row['away_score'])
        home_win_display = hs > as_
        tag_html = '<span class="tag final">Final</span>'
        row_cls = ''
    else:
        hs, as_ = int(row['pred_home_score']), int(row['pred_away_score'])
        home_win_display = hs > as_
        conf_cls, conf_text = confidence_tag(p_home)
        tag_html = f'<span class="tag {conf_cls}">{conf_text}</span>'
        # upset/lock highlighting
        if conf_cls == "lock":
            row_cls = 'lock'
        elif conf_cls == "tossup":
            row_cls = 'upset'
        else:
            row_cls = ''

    away_score_cls = 'winner' if not home_win_display else 'loser'
    home_score_cls = 'winner' if home_win_display else 'loser'

    home_rec = records.get(home, '0-0')
    away_rec = records.get(away, '0-0')
    home_color = primary_color(home)
    away_color = primary_color(away)
    date_str = str(row['gameday'])[:10] if pd.notna(row.get('gameday')) else ''

    date_span = f'<span>{date_str}</span>' if date_str else ''
    return (
        f'<div class="game-row {row_cls}">'
          '<div class="team-side away">'
            f'<div class="team-info"><div class="city">{away_city}</div>'
            f'<div class="name">{short_name(away)}</div>'
            f'<div class="record">{away_rec}</div></div>'
            f'<div class="team-mark"><img src="{logo_url(away)}" alt="{away}"/></div>'
          '</div>'
          '<div class="center-scores">'
            f'<div class="away {away_score_cls}">{as_}</div>'
            '<div class="sep">·</div>'
            f'<div class="home {home_score_cls}">{hs}</div>'
          '</div>'
          '<div class="team-side">'
            f'<div class="team-mark"><img src="{logo_url(home)}" alt="{home}"/></div>'
            f'<div class="team-info"><div class="city">{home_city}</div>'
            f'<div class="name">{short_name(home)}</div>'
            f'<div class="record">{home_rec}</div></div>'
          '</div>'
        '</div>'
        '<div class="game-meta">'
          f'<div class="pct away">{short_name(away).upper()} {pa}%</div>'
          '<div class="prob-track">'
            f'<div class="away-fill" style="width:{pa}%; background:{away_color};"></div>'
            f'<div class="home-fill" style="width:{ph}%; background:{home_color};"></div>'
          '</div>'
          f'<div class="pct">{short_name(home).upper()} {ph}%</div>'
        '</div>'
        '<div class="stats-line">'
          f'{tag_html}'
          f'<span>SPREAD <strong>{row["pred_margin"]:+.1f}</strong></span>'
          f'<span>TOTAL <strong>{row["pred_total"]:.1f}</strong></span>'
          f'{date_span}'
        '</div>'
    )


def standings_row_html(rank, row, focus_team=None, top_seeds=7):
    team = row['team']
    meta = TEAMS.get(team, {})
    is_playoff = rank <= top_seeds
    is_focus = (team == focus_team)
    odds = float(row['playoff_odds'])
    bar_width = int(odds * 100)
    div_odds = float(row['division_win_odds'])
    div_bar = int(div_odds * 100)

    classes = ['standings-row']
    if is_playoff: classes.append('playoff')
    if is_focus:   classes.append('focus')

    rank_cls = ' playoff' if is_playoff else ''
    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="rank{rank_cls}">{rank}</div>'
        f'<div class="team-mark"><img src="{logo_url(team)}" alt="{team}"/></div>'
        f'<div class="team-name"><span class="city">{meta.get("city", "")}</span>'
        f'{meta.get("name", team)}</div>'
        f'<div class="num">{int(row["current_wins"])}</div>'
        f'<div class="num">{row["sim_wins_mean"]:.1f}</div>'
        f'<div class="odds-cell"><div class="val">{odds*100:.1f}%</div>'
        f'<div class="bar" style="width:{bar_width}%;"></div></div>'
        f'</div>'
    )


def kpi(label, value, unit="", sub=""):
    unit_span = f'<span class="unit">{unit}</span>' if unit else ''
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}{unit_span}</div>'
            f'<div class="sub">{sub}</div></div>')


def playoff_status(odds):
    """Return (css_class, label) based on playoff odds."""
    if odds >= 0.95:  return "status-clinched", "Virtual Lock"
    if odds >= 0.75:  return "status-likely",   "Likely"
    if odds >= 0.50:  return "status-hunt",     "In The Hunt"
    if odds >= 0.20:  return "status-bubble",   "Bubble"
    if odds >= 0.05:  return "status-longshot", "Long Shot"
    return "status-out", "On The Outside"


def race_meter(odds, playoff_line=0.5):
    """Playoff race horizontal bar with a 50% marker."""
    pct = int(round(odds * 100))
    color = "var(--success)" if odds >= 0.75 else "var(--warn)" if odds >= 0.5 else "var(--accent)"
    return (
        '<div class="race-meter">'
        '<div class="race-label"><span>Playoff Race</span>'
        f'<span>{pct}% odds</span></div>'
        '<div class="race-track">'
        f'<div class="race-fill" style="width:{pct}%; background:{color};"></div>'
        '<div class="race-marker" style="left:50%;"><span class="mlabel">50% cutoff</span></div>'
        '<div class="race-marker" style="left:75%;"><span class="mlabel">75% likely</span></div>'
        '</div></div>'
    )


# ------------------------------ main ------------------------------

def main():
    config = load_config()
    models_dir = config['data']['models_dir']
    outputs_dir = config['data']['outputs_dir']

    feats = load_features()
    schedules = load_schedules()

    st.html(
        '<div class="masthead">'
        '<div class="masthead-title">NFL <span class="accent">Predictor</span></div>'
        '<div class="masthead-meta">Model v1 · 6,495 games trained<br>'
        '2026 season · Live projections</div>'
        '</div>'
    )

    if feats is None or schedules is None:
        st.error("Data not built. Run `python make_dataset.py` and `python -m src.features.build_features`.")
        return

    winprob_m = load_model(models_dir, 'winprob_model')
    margin_m  = load_model(models_dir, 'margin_model')
    total_m   = load_model(models_dir, 'total_model')

    if not all([winprob_m, margin_m, total_m]):
        st.warning("Models not trained. Run `python -m src.models.train_winprob_spread` and `python -m src.models.train_score`.")
        return

    tab_games, tab_focus, tab_sim, tab_back = st.tabs(
        ["Weekly Games", "Team Focus", "Season Simulator", "Model Backtest"]
    )

    # ============================== Tab: Weekly Games ==============================
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
            sort_mode = st.selectbox("Sort by", ["Kickoff Date", "Confidence (High)", "Confidence (Low)", "Upset Risk"], key="wk_sort")

        preds = predict_all_games(feats, season, winprob_m, margin_m, total_m)
        preds = preds[preds['week'] == week]

        if team_filter != "All Teams":
            preds = preds[(preds['home_team'] == team_filter) | (preds['away_team'] == team_filter)]

        # Sort
        if sort_mode == "Confidence (High)":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: max(p, 1-p))).sort_values('_c', ascending=False)
        elif sort_mode == "Confidence (Low)":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: max(p, 1-p))).sort_values('_c', ascending=True)
        elif sort_mode == "Upset Risk":
            preds = preds.assign(_c=preds['pred_home_win_prob'].apply(lambda p: min(p, 1-p))).sort_values('_c', ascending=False)

        records = compute_records(schedules, season, week)
        if len(preds) == 0:
            st.html('<div class="empty-state">No games match the current filters.</div>')
        else:
            filter_desc = "" if team_filter == "All Teams" else f" · {short_name(team_filter)} only"
            st.html(f'<div class="eyebrow"><span>{season} · Week {week}{filter_desc}</span>'
                    f'<span class="count">{len(preds)} games</span></div>')
            st.html("".join(game_row(g, records) for _, g in preds.iterrows()))

    # ============================== Tab: Team Focus ==============================
    with tab_focus:
        # Team selector row
        c1, c2, _ = st.columns([1.5, 1.2, 4])
        with c1:
            teams_sorted = sorted(TEAMS.keys())
            default_idx = teams_sorted.index("BUF") if "BUF" in teams_sorted else 0
            focus_team = st.selectbox("Team", teams_sorted, index=default_idx,
                                      format_func=lambda t: f"{t} — {team_name(t)}", key="focus_team")
        with c2:
            focus_season = st.selectbox("Season", sorted(feats['season'].unique()),
                                        index=len(feats['season'].unique()) - 1, key="focus_season")

        # Load latest sim for that season
        sim_files = sorted(Path(outputs_dir).glob(f"season_sim_{focus_season}_*.csv"))
        sim_row = None
        if sim_files:
            sim_df = pd.read_csv(sim_files[-1])
            match = sim_df[sim_df['team'] == focus_team]
            if len(match) > 0:
                sim_row = match.iloc[0]

        meta = TEAMS.get(focus_team, {})
        div_name = f"{meta.get('conf', '')} {meta.get('div', '')}".strip()

        if sim_row is not None:
            odds = float(sim_row['playoff_odds'])
            status_cls, status_label = playoff_status(odds)
        else:
            status_cls, status_label = "status-out", "No Sim Data"

        st.html(
            f'<div class="team-header">'
            f'<div class="mark"><img src="{logo_url(focus_team)}" alt="{focus_team}"/></div>'
            f'<div class="name-block"><div class="city">{meta.get("city", "")}</div>'
            f'<div class="name">{meta.get("name", focus_team)}</div>'
            f'<div class="div">{div_name}</div></div>'
            f'<div class="status-badge {status_cls}">{status_label}</div>'
            f'</div>'
        )

        if sim_row is not None:
            # KPI strip for the team
            conf_teams = sim_df[sim_df['conference'] == meta.get('conf', '')].sort_values('playoff_odds', ascending=False).reset_index(drop=True)
            conf_rank = int(conf_teams.index[conf_teams['team'] == focus_team][0]) + 1

            kpi_html = (
                '<div class="kpi-strip">'
                + kpi("Playoff Odds", f"{odds*100:.1f}", "%", sub=f"AFC/NFC seed #{conf_rank}")
                + kpi("Division Odds", f"{float(sim_row['division_win_odds'])*100:.1f}", "%",
                      sub=f"{meta.get('conf', '')} {meta.get('div', '')}")
                + kpi("Projected Wins", f"{sim_row['sim_wins_mean']:.1f}",
                      sub=f"Range: {sim_row['sim_wins_p10']:.0f}–{sim_row['sim_wins_p90']:.0f}")
                + kpi("Current Wins", f"{int(sim_row['current_wins'])}", sub="Regular season")
                + '</div>'
            )
            st.html(kpi_html)

            # Race meter
            st.html(race_meter(odds))

        # Team's schedule with predictions
        all_preds = predict_all_games(feats, focus_season, winprob_m, margin_m, total_m)
        team_games = all_preds[(all_preds['home_team'] == focus_team) | (all_preds['away_team'] == focus_team)].copy()

        if len(team_games) > 0:
            st.html('<div class="eyebrow"><span>Full Schedule</span><span class="count">Predictions + Actuals</span></div>')

            rows_html = ""
            for _, g in team_games.iterrows():
                is_home = g['home_team'] == focus_team
                opp = g['away_team'] if is_home else g['home_team']
                side_indicator = "vs" if is_home else "at"
                team_win_prob = g['pred_home_win_prob'] if is_home else (1 - g['pred_home_win_prob'])
                team_win_pct = int(round(team_win_prob * 100))
                spread_for_team = g['pred_margin'] if is_home else -g['pred_margin']

                played = pd.notna(g.get('home_score'))
                if played:
                    my_score = int(g['home_score']) if is_home else int(g['away_score'])
                    opp_score = int(g['away_score']) if is_home else int(g['home_score'])
                    result_html = f'<span style="color:{"var(--success)" if my_score > opp_score else "var(--accent)"};font-weight:600;">{ "W" if my_score > opp_score else "L"} {my_score}-{opp_score}</span>'
                    row_class = "played"
                else:
                    result_html = f'<span class="win-pct">{team_win_pct}%</span>'
                    row_class = ""

                bar_color = primary_color(focus_team) if team_win_prob >= 0.5 else primary_color(opp)

                rows_html += f"""
                <div class="schedule-row {row_class}">
                  <div class="wk">W{int(g['week'])}</div>
                  <div class="team-mark"><img src="{logo_url(opp)}" alt="{opp}"/></div>
                  <div class="opp">
                    <span class="side">{side_indicator}</span> {team_name(opp)}
                  </div>
                  <div class="win-bar-cell">
                    <div class="win-bar" style="width:{team_win_pct}%; background:{bar_color};"></div>
                  </div>
                  <div class="win-pct">{result_html if played else f'{team_win_pct}%'}</div>
                  <div class="spread">{spread_for_team:+.1f}</div>
                </div>
                """
            st.html(rows_html)

            # Wins distribution chart (from sim if available)
            if sim_row is not None:
                st.html('<div class="eyebrow"><span>Projected Wins Range</span><span class="count">10th–90th percentile</span></div>')
                # Simple horizontal range visualization via Altair
                range_data = pd.DataFrame([{
                    'label': team_name(focus_team),
                    'mean': sim_row['sim_wins_mean'],
                    'low':  sim_row['sim_wins_p10'],
                    'high': sim_row['sim_wins_p90'],
                }])
                base = alt.Chart(range_data).encode(y=alt.Y('label:N', axis=None))
                bar = base.mark_bar(color='#E11D48', opacity=0.35, height=20).encode(
                    x=alt.X('low:Q', title='Wins', scale=alt.Scale(domain=[0, 17])),
                    x2='high:Q'
                )
                point = base.mark_point(color='#F2F2F2', size=200, filled=True, shape='diamond').encode(
                    x='mean:Q'
                )
                st.altair_chart(bar + point, use_container_width=True)

    # ============================== Tab: Season Simulator ==============================
    with tab_sim:
        sim_files = sorted(Path(outputs_dir).glob("season_sim_*.csv")) if Path(outputs_dir).exists() else []
        if not sim_files:
            st.info("No season simulations found. Run `python -m src.simulate.season_sim --season 2026 --week 1`.")
        else:
            c1, c2, _ = st.columns([1.5, 1.5, 3])
            with c1:
                sim_name = st.selectbox("Simulation", [f.name for f in sim_files], index=len(sim_files) - 1, key="sim_file")
            with c2:
                team_opts = ["None"] + sorted(TEAMS.keys())
                sim_focus = st.selectbox("Highlight Team", team_opts, index=0,
                                         format_func=lambda t: t if t == "None" else f"{t} — {team_name(t)}",
                                         key="sim_focus")
            focus_team_sim = None if sim_focus == "None" else sim_focus
            df = pd.read_csv(Path(outputs_dir) / sim_name)

            top_playoff = df.loc[df['playoff_odds'].idxmax()]
            top_div = df.loc[df['division_win_odds'].idxmax()]
            playoff_bids = (df['playoff_odds'] > 0.5).sum()
            virtual_locks = (df['playoff_odds'] > 0.85).sum()

            kpi_html = (
                '<div class="kpi-strip">'
                + kpi("Simulations", "10,000", sub="Monte Carlo iterations")
                + kpi("Virtual Locks", str(virtual_locks), sub=f"teams above 85% odds")
                + kpi("Top Playoff Odds", f"{top_playoff['playoff_odds']*100:.1f}", "%",
                      sub=team_name(top_playoff['team']))
                + kpi("Top Division Odds", f"{top_div['division_win_odds']*100:.1f}", "%",
                      sub=team_name(top_div['team']))
                + '</div>'
            )
            st.html(kpi_html)

            afc_col, nfc_col = st.columns(2, gap="large")
            for label, conf_col in [("AFC", afc_col), ("NFC", nfc_col)]:
                with conf_col:
                    st.html(f'<div class="eyebrow"><span>{label} Playoff Picture</span><span class="count">Top 7 seed</span></div>')
                    conf_df = (df[df['conference'] == label]
                               .sort_values('playoff_odds', ascending=False)
                               .reset_index(drop=True))
                    header = """
                    <div class="standings-header">
                      <div>#</div><div></div><div>Team</div>
                      <div class="right">W</div>
                      <div class="right">Proj</div>
                      <div class="right">Playoff%</div>
                    </div>
                    """
                    rows = "".join(standings_row_html(i + 1, r, focus_team=focus_team_sim) for i, r in conf_df.iterrows())
                    st.html('<div class="standings">' + header + rows + '</div>')

            # Playoff odds distribution chart
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

    # ============================== Tab: Model Backtest ==============================
    with tab_back:
        reports = load_backtest_reports(outputs_dir)
        if not reports:
            st.info("No backtest reports yet. Run the training scripts.")
            return

        def extract(text, key):
            for line in text.splitlines():
                if line.strip().startswith(key):
                    try:
                        return line.split(':', 1)[1].strip().split()[0]
                    except (IndexError, ValueError):
                        return "—"
            return "—"

        elo_bs = extract(reports.get('backtest_elo', ''), "Brier score")
        lgb_bs = extract(reports.get('backtest_winprob', ''), "Brier score")
        margin_mae = extract(reports.get('backtest_margin', ''), "MAE")
        home_mae = extract(reports.get('backtest_scores', ''), "MAE home_score")

        try:
            delta_sub = f"Delta {float(lgb_bs) - float(elo_bs):+.4f} vs Elo"
        except (ValueError, TypeError):
            delta_sub = ""

        kpi_html = (
            '<div class="kpi-strip">'
            + kpi("Elo Brier", elo_bs, sub="Baseline · 2010-2024")
            + kpi("LightGBM Brier", lgb_bs, sub=delta_sub)
            + kpi("Margin MAE", margin_mae, "pts", sub="vs 11.41 predict-zero baseline")
            + kpi("Home Score MAE", home_mae, "pts", sub="per-team point prediction")
            + '</div>'
        )
        st.html(kpi_html)

        st.html('<div class="eyebrow"><span>Full Backtest Reports</span><span class="count">Walk-forward 2010-2024</span></div>')
        for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin', 'backtest_total', 'backtest_scores']:
            if name in reports:
                title = name.replace('backtest_', '').replace('_', ' ').title()
                with st.expander(title, expanded=(name == 'backtest_elo')):
                    st.code(reports[name], language='text')


if __name__ == "__main__":
    main()
