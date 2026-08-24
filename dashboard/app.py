"""
NFL Predictor Dashboard — Streamlit.

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pickle
from pathlib import Path

# Streamlit only adds the script's directory to sys.path; add project root
# so `from src.teams import ...` resolves on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import streamlit as st
import yaml

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
        --border:    #1F2028;
        --border-2:  #2A2C36;
        --text:      #F2F2F2;
        --text-2:    #A1A1A6;
        --text-3:    #6E6E76;
        --accent:    #E11D48;   /* deep rose, desaturated */
        --accent-2:  #10B981;
        --amber:     #F59E0B;
    }

    html, body, [class*="css"], .stApp, .main {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
        background: var(--bg);
        font-feature-settings: "ss01", "ss02", "cv01", "cv02";
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Hide default chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ---------- Masthead ---------- */
    .masthead {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .masthead-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .masthead-title .accent { color: var(--accent); }
    .masthead-meta {
        text-align: right;
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-3);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        line-height: 1.4;
    }

    /* ---------- Section labels ---------- */
    .eyebrow {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--text-3);
        font-weight: 500;
        margin: 2rem 0 0.75rem 0;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
    }
    .eyebrow .count {
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-2);
    }

    /* ---------- Streamlit widget overrides ---------- */
    .stSelectbox label { color: var(--text-3); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--surface);
        border-color: var(--border);
        border-radius: 4px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        color: var(--text-3);
        letter-spacing: -0.01em;
    }
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: var(--text);
        border-bottom-color: var(--accent);
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

    /* ---------- Game row (editorial, no card) ---------- */
    .game-row {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1.5rem;
        padding: 1.25rem 0;
        border-bottom: 1px solid var(--border);
    }
    .game-row:hover { background: rgba(255,255,255,0.015); }
    .team-side { display: flex; align-items: center; gap: 0.9rem; min-width: 0; }
    .team-side.away { justify-content: flex-end; text-align: right; }
    .team-mark img {
        width: 44px; height: 44px; object-fit: contain;
        filter: drop-shadow(0 0 12px rgba(0,0,0,0.3));
    }
    .team-info { min-width: 0; }
    .team-info .city {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-3);
        font-weight: 500;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .team-info .name {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.02em;
        color: var(--text);
    }
    .team-info .record {
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-3);
        margin-top: 0.2rem;
    }
    .center-scores {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        font-family: 'Geist Mono', monospace;
        font-weight: 600;
    }
    .center-scores .away, .center-scores .home {
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        min-width: 2.4ch;
        text-align: center;
    }
    .center-scores .sep {
        color: var(--text-3);
        font-size: 1.1rem;
        font-weight: 400;
    }
    .center-scores .winner { color: var(--text); }
    .center-scores .loser  { color: var(--text-3); }

    /* Probability meta below */
    .game-meta {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1.5rem;
        padding: 0 0 0.5rem 0;
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-2);
    }
    .game-meta .pct { font-weight: 500; letter-spacing: 0.02em; }
    .game-meta .pct.away { text-align: right; }
    .prob-track {
        width: 100%;
        height: 3px;
        background: var(--border);
        border-radius: 1.5px;
        overflow: hidden;
        display: flex;
    }
    .prob-track .away-fill, .prob-track .home-fill {
        height: 100%;
    }
    .stats-line {
        display: flex; gap: 1.25rem; padding: 0.5rem 0 0 0;
        font-family: 'Geist Mono', monospace; font-size: 0.7rem;
        color: var(--text-3); text-transform: uppercase; letter-spacing: 0.08em;
        justify-content: center;
    }
    .stats-line strong { color: var(--text); font-weight: 500; }

    /* ---------- Standings table ---------- */
    .standings {
        border-top: 1px solid var(--border-2);
    }
    .standings-header {
        display: grid;
        grid-template-columns: 32px 32px 1fr 44px 60px 90px;
        gap: 0.75rem;
        padding: 0.5rem 0.5rem;
        font-family: 'Geist Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-3);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid var(--border);
        font-weight: 500;
    }
    .standings-header .right { text-align: right; }
    .standings-row {
        display: grid;
        grid-template-columns: 32px 32px 1fr 44px 60px 90px;
        gap: 0.75rem;
        padding: 0.6rem 0.5rem;
        align-items: center;
        border-bottom: 1px solid var(--border);
        transition: background 0.15s;
    }
    .standings-row:hover { background: rgba(255,255,255,0.02); }
    .standings-row.playoff { background: linear-gradient(90deg, rgba(225,29,72,0.05) 0%, transparent 30%); }
    .standings-row .rank {
        font-family: 'Geist Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-3);
        font-weight: 500;
    }
    .standings-row .rank.playoff { color: var(--accent); font-weight: 700; }
    .standings-row .team-mark img { width: 24px; height: 24px; object-fit: contain; }
    .standings-row .team-name {
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: -0.01em;
    }
    .standings-row .team-name .city {
        color: var(--text-3);
        font-weight: 400;
        margin-right: 0.35rem;
    }
    .standings-row .num {
        font-family: 'Geist Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 0.85rem;
        text-align: right;
    }
    .standings-row .num.dim { color: var(--text-3); }
    .odds-cell { position: relative; text-align: right; }
    .odds-cell .val {
        font-family: 'Geist Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .odds-cell .bar {
        position: absolute;
        bottom: -8px;
        right: 0;
        height: 2px;
        background: var(--accent);
        opacity: 0.6;
        max-width: 100%;
    }

    /* ---------- KPI strip (no cards; underline the number) ---------- */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2.5rem;
        padding: 1.5rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .kpi .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text-3);
        font-weight: 500;
    }
    .kpi .value {
        font-family: 'Geist Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-top: 0.35rem;
        color: var(--text);
    }
    .kpi .value .unit { font-size: 0.9rem; color: var(--text-3); font-weight: 400; margin-left: 0.15rem; }
    .kpi .sub {
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-2);
        margin-top: 0.3rem;
        letter-spacing: 0.02em;
    }

    /* Backtest code block */
    .stCode pre {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        font-family: 'Geist Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Geist', sans-serif;
        font-size: 0.85rem;
        color: var(--text-2);
        border: none;
        background: transparent;
    }

    /* Small tag */
    .tag {
        display: inline-block;
        font-family: 'Geist Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding: 2px 6px;
        border-radius: 2px;
        background: var(--surface);
        color: var(--text-2);
        border: 1px solid var(--border);
    }
    .tag.live  { background: rgba(16,185,129,0.1); color: var(--accent-2); border-color: rgba(16,185,129,0.2); }
    .tag.final { background: rgba(255,255,255,0.03); color: var(--text-3); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    from collections import defaultdict
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

def predict_week(feats, season, week, winprob, margin, total):
    subset = feats[(feats['season'] == season) & (feats['week'] == week)].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    subset['pred_home_win_prob'] = winprob['model'].predict_proba(
        subset[winprob['features']].fillna(0))[:, 1]
    subset['pred_margin'] = margin['model'].predict(subset[margin['features']].fillna(0))
    subset['pred_total'] = total['model'].predict(subset[total['features']].fillna(0))
    subset['pred_home_score'] = ((subset['pred_margin'] + subset['pred_total']) / 2).round(0).astype(int)
    subset['pred_away_score'] = ((subset['pred_total'] - subset['pred_margin']) / 2).round(0).astype(int)

    return subset[['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team',
                   'home_score', 'away_score',
                   'pred_home_win_prob', 'pred_margin', 'pred_total',
                   'pred_home_score', 'pred_away_score']].sort_values('gameday')


# ------------------------------ components ------------------------------

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
        home_win = hs > as_
        tag = '<span class="tag final">Final</span>'
    else:
        hs, as_ = int(row['pred_home_score']), int(row['pred_away_score'])
        home_win = hs > as_
        tag = '<span class="tag">Projected</span>'

    away_score_cls = 'winner' if not home_win else 'loser'
    home_score_cls = 'winner' if home_win else 'loser'

    home_rec = records.get(home, '0-0')
    away_rec = records.get(away, '0-0')

    home_color = primary_color(home)
    away_color = primary_color(away)

    date_str = str(row['gameday'])[:10] if pd.notna(row.get('gameday')) else ''

    return f"""
    <div class="game-row">
      <div class="team-side away">
        <div class="team-info">
          <div class="city">{away_city}</div>
          <div class="name">{short_name(away)}</div>
          <div class="record">{away_rec}</div>
        </div>
        <div class="team-mark"><img src="{logo_url(away)}" alt="{away}"/></div>
      </div>
      <div class="center-scores">
        <div class="away {away_score_cls}">{as_}</div>
        <div class="sep">·</div>
        <div class="home {home_score_cls}">{hs}</div>
      </div>
      <div class="team-side">
        <div class="team-mark"><img src="{logo_url(home)}" alt="{home}"/></div>
        <div class="team-info">
          <div class="city">{home_city}</div>
          <div class="name">{short_name(home)}</div>
          <div class="record">{home_rec}</div>
        </div>
      </div>
    </div>
    <div class="game-meta">
      <div class="pct away">{short_name(away).upper()} {pa}%</div>
      <div class="prob-track">
        <div class="away-fill" style="width:{pa}%; background:{away_color};"></div>
        <div class="home-fill" style="width:{ph}%; background:{home_color};"></div>
      </div>
      <div class="pct">{short_name(home).upper()} {ph}%</div>
    </div>
    <div class="stats-line">
      {tag}
      <span>SPREAD <strong>{row['pred_margin']:+.1f}</strong></span>
      <span>TOTAL <strong>{row['pred_total']:.1f}</strong></span>
      {f'<span>{date_str}</span>' if date_str else ''}
    </div>
    """


def standings_row_html(rank, row, top_seeds=7):
    team = row['team']
    meta = TEAMS.get(team, {})
    is_playoff = rank <= top_seeds
    odds = float(row['playoff_odds'])
    bar_width = int(odds * 100)

    return f"""
    <div class="standings-row{' playoff' if is_playoff else ''}">
      <div class="rank{' playoff' if is_playoff else ''}">{rank}</div>
      <div class="team-mark"><img src="{logo_url(team)}" alt="{team}"/></div>
      <div class="team-name"><span class="city">{meta.get('city', '')}</span>{meta.get('name', team)}</div>
      <div class="num">{int(row['current_wins'])}</div>
      <div class="num">{row['sim_wins_mean']:.1f}</div>
      <div class="odds-cell">
        <div class="val">{odds*100:.1f}%</div>
        <div class="bar" style="width:{bar_width}%;"></div>
      </div>
    </div>
    """


def kpi(label, value, unit="", sub=""):
    unit_span = f'<span class="unit">{unit}</span>' if unit else ''
    return f"""
    <div class="kpi">
      <div class="label">{label}</div>
      <div class="value">{value}{unit_span}</div>
      <div class="sub">{sub}</div>
    </div>
    """


# ------------------------------ main ------------------------------

def main():
    config = load_config()
    models_dir = config['data']['models_dir']
    outputs_dir = config['data']['outputs_dir']

    feats = load_features()
    schedules = load_schedules()

    # Masthead
    st.markdown(f"""
    <div class="masthead">
      <div class="masthead-title">NFL <span class="accent">Predictor</span></div>
      <div class="masthead-meta">
        Model v1 · 6,223 games trained<br>
        2026 season · Live projections
      </div>
    </div>
    """, unsafe_allow_html=True)

    if feats is None or schedules is None:
        st.error("Data not built. Run `python make_dataset.py` and `python -m src.features.build_features`.")
        return

    tab1, tab2, tab3 = st.tabs(["Weekly Games", "Season Simulator", "Model Backtest"])

    # ------------------------------ Tab 1 ------------------------------
    with tab1:
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            seasons = sorted(feats['season'].unique())
            season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="wk_season")
        with c2:
            weeks = sorted(feats[feats['season'] == season]['week'].unique())
            week = st.selectbox("Week", weeks, key="wk_week")

        winprob_m = load_model(models_dir, 'winprob_model')
        margin_m = load_model(models_dir, 'margin_model')
        total_m = load_model(models_dir, 'total_model')

        if not all([winprob_m, margin_m, total_m]):
            st.warning("Models not trained. Run `python -m src.models.train_winprob_spread` and `python -m src.models.train_score`.")
        else:
            preds = predict_week(feats, season, week, winprob_m, margin_m, total_m)
            records = compute_records(schedules, season, week)
            if len(preds) == 0:
                st.info(f"No games scheduled for {season} Week {week}.")
            else:
                st.markdown(f'<div class="eyebrow"><span>{season} · Week {week}</span><span class="count">{len(preds)} games</span></div>',
                            unsafe_allow_html=True)
                st.markdown("".join(game_row(g, records) for _, g in preds.iterrows()),
                            unsafe_allow_html=True)

    # ------------------------------ Tab 2 ------------------------------
    with tab2:
        sim_files = sorted(Path(outputs_dir).glob("season_sim_*.csv")) if Path(outputs_dir).exists() else []
        if not sim_files:
            st.info("No season simulations found. Run `python -m src.simulate.season_sim --season 2026 --week 1`.")
        else:
            c1, _ = st.columns([1, 4])
            with c1:
                sim_name = st.selectbox("Simulation", [f.name for f in sim_files], index=len(sim_files) - 1)
            df = pd.read_csv(Path(outputs_dir) / sim_name)

            # KPI strip
            top_playoff = df.loc[df['playoff_odds'].idxmax()]
            top_div = df.loc[df['division_win_odds'].idxmax()]
            avg_wins = df['sim_wins_mean'].mean()
            playoff_bids = (df['playoff_odds'] > 0.5).sum()

            kpi_html = f"""
            <div class="kpi-strip">
              {kpi("Simulations", f"10,000", sub="Monte Carlo iterations")}
              {kpi("Playoff Locks (>50%)", str(playoff_bids), sub=f"of {len(df)} teams")}
              {kpi("Highest Playoff Odds", f"{top_playoff['playoff_odds']*100:.1f}", unit="%", sub=team_name(top_playoff['team']))}
              {kpi("Highest Division Odds", f"{top_div['division_win_odds']*100:.1f}", unit="%", sub=team_name(top_div['team']))}
            </div>
            """
            st.markdown(kpi_html, unsafe_allow_html=True)

            # AFC / NFC standings
            afc_col, nfc_col = st.columns(2, gap="large")
            for label, conf_col in [("AFC", afc_col), ("NFC", nfc_col)]:
                with conf_col:
                    st.markdown(f'<div class="eyebrow"><span>{label} Playoff Picture</span><span class="count">Top 7 seed</span></div>',
                                unsafe_allow_html=True)
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
                    rows = "".join(standings_row_html(i + 1, r) for i, r in conf_df.iterrows())
                    st.markdown('<div class="standings">' + header + rows + '</div>',
                                unsafe_allow_html=True)

    # ------------------------------ Tab 3 ------------------------------
    with tab3:
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

        kpi_html = f"""
        <div class="kpi-strip">
          {kpi("Elo Brier", elo_bs, sub="Baseline · 2010-2024")}
          {kpi("LightGBM Brier", lgb_bs, sub=f"Delta {float(lgb_bs)-float(elo_bs):+.4f} vs Elo" if lgb_bs != '—' else "")}
          {kpi("Margin MAE", margin_mae, unit="pts", sub="vs 11.41 predict-zero baseline")}
          {kpi("Home Score MAE", home_mae, unit="pts", sub="per-team point prediction")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown('<div class="eyebrow"><span>Full Backtest Reports</span><span class="count">Walk-forward 2010-2024</span></div>',
                    unsafe_allow_html=True)
        for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin', 'backtest_total', 'backtest_scores']:
            if name in reports:
                title = name.replace('backtest_', '').replace('_', ' ').title()
                with st.expander(title, expanded=(name == 'backtest_elo')):
                    st.code(reports[name], language='text')


if __name__ == "__main__":
    main()
