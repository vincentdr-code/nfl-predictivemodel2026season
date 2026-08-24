"""
NFL Predictor Dashboard — Streamlit.

Run: streamlit run dashboard/app.py
"""
import os
import sys
import pickle
from pathlib import Path

# Streamlit puts the script's directory on sys.path; add project root so
# `from src.teams import ...` resolves on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # so relative paths in config.yaml resolve

import pandas as pd
import streamlit as st
import yaml

from src.teams import (
    TEAMS, logo_url, team_name, short_name,
    primary_color, secondary_color,
)


st.set_page_config(
    page_title="NFL Predictor 2026",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Global styling ----------

CUSTOM_CSS = """
<style>
    /* Tighten default padding */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Section headers */
    .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #8b95a5;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* Game card */
    .game-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #151A25 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.06);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .game-card:hover { border-color: rgba(0,230,118,0.35); transform: translateY(-1px); }

    .matchup-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    .team-block { display: flex; align-items: center; gap: 0.75rem; flex: 1; min-width: 0; }
    .team-block.away { justify-content: flex-end; text-align: right; }
    .team-logo { width: 48px; height: 48px; object-fit: contain; flex-shrink: 0; }
    .team-name { font-weight: 700; font-size: 1rem; line-height: 1.1; }
    .team-city { font-size: 0.72rem; color: #8b95a5; text-transform: uppercase; letter-spacing: 0.05em; }
    .score { font-size: 2rem; font-weight: 800; font-variant-numeric: tabular-nums; color: #F5F7FA; letter-spacing: -0.02em; }
    .vs-divider { color: #4a5566; font-size: 0.75rem; font-weight: 600; padding: 0 0.5rem; }

    /* Probability bar */
    .prob-bar-container { margin-top: 1rem; }
    .prob-bar-bg { height: 8px; border-radius: 4px; overflow: hidden; display: flex; background: #2a3140; }
    .prob-bar-away { height: 100%; transition: width 0.3s ease; }
    .prob-bar-home { height: 100%; transition: width 0.3s ease; }
    .prob-labels { display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 0.4rem; color: #8b95a5; font-variant-numeric: tabular-nums; }
    .prob-labels .home-pct { color: #F5F7FA; font-weight: 600; }

    /* Meta line */
    .game-meta {
        display: flex; gap: 1rem; margin-top: 0.75rem;
        font-size: 0.75rem; color: #8b95a5; font-variant-numeric: tabular-nums;
    }
    .game-meta strong { color: #ccd3de; font-weight: 600; }

    /* Standings row */
    .standings-row {
        display: grid;
        grid-template-columns: 40px 32px 1fr 60px 80px 100px;
        gap: 0.75rem;
        align-items: center;
        padding: 0.6rem 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-variant-numeric: tabular-nums;
    }
    .standings-header {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: #8b95a5; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .standings-row.playoff { border-left: 3px solid #00E676; padding-left: 0.5rem; }
    .rank { color: #8b95a5; font-size: 0.85rem; font-weight: 600; }
    .team-cell { display: flex; align-items: center; gap: 0.5rem; }
    .team-cell img { width: 24px; height: 24px; object-fit: contain; }
    .team-cell .name { font-weight: 600; font-size: 0.9rem; }
    .odds-value { font-weight: 700; font-size: 0.9rem; color: #00E676; }
    .odds-value.low { color: #8b95a5; }
    .odds-value.mid { color: #FFC62F; }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #151A25 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    .metric-label { font-size: 0.7rem; color: #8b95a5; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
    .metric-value { font-size: 1.75rem; font-weight: 800; margin-top: 0.25rem; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .metric-sub { font-size: 0.75rem; color: #8b95a5; margin-top: 0.15rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #1A1F2E; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Data loading ----------

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


# ---------- Prediction helpers ----------

def predict_week(feats, season, week, winprob, margin, total):
    subset = feats[(feats['season'] == season) & (feats['week'] == week)].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    subset['pred_home_win_prob'] = winprob['model'].predict_proba(
        subset[winprob['features']].fillna(0)
    )[:, 1]
    subset['pred_margin'] = margin['model'].predict(subset[margin['features']].fillna(0))
    subset['pred_total'] = total['model'].predict(subset[total['features']].fillna(0))
    subset['pred_home_score'] = ((subset['pred_margin'] + subset['pred_total']) / 2).round(1)
    subset['pred_away_score'] = ((subset['pred_total'] - subset['pred_margin']) / 2).round(1)

    keep = ['game_id', 'season', 'week', 'gameday', 'home_team', 'away_team',
            'home_score', 'away_score',
            'pred_home_win_prob', 'pred_margin', 'pred_total',
            'pred_home_score', 'pred_away_score']
    return subset[keep]


# ---------- UI components ----------

def game_card(row):
    """Render a single game as a rich HTML card."""
    home = row['home_team']
    away = row['away_team']
    p_home = row['pred_home_win_prob']
    p_away = 1 - p_home
    ph_pct = int(round(p_home * 100))
    pa_pct = 100 - ph_pct

    home_played = pd.notna(row.get('home_score'))
    if home_played:
        h_score = int(row['home_score'])
        a_score = int(row['away_score'])
    else:
        h_score = row['pred_home_score']
        a_score = row['pred_away_score']

    home_color = primary_color(home)
    away_color = primary_color(away)

    date_str = str(row['gameday'])[:10] if pd.notna(row.get('gameday')) else ''
    label = "Final" if home_played else "Predicted"

    return f"""
    <div class="game-card">
      <div class="matchup-row">
        <div class="team-block away">
          <div>
            <div class="team-name">{short_name(away)}</div>
            <div class="team-city">{TEAMS.get(away, {{}}).get('city', away)}</div>
          </div>
          <img class="team-logo" src="{logo_url(away)}" alt="{away}"/>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <div class="score" style="color:{away_color if not home_played or a_score > h_score else '#8b95a5'};">{a_score}</div>
          <div class="vs-divider">·</div>
          <div class="score" style="color:{home_color if not home_played or h_score > a_score else '#8b95a5'};">{h_score}</div>
        </div>
        <div class="team-block">
          <img class="team-logo" src="{logo_url(home)}" alt="{home}"/>
          <div>
            <div class="team-name">{short_name(home)}</div>
            <div class="team-city">{TEAMS.get(home, {{}}).get('city', home)}</div>
          </div>
        </div>
      </div>
      <div class="prob-bar-container">
        <div class="prob-bar-bg">
          <div class="prob-bar-away" style="width:{pa_pct}%; background:{away_color};"></div>
          <div class="prob-bar-home" style="width:{ph_pct}%; background:{home_color};"></div>
        </div>
        <div class="prob-labels">
          <span>{short_name(away)} {pa_pct}%</span>
          <span class="home-pct">{short_name(home)} {ph_pct}%</span>
        </div>
      </div>
      <div class="game-meta">
        <span><strong>{label}</strong></span>
        <span>Spread: <strong>{row['pred_margin']:+.1f}</strong> ({short_name(home)})</span>
        <span>Total: <strong>{row['pred_total']:.1f}</strong></span>
        {f'<span>{date_str}</span>' if date_str else ''}
      </div>
    </div>
    """


def standings_row(rank, row, top_seeds=7):
    """Render one team row in the standings table."""
    team = row['team']
    is_playoff = rank <= top_seeds

    odds = row['playoff_odds']
    odds_cls = 'odds-value'
    if odds < 0.30: odds_cls = 'odds-value low'
    elif odds < 0.70: odds_cls = 'odds-value mid'

    return f"""
    <div class="standings-row{' playoff' if is_playoff else ''}">
      <div class="rank">#{rank}</div>
      <div class="team-cell">
        <img src="{logo_url(team)}" alt="{team}"/>
      </div>
      <div class="team-cell">
        <div class="name">{team_name(team)}</div>
      </div>
      <div style="text-align:right;">{int(row['current_wins'])}</div>
      <div style="text-align:right;">{row['sim_wins_mean']:.1f}</div>
      <div class="{odds_cls}" style="text-align:right;">{odds*100:.1f}%</div>
    </div>
    """


def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>
    """


# ---------- Main ----------

def main():
    config = load_config()
    models_dir = config['data']['models_dir']
    outputs_dir = config['data']['outputs_dir']

    st.markdown(
        """<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
             <span style="font-size:2rem;">🏈</span>
             <div>
               <div style="font-size:1.8rem; font-weight:800; letter-spacing:-0.02em;">NFL Predictor</div>
               <div style="font-size:0.85rem; color:#8b95a5;">2026 Season · Win probability, spread, and score predictions</div>
             </div>
           </div>""",
        unsafe_allow_html=True
    )

    feats = load_features()
    schedules = load_schedules()

    if feats is None or schedules is None:
        st.error("Data not built. Run `python make_dataset.py` and `python -m src.features.build_features` first.")
        return

    tab1, tab2, tab3 = st.tabs(["Weekly Games", "Season Simulator", "Model Backtest"])

    # ----- TAB 1: Weekly Games -----
    with tab1:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            seasons = sorted(feats['season'].unique())
            season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="wk_season")
        with col2:
            weeks = sorted(feats[feats['season'] == season]['week'].unique())
            week = st.selectbox("Week", weeks, key="wk_week")

        winprob_m = load_model(models_dir, 'winprob_model')
        margin_m = load_model(models_dir, 'margin_model')
        total_m = load_model(models_dir, 'total_model')

        if not all([winprob_m, margin_m, total_m]):
            st.warning("Models not trained. Run `python -m src.models.train_winprob_spread` and `python -m src.models.train_score`.")
        else:
            preds = predict_week(feats, season, week, winprob_m, margin_m, total_m)
            if len(preds) == 0:
                st.info(f"No games scheduled for season {season} week {week}.")
            else:
                st.markdown(f'<div class="section-title">Season {season} · Week {week} · {len(preds)} games</div>', unsafe_allow_html=True)
                # Two-column layout for cards
                left, right = st.columns(2)
                for i, (_, game) in enumerate(preds.iterrows()):
                    (left if i % 2 == 0 else right).markdown(game_card(game), unsafe_allow_html=True)

    # ----- TAB 2: Season Simulator -----
    with tab2:
        sim_files = sorted(Path(outputs_dir).glob("season_sim_*.csv")) if Path(outputs_dir).exists() else []
        if not sim_files:
            st.info("No season simulations found. Run `python -m src.simulate.season_sim --season 2026 --week 1`.")
        else:
            col1, _ = st.columns([1, 3])
            with col1:
                sim_name = st.selectbox("Simulation", [f.name for f in sim_files], index=len(sim_files) - 1)
            df = pd.read_csv(Path(outputs_dir) / sim_name)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(metric_card("Teams", str(len(df))), unsafe_allow_html=True)
            m2.markdown(metric_card("Playoff Bids (>50%)", str((df['playoff_odds'] > 0.5).sum()),
                                    "teams with >50% odds"), unsafe_allow_html=True)
            m3.markdown(metric_card("Top Playoff Odds", f"{df['playoff_odds'].max()*100:.1f}%",
                                    df.loc[df['playoff_odds'].idxmax(), 'team']), unsafe_allow_html=True)
            m4.markdown(metric_card("Top Div Odds", f"{df['division_win_odds'].max()*100:.1f}%",
                                    df.loc[df['division_win_odds'].idxmax(), 'team']), unsafe_allow_html=True)

            # AFC / NFC side by side
            afc_col, nfc_col = st.columns(2)
            for label, conf_col in [("AFC", afc_col), ("NFC", nfc_col)]:
                with conf_col:
                    st.markdown(f'<div class="section-title">{label} Playoff Picture</div>', unsafe_allow_html=True)
                    conf_df = df[df['conference'] == label].sort_values('playoff_odds', ascending=False).reset_index(drop=True)
                    header = """
                    <div class="standings-row standings-header">
                      <div>Seed</div><div></div><div>Team</div>
                      <div style="text-align:right;">W</div>
                      <div style="text-align:right;">Proj</div>
                      <div style="text-align:right;">Playoffs</div>
                    </div>
                    """
                    rows_html = "".join(standings_row(i + 1, r) for i, r in conf_df.iterrows())
                    st.markdown(header + rows_html, unsafe_allow_html=True)

    # ----- TAB 3: Backtest -----
    with tab3:
        reports = load_backtest_reports(outputs_dir)
        if not reports:
            st.info("No backtest reports yet. Run the training scripts.")
            return

        # Summary metrics at top
        elo_txt = reports.get('backtest_elo', '')
        winprob_txt = reports.get('backtest_winprob', '')
        margin_txt = reports.get('backtest_margin', '')
        scores_txt = reports.get('backtest_scores', '')

        def extract(text, key):
            for line in text.splitlines():
                if line.strip().startswith(key):
                    try:
                        return line.split(':')[1].strip().split()[0]
                    except (IndexError, ValueError):
                        return "—"
            return "—"

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Elo Brier", extract(elo_txt, "Brier score"), "baseline"), unsafe_allow_html=True)
        m2.markdown(metric_card("LGB Brier", extract(winprob_txt, "Brier score"), "vs Elo"), unsafe_allow_html=True)
        m3.markdown(metric_card("Margin MAE", extract(margin_txt, "MAE"), "points"), unsafe_allow_html=True)
        m4.markdown(metric_card("Score MAE (avg)", extract(scores_txt, "MAE home_score"), "per team"), unsafe_allow_html=True)

        # Raw reports
        st.markdown('<div class="section-title">Full Backtest Reports</div>', unsafe_allow_html=True)
        for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin', 'backtest_total', 'backtest_scores']:
            if name in reports:
                with st.expander(name.replace('backtest_', '').replace('_', ' ').title(), expanded=(name == 'backtest_elo')):
                    st.code(reports[name], language='text')


if __name__ == "__main__":
    main()
