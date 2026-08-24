"""
Streamlit dashboard for the NFL Predictor.

Run: streamlit run dashboard/app.py
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import pickle
import yaml


st.set_page_config(page_title="NFL Predictor 2026", layout="wide")


@st.cache_data
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")


@st.cache_data
def load_schedules():
    return pd.read_csv("data/processed/games_master.csv")


@st.cache_data
def load_backtest_reports(outputs_dir):
    reports = {}
    for name in ['backtest_elo', 'backtest_winprob', 'backtest_margin',
                 'backtest_total', 'backtest_scores']:
        p = Path(outputs_dir) / f"{name}.txt"
        if p.exists():
            reports[name] = p.read_text()
    return reports


@st.cache_resource
def load_winprob_model(models_dir):
    p = Path(models_dir) / "winprob_model.pkl"
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_margin_model(models_dir):
    p = Path(models_dir) / "margin_model.pkl"
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_total_model(models_dir):
    p = Path(models_dir) / "total_model.pkl"
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def predict_week(feats, season, week, winprob, margin, total):
    """Score a specific week's games with the three models."""
    subset = feats[(feats['season'] == season) & (feats['week'] == week)].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    X_wp = subset[winprob['features']].fillna(0)
    X_m = subset[margin['features']].fillna(0)
    X_t = subset[total['features']].fillna(0)

    subset['pred_home_win_prob'] = winprob['model'].predict_proba(X_wp)[:, 1]
    subset['pred_margin'] = margin['model'].predict(X_m)
    subset['pred_total'] = total['model'].predict(X_t)
    subset['pred_home_score'] = (subset['pred_margin'] + subset['pred_total']) / 2
    subset['pred_away_score'] = (subset['pred_total'] - subset['pred_margin']) / 2

    return subset[['game_id', 'season', 'week', 'gameday',
                   'home_team', 'away_team',
                   'pred_home_win_prob', 'pred_margin', 'pred_total',
                   'pred_home_score', 'pred_away_score']]


def main():
    config = load_config()
    models_dir = config['data']['models_dir']
    outputs_dir = config['data']['outputs_dir']

    st.title("NFL Score & Season Predictor")
    st.caption("Win probability, spread, and full-score predictions with Monte Carlo season simulation")

    feats = load_features() if Path("data/processed/features.parquet").exists() else None
    schedules = load_schedules() if Path("data/processed/games_master.csv").exists() else None

    if feats is None or schedules is None:
        st.warning("Data not yet built. Run `python make_dataset.py` and `python -m src.features.build_features` first.")
        return

    tab1, tab2, tab3 = st.tabs(["This Week's Games", "Season Standings", "Backtest"])

    with tab1:
        st.header("Weekly Predictions")

        seasons_available = sorted(feats['season'].unique())
        col1, col2 = st.columns(2)
        with col1:
            season = st.selectbox("Season", seasons_available, index=len(seasons_available) - 1)
        with col2:
            weeks_available = sorted(feats[feats['season'] == season]['week'].unique())
            week = st.selectbox("Week", weeks_available)

        winprob = load_winprob_model(models_dir)
        margin_m = load_margin_model(models_dir)
        total_m = load_total_model(models_dir)

        if winprob is None or margin_m is None or total_m is None:
            st.warning("Models not yet trained. Run `python -m src.models.train_winprob_spread` and "
                       "`python -m src.models.train_score` first.")
        else:
            preds = predict_week(feats, season, week, winprob, margin_m, total_m)
            if len(preds) == 0:
                st.info(f"No games for season {season} week {week}")
            else:
                display = preds.rename(columns={
                    'gameday': 'Date',
                    'home_team': 'Home',
                    'away_team': 'Away',
                    'pred_home_win_prob': 'Home Win %',
                    'pred_margin': 'Predicted Spread',
                    'pred_total': 'Predicted Total',
                    'pred_home_score': 'Home Score',
                    'pred_away_score': 'Away Score',
                }).drop(columns=['game_id', 'season', 'week'])
                display['Home Win %'] = (display['Home Win %'] * 100).round(1)
                display['Predicted Spread'] = display['Predicted Spread'].round(1)
                display['Predicted Total'] = display['Predicted Total'].round(1)
                display['Home Score'] = display['Home Score'].round(1)
                display['Away Score'] = display['Away Score'].round(1)
                st.dataframe(display, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Season Simulation")
        sim_files = sorted(Path(outputs_dir).glob("season_sim_*.csv")) if Path(outputs_dir).exists() else []
        if not sim_files:
            st.info("No season simulations found. Run `python -m src.simulate.season_sim --season 2025 --week 1`")
        else:
            sim_file = st.selectbox("Simulation file", [f.name for f in sim_files])
            df = pd.read_csv(Path(outputs_dir) / sim_file)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("AFC")
                afc = df[df['conference'] == 'AFC'].sort_values('playoff_odds', ascending=False)
                st.dataframe(afc.drop(columns=['conference']).reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            with col2:
                st.subheader("NFC")
                nfc = df[df['conference'] == 'NFC'].sort_values('playoff_odds', ascending=False)
                st.dataframe(nfc.drop(columns=['conference']).reset_index(drop=True),
                             use_container_width=True, hide_index=True)

            st.subheader("Playoff Odds — All Teams")
            chart_df = df.set_index('team')['playoff_odds'].sort_values(ascending=True)
            st.bar_chart(chart_df, horizontal=True)

    with tab3:
        st.header("Backtest Reports")
        reports = load_backtest_reports(outputs_dir)
        if not reports:
            st.info("No backtest reports yet. Run the model training scripts.")
        else:
            for name, text in reports.items():
                with st.expander(name.replace('backtest_', '').title(), expanded=True):
                    st.code(text, language='text')


if __name__ == "__main__":
    main()
