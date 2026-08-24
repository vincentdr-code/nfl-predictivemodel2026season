# NFL Score & Season Predictor — Project Memory

## Project Goal
Build a disciplined, full-stack NFL prediction system: win probability, point spread, full score prediction, and season Monte Carlo simulator.

## Current Status
**All Phases 1-7 complete and validated end-to-end.**

## Data Sources
- **Games & PBP**: `nfl_data_py` — schedules 2002-2025 (6,223 regular-season games), PBP with EPA (1,143,032 plays)
  - Schedules already contain: temp, wind, home_qb_name, home_rest/away_rest, div_game, roof, spread_line
- **Historical injuries**: Stubbed (no free historical source); current-week only via ESPN
- **Weather**: Comes bundled with `nfl_data_py` schedule; Open-Meteo module available for precipitation supplement if needed

## Backtest Results (walk-forward, seasons 2010-2024, 3,903 games)

| Model | Metric | Value | vs Baseline |
|---|---|---|---|
| **Elo baseline** | Brier | **0.2220** | (baseline) |
| Elo | Accuracy | 63.5% | |
| **LightGBM winprob** | Brier | 0.2234 | -0.0014 (tied) |
| LightGBM winprob | AUC | 0.6827 | |
| LightGBM winprob | Accuracy | 63.2% | |
| **LightGBM margin** | MAE | 10.43 | -0.98 vs predict-zero (11.41) |
| **LightGBM total** | MAE | 10.78 | |
| **Full score model** | MAE home_score | 7.74 | |
| Full score model | MAE away_score | 7.50 | |

**Note on Brier tie with Elo**: LightGBM includes Elo rating as one of its features. The remaining ~50 features collectively add little additional signal over Elo alone for win probability. Margin/total regression benefits meaningfully from richer features.

## Modeling Conventions
- **Target variables**: `home_win` (1/0), `margin` (home-away), `total` (home+away)
- **Validation**: Walk-forward — train seasons < S, validate on last 20% of training seasons (early stopping), test season S. Seasons 2010-2024 tested.
- **No leakage**: Rolling features use `.shift(1)` per (team, season) to exclude current game. Prior-season means fill early-season NaNs. Elo is computed chronologically with pre-game snapshot.
- **Leaky columns explicitly excluded** from features (`train_winprob_spread.LEAKY_COLS`): home_win, margin, total, home_score, away_score, spread_line, total_line (Vegas lines kept as gold benchmark only, not model input).

## Hyperparameters
- **Elo**: K=20, home_advantage=65, season_reversion=1/3, initial=1500
- **LightGBM**: n_estimators=2000 (capped, actual chosen by early stopping ~50-150), lr=0.03, max_depth=4, num_leaves=15, min_child_samples=30, reg_alpha=0.1, reg_lambda=0.5, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50
- **Season sim**: N=10,000 sims per week, 2024 playoff format (7 seeds/conference)

## Files Reference
| File | Purpose |
|---|---|
| `make_dataset.py` | Phase 1 CLI: fetch schedules + PBP, write games_master.csv + team_game_stats.parquet |
| `src/ingest/games.py` | Schedule + PBP ingest with per-season caching + retries |
| `src/ingest/injuries.py` | ESPN current-week injuries (historical is stub) |
| `src/ingest/weather.py` | Open-Meteo historical weather (supplement to PBP-bundled temp/wind) |
| `src/features/build_features.py` | Phase 3 feature matrix (rolling + Elo + situational) |
| `src/models/elo.py` | Phase 2 Elo baseline + backtest CLI |
| `src/models/train_winprob_spread.py` | Phase 4 LightGBM classifier + margin regressor |
| `src/models/train_score.py` | Phase 5 total-points regressor + score derivation |
| `src/simulate/season_sim.py` | Phase 6 Monte Carlo (playoff odds, projected wins) |
| `dashboard/app.py` | Phase 7 Streamlit dashboard (3 tabs) |
| `tests/test_ingest.py` | Phase 1 data checks |
| `tests/test_no_leakage.py` | Phase 3 leakage guard (extend as features added) |

## Run Order
1. `python make_dataset.py` — build games_master.csv + team_game_stats.parquet (once)
2. `python -m src.models.elo` — Elo baseline backtest
3. `python -m src.features.build_features` — features.parquet
4. `python -m src.models.train_winprob_spread` — winprob + margin models
5. `python -m src.models.train_score` — total + derived scores
6. `python -m src.simulate.season_sim --season 2025 --week 1` — season sim
7. `streamlit run dashboard/app.py` — dashboard

## Environment
- Python 3.14.4, venv at `.venv/`
- Deps in `requirements.txt` (pinned; regenerate with `pip freeze > requirements.txt`)
- Repo: https://github.com/vincentdr-code/nfl-predictivemodel2026season.git
- Local path: `C:\Users\danie\projects\nfl-predictivemodel2026season`

## Known Limitations
- Historical injuries not integrated (would need paid data source); current-week ESPN pull works
- Vegas `spread_line` deliberately excluded from features (kept as external benchmark)
- Weather is limited to what `nfl_data_py` provides (temp, wind); precipitation stub in weather.py available but not wired in
- LightGBM winprob ties Elo — to actually beat Elo would need higher-signal features (QB-specific EPA vs backup, coach tendencies)
