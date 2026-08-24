# NFL Score & Season Predictor — Project Memory

## Project Goal
Build a disciplined, full-stack NFL prediction system: win probability, point spread, full score prediction, and season Monte Carlo simulator. Phase-gated approach ensures each checkpoint is validated before the next begins.

## Current Status
**Phase**: 0 (Bootstrap)  
**Stage**: Scaffolding complete; venv + dependencies installing

## Data Sources
- **Games & play-by-play**: `nfl_data_py` (schedules 2002–2025, PBP with EPA)
- **Injuries**: ESPN API `https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries`
- **Weather**: Open-Meteo historical `https://archive-api.open-meteo.com/v1/archive` (free, no key)
- **Output**: Single master CSV at `data/processed/games_master.csv` per run

## Modeling Conventions
- **Target variable (Phase 2–4)**: `home_win` (1/0), `home_margin` (score diff)
- **Target variable (Phase 5)**: `home_score`, `away_score`, `total_points`
- **Validation method**: Walk-forward (train on seasons 1–N, test on N+1)
- **No leakage rule**: Features only use data from before kickoff of target game; strictly enforced via `tests/test_no_leakage.py`
- **Metric (Phase 2–4)**: Brier score + calibration curve; Phase 2 (Elo) is permanent baseline
- **Metric (Phase 5)**: MAE on individual team scores

## Key Hyperparameters
- **Elo**: K-factor=20, home_advantage=65, season_reversion=1/3, initial_rating=1500
- **LightGBM**: n_estimators=500, max_depth=7, learning_rate=0.1, subsample=0.8
- **Simulation**: N=10,000 sims per week, 2024 playoff format (7 seeds per conference)

## Phase Checkpoints (update as completed)

### Phase 1 — Data Ingestion
- [ ] `games.py` pulls schedules, PBP via nfl_data_py; caches in `data/raw/`
- [ ] `injuries.py` pulls ESPN injury reports, normalizes to (game_id, team, week, player, position, status)
- [ ] `weather.py` pulls Open-Meteo historical data for outdoor stadiums only
- [ ] `make_dataset.py` orchestrates all three → writes `games_master.csv`
- [ ] Tests pass: row count ~6k, no NaNs in scores, unique game_ids

### Phase 2 — Elo Baseline
- [ ] `elo.py` implements FiveThirtyEight Elo with K=20, home_advantage=65
- [ ] Walk-forward backtest on seasons 2010–2024
- [ ] **Brier score**: _____ (baseline for all future models)
- [ ] **Calibration**: _____ (predicted vs. actual win% by decile)
- [ ] Tests pass: ratings within [1000, 2000], symmetric (home/away swap)

### Phase 3 — Feature Engineering
- [ ] `build_features.py` computes rolling EPA, situational, injury, weather, Elo features
- [ ] Strict no-leakage enforcement: every feature uses only pre-kickoff data
- [ ] `test_no_leakage.py` passes: assert no future games in rolling windows
- [ ] Output: `features.parquet` with all games 2002–2025 + features

### Phase 4 — Win Probability + Spread
- [ ] `train_winprob_spread.py`: LightGBM classifier (win) + regressor (margin)
- [ ] Walk-forward validation: Brier score, MAE, calibration curve
- [ ] **Brier score**: _____ (must be < Elo baseline to proceed)
- [ ] **MAE (margin)**: _____ 
- [ ] Backtest report saved to `outputs/backtest_phase4.txt`

### Phase 5 — Full Score Model
- [ ] `train_score.py`: LightGBM regressor for total_points; back out home/away scores
- [ ] Validate: MAE on individual team scores, margin/total consistency
- [ ] **MAE (home_score)**: _____
- [ ] **MAE (away_score)**: _____
- [ ] Models saved to `data/processed/models/`

### Phase 6 — Season Simulator
- [ ] `season_sim.py`: simulate remaining season N=10k times
- [ ] Outputs: playoff_odds, division_win_odds, projected_wins_mean/p10/p90 per team
- [ ] Write to `outputs/season_sim_week{W}.csv`

### Phase 7 — Dashboard + Weekly Refresh
- [ ] `make_dataset.py --refresh --week W` orchestrates full pipeline
- [ ] Streamlit dashboard: This Week's Games, Season Standings, Backtest tabs
- [ ] Run: `streamlit run dashboard/app.py`

## Critical Files
- `src/ingest/games.py` — schedule + PBP
- `src/ingest/injuries.py` — injury data
- `src/ingest/weather.py` — weather data
- `src/models/elo.py` — Elo baseline (permanent reference)
- `src/features/build_features.py` — feature pipeline (no leakage!)
- `src/models/train_winprob_spread.py` — Phase 4 model
- `src/models/train_score.py` — Phase 5 model
- `src/simulate/season_sim.py` — Monte Carlo
- `tests/test_no_leakage.py` — **CRITICAL**: run before any model training
- `dashboard/app.py` — Streamlit UI

## Commit Strategy
- Commit after each phase checkpoint with backtest numbers in the commit message.
- Update this file after each phase.
- Never start a new phase until the previous phase's tests pass and numbers are committed.

## Notes
- Python 3.14.4 + venv at `.venv/`
- All data pulls are cached (raw → `data/raw/`, processed → `data/processed/`)
- Seasons 2002–2025 are the full historical range; validation starts at 2010
- No external APIs require keys (ESPN injury, Open-Meteo weather are free)
