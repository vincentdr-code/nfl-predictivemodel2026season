# NFL Score & Season Predictor — Project Memory

## Project Goal
Full-stack NFL prediction system: win probability, spread, full score, season Monte Carlo, playoff bracket, and Pick 'Em vs the model.

## Current Status — v2 in progress

**v1 shipped:** Elo baseline + LightGBM winprob/margin/total + season sim + dashboard w/ team logos and editorial UI.
**v2 additions (this build):**
- **A1 QB adjustment**: per-QB EPA-based rating layered on top of team Elo
- **A2 Vegas benchmark**: ATS analysis vs closing lines
- **B1 Pick 'Em**: per-user picks, session-persistent, scored vs model
- **B3 Achievements**: first pick, perfect week, beat model, upset caller
- **C1 Playoff bracket**: round-by-round odds visualized
- **C2 Team comparison**: side-by-side head-to-head in Team Focus
- **C3 Historical accuracy**: to add (backtest tab will show model's week-by-week record)
- **C5 Power rankings**: Elo-based with week-over-week movement arrows
- **C6 Automated refresh**: GitHub Action runs every Tuesday 7am ET

## Data Sources
- **Games & PBP**: `nfl_data_py` (6,495 games, 1.14M+ plays, seasons 2002-2026)
- **QB names**: from schedules (`home_qb_name`, `away_qb_name`)
- **QB stats**: computed from PBP `passer_player_name` + EPA per dropback
- **Vegas lines**: `spread_line` in schedules (closing spread)
- **Team logos**: ESPN CDN

## Backtest (walk-forward 2010-2024, 3,903 games)
| Model | Metric | v1 | v2 target |
|---|---|---|---|
| Elo | Brier | 0.2220 | 0.2220 |
| LightGBM winprob | Brier | 0.2234 | (with QB adj) 0.220? |
| LightGBM margin | MAE | 10.43 | (with QB adj) 10.2? |
| Score home MAE | | 7.74 | tbd |

Fill in after v2 retrain.

## Architecture

**Data flow (Phase 1):** `nfl_data_py` → games_master.csv + team_game_stats.parquet + qb_ratings.parquet → features.parquet
**Model flow (Phase 2+):** features → LightGBM {winprob, margin, total} → predictions → season sim → playoff bracket
**Dashboard (Phase 7):** Streamlit reads all outputs from `data/processed/` and `outputs/`

## Files Reference (v2)
| File | Purpose |
|---|---|
| `make_dataset.py` | Phase 1 CLI |
| `src/ingest/games.py` | Schedules + PBP ingest (per-season cached) |
| `src/models/qb_ratings.py` | Per-QB EPA rating (NEW v2) |
| `src/features/build_features.py` | Feature matrix incl QB rating diff |
| `src/models/elo.py` | Elo baseline |
| `src/models/train_winprob_spread.py` | LightGBM classifier + regressor |
| `src/models/train_score.py` | Total-points model + score derivation |
| `src/simulate/season_sim.py` | Monte Carlo season sim |
| `src/simulate/playoff_bracket.py` | Round-by-round bracket odds (NEW v2) |
| `src/simulate/power_rankings.py` | Elo-based rankings + WoW movement (NEW v2) |
| `src/evaluate/vegas_benchmark.py` | ATS vs closing lines (NEW v2) |
| `src/gamification.py` | Pick 'Em + achievements + SQLite (NEW v2) |
| `src/teams.py` | Team metadata (logos, colors) |
| `dashboard/app.py` | Streamlit dashboard (v2: 5 tabs) |
| `.github/workflows/weekly-refresh.yml` | Weekly automated refresh (NEW v2) |

## Run Order (v2)
```
python make_dataset.py                           # schedules + PBP
python -m src.models.qb_ratings                  # NEW: QB rating table
python -m src.features.build_features            # features + QB diff
python -m src.models.elo                         # Elo baseline
python -m src.models.train_winprob_spread        # winprob + margin
python -m src.models.train_score                 # total → scores
python -m src.evaluate.vegas_benchmark           # NEW: ATS analysis
python -m src.simulate.season_sim --season 2026 --week 1
python -m src.simulate.power_rankings            # NEW: power rankings
streamlit run dashboard/app.py                   # Dashboard
```

## Environment
- Python 3.14.4 local, 3.12 on GitHub Action (venv locally, containerized on CI)
- Streamlit Cloud auto-redeploys on push to main
- SQLite for Pick 'Em picks (`data/processed/gamification.db`) — regenerable
- Repo: https://github.com/vincentdr-code/nfl-predictivemodel2026season.git

## Design System
- Font: Geist + Geist Mono (via Google Fonts)
- Accent: `#E11D48` (deep rose)
- Palette: `#0A0A0B` bg, `#111114` surface
- No emojis, no neon, tabular numbers everywhere for stats
- Cards omitted; use border-b dividers + editorial spacing

## Known Limitations
- Historical injuries not integrated (needs paid data source)
- QB rating uses simple shrinkage (k=200), no positional adjustment
- Playoff bracket odds are a heuristic (not proper Monte Carlo tournament sim)
- Pick 'Em is honor-system multi-user via username input (no auth)
- Weather uses only what nfl_data_py provides (temp + wind, no precipitation)
