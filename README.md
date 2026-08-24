# NFL Score & Season Predictor

A disciplined, phase-gated approach to building NFL prediction models: win probability, point spread, full score prediction, and season Monte Carlo simulation.

## Stack
- **Python 3.14+** with `nfl_data_py`, XGBoost/LightGBM, scikit-learn
- **Validation**: Walk-forward testing; Brier score + calibration
- **Data**: ESPN injuries, Open-Meteo weather, nfl_data_py schedules + PBP
- **UI** (Phase 7): Streamlit dashboard

## Quick Start

1. **Clone & venv**:
   ```bash
   git clone https://github.com/vincentdr-code/nfl-predictivemodel2026season.git
   cd nfl-predictivemodel2026season
   python -m venv .venv
   .venv/Scripts/Activate.ps1  # Windows
   pip install -r requirements.txt
   ```

2. **Build dataset** (Phase 1):
   ```bash
   python make_dataset.py
   ```

3. **Run Elo baseline** (Phase 2):
   ```bash
   python -m src.models.elo
   ```

4. **Train models & view dashboard** (Phase 4+):
   ```bash
   python -m src.models.train_winprob_spread
   streamlit run dashboard/app.py
   ```

See `CLAUDE.md` for current status, data sources, and modeling conventions.
