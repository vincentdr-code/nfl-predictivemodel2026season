"""
Phase 6: Monte Carlo season simulator.

Given current win probabilities for every remaining game, simulate the
remainder of the season N times to derive:
    - projected wins (mean, p10, p90)
    - division win odds
    - playoff berth odds (top 7 per conference under 2024 format)
"""
import argparse
import pickle
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict


# NFL divisions (32-team, 2024 alignment)
DIVISIONS = {
    'AFC_EAST':  ['BUF', 'MIA', 'NE', 'NYJ'],
    'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC_WEST':  ['DEN', 'KC', 'LAC', 'LV'],
    'NFC_EAST':  ['DAL', 'NYG', 'PHI', 'WAS'],
    'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
    'NFC_WEST':  ['ARI', 'LA', 'SEA', 'SF'],
}
TEAM_TO_DIVISION = {t: d for d, teams in DIVISIONS.items() for t in teams}
CONFERENCES = {
    'AFC': [t for d in ['AFC_EAST', 'AFC_NORTH', 'AFC_SOUTH', 'AFC_WEST'] for t in DIVISIONS[d]],
    'NFC': [t for d in ['NFC_EAST', 'NFC_NORTH', 'NFC_SOUTH', 'NFC_WEST'] for t in DIVISIONS[d]],
}
TEAM_TO_CONF = {t: c for c, teams in CONFERENCES.items() for t in teams}


def _predict_win_probs(feats, model_path):
    """Score every game in `feats` with the saved winprob model."""
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    features = bundle['features']
    X = feats[features].fillna(0)
    return model.predict_proba(X)[:, 1]


def _current_wins(schedules, season, upto_week):
    """Count actual wins for each team through weeks 1..upto_week-1 in `season`."""
    wins = defaultdict(int)
    played = schedules[(schedules['season'] == season) & (schedules['week'] < upto_week)]
    for _, g in played.iterrows():
        if g['home_score'] > g['away_score']:
            wins[g['home_team']] += 1
        else:
            wins[g['away_team']] += 1
    return wins


def simulate_season(feats, schedules, season, current_week, model_path,
                    n_sims=10000, seed=42):
    """
    Simulate the remainder of `season` starting from `current_week`.

    Args:
        feats: feature matrix (must include game rows for `season`)
        schedules: full games_master schedule
        season: int
        current_week: int (games in weeks >= current_week are simulated)
        model_path: path to winprob_model.pkl
        n_sims: number of simulations
        seed: RNG seed

    Returns DataFrame: one row per team with
        current_wins, sim_wins_mean, sim_wins_p10, sim_wins_p90,
        division_win_odds, playoff_odds
    """
    rng = np.random.default_rng(seed)

    season_feats = feats[(feats['season'] == season) & (feats['week'] >= current_week)].copy()
    if len(season_feats) == 0:
        raise ValueError(f"No games found for season={season}, week>={current_week}")

    season_feats['home_win_prob'] = _predict_win_probs(season_feats, model_path)

    # Current wins from played games
    base_wins = _current_wins(schedules, season, current_week)

    # Preallocate: outcomes matrix (n_sims x n_remaining_games) via one draw
    probs = season_feats['home_win_prob'].values
    n_games = len(probs)
    home_teams = season_feats['home_team'].values
    away_teams = season_feats['away_team'].values

    print(f"  Simulating {n_sims:,} seasons x {n_games} remaining games...")
    draws = rng.random((n_sims, n_games))
    home_wins_matrix = draws < probs  # shape (n_sims, n_games), True if home wins

    # For each simulation, tally each team's total wins
    all_teams = sorted(set(list(home_teams) + list(away_teams) + list(base_wins.keys())))
    team_idx = {t: i for i, t in enumerate(all_teams)}
    sim_wins = np.zeros((n_sims, len(all_teams)), dtype=np.int16)

    # Seed with base wins
    for team, w in base_wins.items():
        if team in team_idx:
            sim_wins[:, team_idx[team]] = w

    # Add simulated wins vectorized per game
    for g_idx in range(n_games):
        h_idx = team_idx[home_teams[g_idx]]
        a_idx = team_idx[away_teams[g_idx]]
        home_wins_this = home_wins_matrix[:, g_idx]
        sim_wins[home_wins_this, h_idx] += 1
        sim_wins[~home_wins_this, a_idx] += 1

    # Tally division winners and playoff seeds per sim
    div_wins_count = defaultdict(int)      # team -> sims where they won their division
    playoff_count = defaultdict(int)       # team -> sims where they made the playoffs

    for s in range(n_sims):
        # Division winners: highest wins in each division (tie-break random for now)
        div_winners = set()
        for div, teams in DIVISIONS.items():
            candidates = [(sim_wins[s, team_idx[t]], rng.random(), t) for t in teams if t in team_idx]
            if not candidates:
                continue
            _, _, winner = max(candidates)
            div_winners.add(winner)
            div_wins_count[winner] += 1

        # Playoff seeds: 4 div winners + top 3 non-div-winners per conference (by wins)
        for conf, conf_teams in CONFERENCES.items():
            conf_div_winners = [t for t in conf_teams if t in div_winners]
            for t in conf_div_winners:
                playoff_count[t] += 1

            # Wild cards: top 3 by wins among non-div-winners
            wild_candidates = [(sim_wins[s, team_idx[t]], rng.random(), t)
                               for t in conf_teams
                               if t in team_idx and t not in div_winners]
            wild_candidates.sort(reverse=True)
            for _, _, t in wild_candidates[:3]:
                playoff_count[t] += 1

    # Build output DataFrame
    rows = []
    for team in all_teams:
        idx = team_idx[team]
        wins_arr = sim_wins[:, idx]
        rows.append({
            'team': team,
            'division': TEAM_TO_DIVISION.get(team, ''),
            'conference': TEAM_TO_CONF.get(team, ''),
            'current_wins': int(base_wins.get(team, 0)),
            'sim_wins_mean': float(wins_arr.mean()),
            'sim_wins_p10': float(np.percentile(wins_arr, 10)),
            'sim_wins_p90': float(np.percentile(wins_arr, 90)),
            'division_win_odds': div_wins_count[team] / n_sims,
            'playoff_odds': playoff_count[team] / n_sims,
        })

    return pd.DataFrame(rows).sort_values(
        ['conference', 'playoff_odds'], ascending=[True, False]
    ).reset_index(drop=True)


def main(config_file='config.yaml', season=None, week=None, n_sims=None):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    processed = config['data']['processed_dir']
    outputs = config['data']['outputs_dir']
    models_dir = config['data']['models_dir']

    if season is None:
        season = max(config['data']['seasons'])
    if week is None:
        week = 1
    if n_sims is None:
        n_sims = config['simulation']['n_sims']

    print(f"Season simulation: season={season}, from week={week}, sims={n_sims:,}")

    feats = pd.read_parquet(f"{processed}/features.parquet")
    schedules = pd.read_csv(f"{processed}/games_master.csv")
    model_path = f"{models_dir}/winprob_model.pkl"

    result = simulate_season(feats, schedules, season, week, model_path,
                             n_sims=n_sims, seed=config['simulation']['random_seed'])

    Path(outputs).mkdir(parents=True, exist_ok=True)
    output_file = f"{outputs}/season_sim_{season}_week{week}.csv"
    result.to_csv(output_file, index=False)
    print(f"\n[OK] Wrote {output_file}")
    print("\nTop 8 playoff odds:")
    print(result.nlargest(8, 'playoff_odds')[
        ['team', 'division', 'current_wins', 'sim_wins_mean',
         'division_win_odds', 'playoff_odds']
    ].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--n-sims", type=int, dest='n_sims')
    args = parser.parse_args()
    main(args.config, args.season, args.week, args.n_sims)
