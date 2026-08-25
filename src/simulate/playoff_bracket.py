"""
Playoff bracket probabilities from Monte Carlo season sim output.

For each conference, computes the joint probability distribution over
which teams win in each round (Wildcard, Divisional, Conference, Super Bowl).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.simulate.season_sim import DIVISIONS, CONFERENCES, TEAM_TO_DIVISION


def _sim_seed_distribution(schedules, season, current_week, model_path,
                           n_sims=10000, seed=42):
    """
    Reuse season_sim's core loop but return a per-sim seed assignment
    instead of aggregated odds. Returns list of dicts:
        [{team: seed}, {team: seed}, ...] length n_sims
        where seed is 1-7 for playoff teams, None otherwise, per conference
    """
    from src.simulate.season_sim import _predict_win_probs, _current_wins
    import pickle

    feats_path = "data/processed/features.parquet"
    feats = pd.read_parquet(feats_path)
    season_feats = feats[(feats['season'] == season) & (feats['week'] >= current_week)].copy()
    season_feats['home_win_prob'] = _predict_win_probs(season_feats, model_path)

    base_wins = _current_wins(schedules, season, current_week)
    rng = np.random.default_rng(seed)

    probs = season_feats['home_win_prob'].values
    n_games = len(probs)
    home_teams = season_feats['home_team'].values
    away_teams = season_feats['away_team'].values
    all_teams = sorted(set(list(home_teams) + list(away_teams) + list(base_wins.keys())))
    team_idx = {t: i for i, t in enumerate(all_teams)}

    draws = rng.random((n_sims, n_games))
    home_wins_matrix = draws < probs
    sim_wins = np.zeros((n_sims, len(all_teams)), dtype=np.int16)
    for team, w in base_wins.items():
        if team in team_idx:
            sim_wins[:, team_idx[team]] = w
    for g in range(n_games):
        h_idx = team_idx[home_teams[g]]
        a_idx = team_idx[away_teams[g]]
        home_wins_this = home_wins_matrix[:, g]
        sim_wins[home_wins_this, h_idx] += 1
        sim_wins[~home_wins_this, a_idx] += 1

    # Per-sim seed assignments per conference
    per_sim_seeds = []  # list of {conf: [team_by_seed_1, ...team_by_seed_7]}
    for s in range(n_sims):
        conf_seeds = {}
        for conf, conf_teams in CONFERENCES.items():
            # Division winners
            div_winners = []
            for div, teams in DIVISIONS.items():
                if not teams or teams[0] not in [t for t in conf_teams]:
                    continue
                if teams[0] not in team_idx:
                    continue
                candidates = [(sim_wins[s, team_idx[t]], rng.random(), t)
                              for t in teams if t in team_idx]
                if not candidates:
                    continue
                _, _, winner = max(candidates)
                div_winners.append(winner)
            # Sort div winners by wins for seeds 1-4
            div_winners.sort(key=lambda t: (sim_wins[s, team_idx[t]], rng.random()), reverse=True)
            # Wild cards: best 3 non-div-winners
            wild_candidates = [(sim_wins[s, team_idx[t]], rng.random(), t)
                               for t in conf_teams
                               if t in team_idx and t not in set(div_winners)]
            wild_candidates.sort(reverse=True)
            wild_cards = [t for _, _, t in wild_candidates[:3]]
            conf_seeds[conf] = div_winners + wild_cards  # 7 teams
        per_sim_seeds.append(conf_seeds)

    return per_sim_seeds


def compute_bracket_probs(sim_file):
    """
    Simpler approach: derive round-by-round win probabilities from
    the aggregated sim CSV alone.

    For each conference, top 7 teams by playoff odds. Compute round
    probabilities using pairwise strength (proxied by projected wins).

    Returns dict:
        {conf: {
            'seeds': [{team, seed, playoff_odds, div_win_odds}, ...],
            'wc_win': {team: pct},
            'div_win': {team: pct},
            'conf_win': {team: pct},
            'sb_win': {team: pct}
        }}
    """
    df = pd.read_csv(sim_file)
    result = {}
    for conf in ['AFC', 'NFC']:
        conf_df = df[df['conference'] == conf].sort_values('playoff_odds', ascending=False).head(7).reset_index(drop=True)

        # Playoff odds already give us "in the field" probability
        # For simplicity, deeper-round probability = playoff_odds * (proj_wins / total_proj_wins)^depth
        seeds = []
        for i, r in conf_df.iterrows():
            seeds.append({
                'team': r['team'],
                'seed': i + 1,
                'playoff_odds': float(r['playoff_odds']),
                'proj_wins': float(r['sim_wins_mean']),
            })

        total_strength = sum(s['proj_wins'] for s in seeds)
        for s in seeds:
            base = s['playoff_odds']
            strength_share = s['proj_wins'] / total_strength if total_strength else 1/7
            # Rough tournament progression model
            s['wc_win']   = base * (0.55 + 0.9 * strength_share) if s['seed'] > 1 else base
            s['wc_win']   = min(base, s['wc_win'])
            s['div_win']  = s['wc_win'] * (0.5 + 0.7 * strength_share)
            s['conf_win'] = s['div_win'] * (0.5 + 0.7 * strength_share)
            s['sb_win']   = s['conf_win'] * (0.5 + 0.7 * strength_share) * 0.5  # 50% to win once in SB

        result[conf] = seeds
    return result


def main(config_file='config.yaml', sim_file=None):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    outputs = config['data']['outputs_dir']
    if sim_file is None:
        candidates = sorted(Path(outputs).glob("season_sim_*.csv"))
        if not candidates:
            print("No season sim files found.")
            return
        sim_file = str(candidates[-1])
    print(f"Computing bracket from {sim_file}...")
    result = compute_bracket_probs(sim_file)
    for conf, seeds in result.items():
        print(f"\n{conf} bracket odds:")
        header = f"{'Seed':<5} {'Team':<5} {'Playoff':<10} {'WC':<8} {'Div':<8} {'Conf':<8} {'SB':<8}"
        print(header)
        print("-" * len(header))
        for s in seeds:
            print(f"{s['seed']:<5} {s['team']:<5} "
                  f"{s['playoff_odds']*100:>6.1f}%   "
                  f"{s['wc_win']*100:>5.1f}%  "
                  f"{s['div_win']*100:>5.1f}%  "
                  f"{s['conf_win']*100:>5.1f}%  "
                  f"{s['sb_win']*100:>5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sim-file", dest='sim_file')
    args = parser.parse_args()
    main(args.config, args.sim_file)
