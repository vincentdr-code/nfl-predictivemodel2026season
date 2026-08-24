"""
Elo rating system for NFL teams (FiveThirtyEight methodology).
"""
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class EloRating:
    """Tracks and updates Elo ratings for NFL teams."""

    NFL_TEAMS = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LA', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
        'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SF', 'SEA',
        'STL', 'TB', 'TEN', 'WAS'
    ]

    def __init__(self, initial_rating=1500, k_factor=20, home_advantage=65,
                 season_reversion_factor=0.333, min_rating=1000, max_rating=2000):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_reversion_factor = season_reversion_factor
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.ratings = {team: initial_rating for team in self.NFL_TEAMS}

    def get(self, team):
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]

    def get_win_probability(self, elo_diff):
        """Log5 conversion: elo_diff (with HFA applied) -> home win prob."""
        return 1.0 / (10.0 ** (-elo_diff / 400.0) + 1.0)

    def margin_of_victory_multiplier(self, margin, elo_diff):
        """FiveThirtyEight's MOV multiplier."""
        return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))

    def process_game(self, home_team, away_team, home_score, away_score):
        """
        Compute pre-game prediction, update ratings, return dict of results.
        """
        home_pre = self.get(home_team)
        away_pre = self.get(away_team)

        # Predictions use home-field advantage
        elo_diff_with_hfa = (home_pre + self.home_advantage) - away_pre
        home_win_prob = self.get_win_probability(elo_diff_with_hfa)

        # Actual outcome
        if home_score > away_score:
            actual_home = 1.0
            margin = home_score - away_score
            winner_elo_diff = elo_diff_with_hfa
        else:
            actual_home = 0.0
            margin = away_score - home_score
            winner_elo_diff = -elo_diff_with_hfa

        # MOV-adjusted K
        mov_mult = self.margin_of_victory_multiplier(margin, winner_elo_diff)
        delta = self.k_factor * mov_mult * (actual_home - home_win_prob)

        new_home = float(np.clip(home_pre + delta, self.min_rating, self.max_rating))
        new_away = float(np.clip(away_pre - delta, self.min_rating, self.max_rating))

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        return {
            'home_elo_pre': home_pre,
            'away_elo_pre': away_pre,
            'elo_diff_hfa': elo_diff_with_hfa,
            'elo_win_prob_home': home_win_prob,
            'home_elo_post': new_home,
            'away_elo_post': new_away,
        }

    def revert_season(self):
        """Regress all ratings toward the initial rating between seasons."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.season_reversion_factor)
                + self.initial_rating * self.season_reversion_factor
            )


def backtest_elo(schedules, config):
    """
    Walk-forward chronological Elo backtest.

    Args:
        schedules: DataFrame with columns season, week, home_team, away_team,
                   home_score, away_score
        config: dict with 'elo' and 'backtest' sections

    Returns:
        DataFrame of per-game predictions with pre-game Elo, win prob, actual outcome.
        Filtered to seasons in [walk_forward_start_season, walk_forward_end_season].
    """
    elo_config = config['elo']
    start = config['backtest']['walk_forward_start_season']
    end = config['backtest']['walk_forward_end_season']

    elo = EloRating(**elo_config)

    # Only played games contribute to Elo ratings and backtest metrics
    played = schedules[schedules['home_score'].notna() & schedules['away_score'].notna()]
    games = played.sort_values(['season', 'week', 'gameday']).reset_index(drop=True)
    results = []
    prev_season = None

    for _, g in games.iterrows():
        season = int(g['season'])

        if prev_season is not None and season != prev_season:
            elo.revert_season()

        r = elo.process_game(
            g['home_team'], g['away_team'],
            g['home_score'], g['away_score']
        )
        r['game_id'] = g['game_id']
        r['season'] = season
        r['week'] = int(g['week'])
        r['home_team'] = g['home_team']
        r['away_team'] = g['away_team']
        r['home_score'] = g['home_score']
        r['away_score'] = g['away_score']
        r['home_win'] = int(g['home_score'] > g['away_score'])
        results.append(r)

        prev_season = season

    out = pd.DataFrame(results)
    return out[(out['season'] >= start) & (out['season'] <= end)].reset_index(drop=True)


def brier_score(y_true, y_prob):
    """Brier score = mean squared error on probabilities."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def log_loss(y_true, y_prob, eps=1e-15):
    """Binary log loss."""
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    y = np.asarray(y_true)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y_true, y_prob):
    """Accuracy of the max-probability prediction."""
    return float(np.mean((np.asarray(y_prob) > 0.5) == np.asarray(y_true)))


def calibration_table(results, n_buckets=10):
    """Predicted vs actual win% by probability bucket."""
    bins = np.linspace(0, 1, n_buckets + 1)
    df = results.copy()
    df['bucket'] = pd.cut(df['elo_win_prob_home'], bins=bins, include_lowest=True)
    tab = df.groupby('bucket', observed=True).agg(
        n=('home_win', 'size'),
        predicted=('elo_win_prob_home', 'mean'),
        actual=('home_win', 'mean'),
    ).reset_index()
    return tab


def evaluate_and_report(results, output_path=None):
    """Compute metrics and print a report; optionally save to file."""
    y_true = results['home_win'].values
    y_prob = results['elo_win_prob_home'].values

    bs = brier_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    acc = accuracy(y_true, y_prob)

    # Reference baselines
    home_rate = y_true.mean()
    bs_home_always = brier_score(y_true, np.ones_like(y_true) * home_rate)
    bs_coinflip = brier_score(y_true, np.ones_like(y_true) * 0.5)

    cal = calibration_table(results)

    lines = [
        "=" * 60,
        "Elo Baseline — Walk-Forward Backtest",
        "=" * 60,
        f"Seasons evaluated: {results['season'].min()}-{results['season'].max()}",
        f"Games evaluated:   {len(results):,}",
        f"Home win rate:     {home_rate:.3f}",
        "",
        f"Brier score:       {bs:.4f}   (lower is better)",
        f"  vs home-baserate {bs_home_always:.4f}",
        f"  vs coin flip     {bs_coinflip:.4f}",
        f"Log loss:          {ll:.4f}",
        f"Accuracy (p>0.5):  {acc:.3f}",
        "",
        "Calibration:",
        cal.to_string(index=False),
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[OK] Saved report to {output_path}")

    return {'brier': bs, 'log_loss': ll, 'accuracy': acc, 'calibration': cal}


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    schedules_path = f"{config['data']['processed_dir']}/games_master.csv"
    print(f"Loading schedules from {schedules_path}...")
    schedules = pd.read_csv(schedules_path)
    print(f"  {len(schedules):,} games loaded")

    print("Running walk-forward Elo backtest...")
    results = backtest_elo(schedules, config)

    output_dir = config['data']['outputs_dir']
    report_path = f"{output_dir}/backtest_elo.txt"
    predictions_path = f"{output_dir}/elo_predictions.csv"

    metrics = evaluate_and_report(results, output_path=report_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(predictions_path, index=False)
    print(f"[OK] Saved predictions to {predictions_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
