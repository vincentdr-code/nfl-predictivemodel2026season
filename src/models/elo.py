"""
Elo rating system for NFL teams.
Implements the FiveThirtyEight methodology for NFL Elo ratings.
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class EloRating:
    """
    Tracks and updates Elo ratings for all NFL teams.
    """

    def __init__(self, initial_rating=1500, k_factor=20, home_advantage=65,
                 season_reversion_factor=0.333, min_rating=1000, max_rating=2000):
        """
        Initialize Elo system.

        Args:
            initial_rating: starting Elo (typically 1500)
            k_factor: how much ratings change per game
            home_advantage: Elo points added to home team pre-game
            season_reversion_factor: fraction to revert toward 1500 between seasons
            min_rating, max_rating: bounds on ratings
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_reversion_factor = season_reversion_factor
        self.min_rating = min_rating
        self.max_rating = max_rating

        # Initialize all 32 teams
        self.ratings = {
            team: initial_rating for team in [
                'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                'LA', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',
                'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
            ]
        }

    def get_win_probability(self, elo_diff):
        """
        Convert Elo difference to win probability using log5 formula.

        Args:
            elo_diff: home_elo - away_elo

        Returns:
            Probability that home team wins [0, 1]
        """
        return 1.0 / (10.0 ** (-elo_diff / 400.0) + 1.0)

    def margin_of_victory_multiplier(self, margin, elo_diff):
        """
        FiveThirtyEight's MOV multiplier for Elo update magnitude.

        Args:
            margin: absolute margin of victory (winner_score - loser_score)
            elo_diff: winner_elo - loser_elo

        Returns:
            Multiplier on K-factor
        """
        return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))

    def update_rating(self, winner_elo, loser_elo, margin):
        """
        Update winner and loser ratings after a game.

        Args:
            winner_elo: pre-game rating of winning team
            loser_elo: pre-game rating of losing team
            margin: margin of victory (winner_score - loser_score)

        Returns:
            tuple (new_winner_elo, new_loser_elo)
        """
        elo_diff = winner_elo - loser_elo
        mov_mult = self.margin_of_victory_multiplier(margin, elo_diff)
        rating_change = self.k_factor * mov_mult

        # Winner gains, loser loses
        new_winner_elo = winner_elo + rating_change
        new_loser_elo = loser_elo - rating_change

        # Clamp to bounds
        new_winner_elo = np.clip(new_winner_elo, self.min_rating, self.max_rating)
        new_loser_elo = np.clip(new_loser_elo, self.min_rating, self.max_rating)

        return new_winner_elo, new_loser_elo

    def process_game(self, home_team, away_team, home_score, away_score):
        """
        Update ratings after a game is played.

        Args:
            home_team, away_team: team codes
            home_score, away_score: final scores

        Returns:
            dict with updated ratings and win probability
        """
        home_elo_pre = self.ratings[home_team]
        away_elo_pre = self.ratings[away_team]

        # Home advantage built into Elo diff for probability
        elo_diff = (home_elo_pre + self.home_advantage) - away_elo_pre
        home_win_prob = self.get_win_probability(elo_diff)

        # Determine winner
        if home_score > away_score:
            winner = 'home'
            margin = home_score - away_score
            new_home_elo, new_away_elo = self.update_rating(
                home_elo_pre, away_elo_pre, margin
            )
        else:
            winner = 'away'
            margin = away_score - home_score
            new_away_elo, new_home_elo = self.update_rating(
                away_elo_pre, home_elo_pre, margin
            )

        # Update ratings
        self.ratings[home_team] = new_home_elo
        self.ratings[away_team] = new_away_elo

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_elo_pre': home_elo_pre,
            'away_elo_pre': away_elo_pre,
            'home_elo_post': new_home_elo,
            'away_elo_post': new_away_elo,
            'elo_win_prob_home': home_win_prob,
            'winner': winner,
        }

    def revert_season(self):
        """Regress all ratings toward 1500 at end of season."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.season_reversion_factor) +
                self.initial_rating * self.season_reversion_factor
            )


def backtest_elo(games_master_df, config_file="config.yaml"):
    """
    Walk-forward backtest of Elo model.

    Args:
        games_master_df: DataFrame from make_dataset.py
        config_file: path to config.yaml

    Returns:
        DataFrame with game-level predictions and actuals
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    elo_config = config['elo']
    elo = EloRating(**elo_config)

    backtest_start = config['backtest']['walk_forward_start_season']
    backtest_end = config['backtest']['walk_forward_end_season']

    # Sort games by season and week
    games = games_master_df.sort_values(['season', 'week']).reset_index(drop=True)

    results = []

    for season in range(backtest_start, backtest_end + 1):
        # Reset Elo at start of each season
        elo = EloRating(**elo_config)

        season_games = games[games['season'] < season].copy()

        # Train (play through) all games before this season to get ratings
        for _, game in season_games.iterrows():
            elo.process_game(
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score']
            )

        # Revert at end of prior season
        elo.revert_season()

        # Evaluate on games in this season
        eval_games = games[games['season'] == season].copy()
        for _, game in eval_games.iterrows():
            result = elo.process_game(
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score']
            )
            result['season'] = season
            result['week'] = game['week']
            result['actual_winner'] = 'home' if game['home_score'] > game['away_score'] else 'away'
            result['home_score'] = game['home_score']
            result['away_score'] = game['away_score']
            results.append(result)

    return pd.DataFrame(results)


def compute_brier_score(results_df):
    """
    Brier score: mean squared difference between predicted and actual.

    Args:
        results_df: DataFrame from backtest_elo() with elo_win_prob_home and actual_winner

    Returns:
        Brier score (lower is better; 0.25 = random, ~0.21 = good NFL model)
    """
    home_win_actual = (results_df['actual_winner'] == 'home').astype(float)
    predictions = results_df['elo_win_prob_home']
    return ((predictions - home_win_actual) ** 2).mean()


def compute_calibration(results_df, n_buckets=10):
    """
    Calibration curve: predicted win% by probability bucket vs. actual.

    Args:
        results_df: DataFrame from backtest_elo()
        n_buckets: number of probability buckets

    Returns:
        DataFrame with predicted_prob, actual_prob, count per bucket
    """
    results_df['pred_bucket'] = pd.cut(
        results_df['elo_win_prob_home'],
        bins=np.linspace(0, 1, n_buckets + 1)
    )

    home_win_actual = (results_df['actual_winner'] == 'home').astype(float)

    calibration = results_df.groupby('pred_bucket', observed=True).agg({
        'elo_win_prob_home': 'mean',
        'home_team': 'count'
    }).rename(columns={'home_team': 'count'})

    calibration['actual_win_rate'] = results_df.groupby(
        'pred_bucket', observed=True
    ).apply(lambda x: home_win_actual[x.index].mean())

    return calibration


if __name__ == "__main__":
    # Stub: backtest will run from Claude Code in Phase 2
    print("Elo model defined. Run from make_dataset output.")
