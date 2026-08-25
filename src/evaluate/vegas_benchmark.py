"""
Compare model predictions to Vegas closing lines.

For each game:
- model_spread = our predicted home margin (from margin regressor)
- vegas_spread = closing spread_line (negative = home favored)
- pick_home = model likes home team ATS if model_spread > (-vegas_spread)
- ATS result = did the home team cover the vegas_spread?
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def analyze(margin_preds_path, features_path, output_path):
    """
    Args:
        margin_preds_path: outputs/margin_predictions.csv
        features_path:     data/processed/features.parquet (for spread_line)
        output_path:       outputs/backtest_vs_vegas.txt

    Returns:
        dict of metrics: ats_wins, ats_losses, ats_pct,
                          agree_with_book_pct, disagree_pct
    """
    preds = pd.read_csv(margin_preds_path)  # game_id, y_true (actual margin), y_pred (model)
    feats = pd.read_parquet(features_path)[['game_id', 'spread_line']]

    df = preds.merge(feats, on='game_id', how='inner')
    df = df.dropna(subset=['spread_line'])

    # spread_line convention in nfl_data_py: home-team margin the line implies
    # (positive = home favored by that many pts). Verified via correlation
    # with actual home margin (should be strongly positive).
    df['vegas_home_line'] = df['spread_line']
    df['home_covered'] = (df['y_true'] > df['vegas_home_line']).astype(int)

    # Model pick ATS: does the model's margin beat the vegas line?
    df['model_pick_home_ats'] = (df['y_pred'] > df['vegas_home_line']).astype(int)
    df['ats_correct'] = (df['model_pick_home_ats'] == df['home_covered']).astype(int)

    # Vegas pick ATS: whichever side vegas favors
    df['vegas_pick_home'] = (df['vegas_home_line'] > 0).astype(int)
    df['vegas_ats_correct'] = (df['vegas_pick_home'] == df['home_covered']).astype(int)

    # Where model and vegas agreed vs disagreed
    df['agrees_with_book'] = (df['model_pick_home_ats'] == df['vegas_pick_home']).astype(int)

    # Metrics
    total = len(df)
    ats_pct = df['ats_correct'].mean()
    vegas_pct = df['vegas_ats_correct'].mean()

    agree_df = df[df['agrees_with_book'] == 1]
    disagree_df = df[df['agrees_with_book'] == 0]

    # By season
    seasonal = df.groupby('season').agg(
        n=('ats_correct', 'size'),
        model_ats_pct=('ats_correct', 'mean'),
        vegas_ats_pct=('vegas_ats_correct', 'mean'),
        avg_edge=('y_pred', lambda x: (x - df.loc[x.index, 'vegas_home_line']).abs().mean()),
    ).reset_index()

    lines = [
        "=" * 66,
        "Model vs Vegas — Against The Spread (ATS)",
        "=" * 66,
        f"Games with closing line: {total:,}",
        "",
        f"Model ATS record:   {int(df['ats_correct'].sum())}-{total - int(df['ats_correct'].sum())}  ({ats_pct*100:.1f}%)",
        f"Vegas ATS record:   {int(df['vegas_ats_correct'].sum())}-{total - int(df['vegas_ats_correct'].sum())}  ({vegas_pct*100:.1f}%)",
        f"Break-even needed:  52.4% (accounting for -110 juice)",
        "",
        "Where model agrees with book:",
        f"  Games:   {len(agree_df):,}",
        f"  ATS:     {agree_df['ats_correct'].mean()*100:.1f}%",
        "",
        "Where model disagrees with book (contrarian plays):",
        f"  Games:   {len(disagree_df):,}",
        f"  ATS:     {disagree_df['ats_correct'].mean()*100:.1f}%",
        "",
        "By season:",
        seasonal.to_string(index=False),
        "=" * 66,
    ]
    report = "\n".join(lines)
    print(report)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Also save per-game ATS detail for the dashboard
    detail_path = str(Path(output_path).with_suffix('.csv'))
    df[['game_id', 'season', 'week', 'home_team', 'away_team', 'y_true', 'y_pred',
        'vegas_home_line', 'home_covered', 'model_pick_home_ats',
        'ats_correct', 'vegas_ats_correct', 'agrees_with_book']].to_csv(detail_path, index=False)
    print(f"[OK] Saved detail to {detail_path}")

    return {
        'model_ats_pct': ats_pct,
        'vegas_ats_pct': vegas_pct,
        'games': total,
        'agree_ats_pct': agree_df['ats_correct'].mean() if len(agree_df) else 0,
        'disagree_ats_pct': disagree_df['ats_correct'].mean() if len(disagree_df) else 0,
    }


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    outputs = config['data']['outputs_dir']
    processed = config['data']['processed_dir']

    analyze(
        margin_preds_path=f"{outputs}/margin_predictions.csv",
        features_path=f"{processed}/features.parquet",
        output_path=f"{outputs}/backtest_vs_vegas.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
