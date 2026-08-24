"""
Phase 5: Full score model.

Predict total_points (home + away) with LightGBM, then combine with the
Phase 4 margin regressor to derive individual home_score, away_score:
    home_score = (margin + total) / 2
    away_score = (total - margin) / 2
"""
import argparse
import pickle
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from src.models.train_winprob_spread import (
    select_feature_columns, walk_forward_train_eval, report_regressor,
    _lgb_kwargs, _split_train_val,
)


def combine_margin_and_total(margin_preds, total_preds):
    """
    Merge the two prediction DataFrames on game_id, derive home/away scores.
    """
    m = margin_preds[['game_id', 'season', 'week', 'home_team', 'away_team',
                      'y_true', 'y_pred']].rename(
        columns={'y_true': 'actual_margin', 'y_pred': 'pred_margin'})
    t = total_preds[['game_id', 'y_true', 'y_pred']].rename(
        columns={'y_true': 'actual_total', 'y_pred': 'pred_total'})

    df = m.merge(t, on='game_id', how='inner')
    df['pred_home_score'] = (df['pred_margin'] + df['pred_total']) / 2.0
    df['pred_away_score'] = (df['pred_total'] - df['pred_margin']) / 2.0
    df['actual_home_score'] = (df['actual_margin'] + df['actual_total']) / 2.0
    df['actual_away_score'] = (df['actual_total'] - df['actual_margin']) / 2.0
    return df


def report_scores(df, output_path=None):
    mae_home = mean_absolute_error(df['actual_home_score'], df['pred_home_score'])
    mae_away = mean_absolute_error(df['actual_away_score'], df['pred_away_score'])
    mae_margin = mean_absolute_error(df['actual_margin'], df['pred_margin'])
    mae_total = mean_absolute_error(df['actual_total'], df['pred_total'])

    winner_pred_correct = float(
        ((df['pred_margin'] > 0) == (df['actual_margin'] > 0)).mean()
    )

    lines = [
        "=" * 60,
        "Phase 5 — Full Score Model",
        "=" * 60,
        f"Games evaluated:   {len(df):,}",
        "",
        f"MAE home_score:    {mae_home:.3f}",
        f"MAE away_score:    {mae_away:.3f}",
        f"MAE margin:        {mae_margin:.3f}",
        f"MAE total:         {mae_total:.3f}",
        f"Winner-from-margin accuracy: {winner_pred_correct:.3f}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

    return {'mae_home': mae_home, 'mae_away': mae_away,
            'mae_margin': mae_margin, 'mae_total': mae_total,
            'winner_accuracy': winner_pred_correct}


def train_final_total_model(feats, config, output_dir):
    features = select_feature_columns(feats)
    lgb_params = config['lightgbm']
    early_stop = lgb_params.get('early_stopping_rounds', 50)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = feats.dropna(subset=['total'])
    tr_df, val_df = _split_train_val(data, 'total', val_frac=0.2)
    X_tr = tr_df[features].fillna(0); y_tr = tr_df['total'].values
    X_val = val_df[features].fillna(0); y_val = val_df['total'].values

    model = lgb.LGBMRegressor(**_lgb_kwargs(lgb_params))
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='l1',
              callbacks=[lgb.early_stopping(early_stop, verbose=False)])
    with open(f"{output_dir}/total_model.pkl", 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)
    print(f"[OK] Saved total_model.pkl to {output_dir}")


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    processed = config['data']['processed_dir']
    outputs = config['data']['outputs_dir']
    models_dir = config['data']['models_dir']

    print("Loading features...")
    feats = pd.read_parquet(f"{processed}/features.parquet")
    print(f"  {len(feats):,} games")

    # We need margin predictions from Phase 4 — either load them or re-run
    margin_pred_path = f"{outputs}/margin_predictions.csv"
    if not Path(margin_pred_path).exists():
        print("  margin_predictions.csv not found — run train_winprob_spread first.")
        return
    margin_preds = pd.read_csv(margin_pred_path)

    print("\nWalk-forward: total-points regressor")
    total_preds = walk_forward_train_eval(feats, config, target='total', model_type='regressor')
    total_preds.to_csv(f"{outputs}/total_predictions.csv", index=False)
    report_regressor(total_preds, output_path=f"{outputs}/backtest_total.txt", label='Total')

    print("\nCombining margin + total -> individual scores")
    combined = combine_margin_and_total(margin_preds, total_preds)
    combined.to_csv(f"{outputs}/score_predictions.csv", index=False)
    report_scores(combined, output_path=f"{outputs}/backtest_scores.txt")

    print("\nTraining final total-points model on all data...")
    train_final_total_model(feats, config, models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
