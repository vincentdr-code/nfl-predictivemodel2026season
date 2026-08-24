"""
Train LightGBM models for win probability (classification) and margin (regression).
Walk-forward validation: train on all seasons < S, evaluate on season S.
"""
import argparse
import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

from src.models.elo import brier_score, calibration_table, evaluate_and_report


# Columns that are TARGETS or leak the outcome — NEVER include in X.
LEAKY_COLS = {
    'home_win', 'margin', 'total',
    'home_score', 'away_score',
    'spread_line', 'total_line',  # Vegas lines: available pre-kickoff but treat as gold benchmarks
}

# Identifier / metadata columns
META_COLS = {
    'game_id', 'season', 'week', 'gameday',
    'home_team', 'away_team', 'home_qb_name', 'away_qb_name', 'roof',
}


def select_feature_columns(feats):
    """Return the list of columns safe to use as model features."""
    return [c for c in feats.columns
            if c not in LEAKY_COLS
            and c not in META_COLS
            and feats[c].dtype != 'O']  # drop object dtypes


def _lgb_kwargs(lgb_params):
    """Common LightGBM keyword arguments from config, excluding early-stopping."""
    return dict(
        n_estimators=lgb_params['n_estimators'],
        learning_rate=lgb_params['learning_rate'],
        max_depth=lgb_params['max_depth'],
        num_leaves=lgb_params['num_leaves'],
        min_child_samples=lgb_params.get('min_child_samples', 20),
        reg_alpha=lgb_params.get('reg_alpha', 0.0),
        reg_lambda=lgb_params.get('reg_lambda', 0.0),
        subsample=lgb_params['subsample'],
        subsample_freq=lgb_params.get('subsample_freq', 0),
        colsample_bytree=lgb_params['colsample_bytree'],
        verbose=-1,
        random_state=42,
    )


def _split_train_val(train_df, target, val_frac=0.2):
    """
    Split training data by season chronologically (last N seasons -> validation)
    to preserve the walk-forward assumption inside each fold.
    """
    seasons_sorted = sorted(train_df['season'].unique())
    n_val = max(1, int(round(len(seasons_sorted) * val_frac)))
    val_seasons = set(seasons_sorted[-n_val:])
    val_mask = train_df['season'].isin(val_seasons)
    return train_df[~val_mask], train_df[val_mask]


def walk_forward_train_eval(feats, config, target='home_win', model_type='classifier'):
    """
    Walk-forward with early stopping. For each test season S, train on
    seasons < S using last 20% of training seasons as validation for early
    stopping, then predict on season S.
    """
    start = config['backtest']['walk_forward_start_season']
    end = config['backtest']['walk_forward_end_season']
    lgb_params = config['lightgbm']
    early_stop = lgb_params.get('early_stopping_rounds', 50)

    features = select_feature_columns(feats)
    print(f"  Using {len(features)} features")

    all_preds = []
    for season in range(start, end + 1):
        train = feats[feats['season'] < season].dropna(subset=[target])
        test = feats[feats['season'] == season].dropna(subset=[target])
        if len(train) == 0 or len(test) == 0:
            continue

        tr_df, val_df = _split_train_val(train, target, val_frac=0.2)

        X_tr = tr_df[features].fillna(0); y_tr = tr_df[target].values
        X_val = val_df[features].fillna(0); y_val = val_df[target].values
        X_test = test[features].fillna(0); y_test = test[target].values

        if model_type == 'classifier':
            model = lgb.LGBMClassifier(**_lgb_kwargs(lgb_params))
            eval_metric = 'binary_logloss'
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=eval_metric,
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            model = lgb.LGBMRegressor(**_lgb_kwargs(lgb_params))
            eval_metric = 'l1'
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=eval_metric,
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
            y_pred = model.predict(X_test)

        preds = test[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
        preds['y_true'] = y_test
        preds['y_pred'] = y_pred
        all_preds.append(preds)

        best_iter = getattr(model, 'best_iteration_', None) or lgb_params['n_estimators']
        print(f"    season {season}: train {len(tr_df):,} / val {len(val_df):,} / test {len(test):,} (best_iter={best_iter})")

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def report_classifier(preds, elo_baseline=None, output_path=None):
    """Print + save Brier / log loss / AUC / calibration report."""
    y = preds['y_true'].astype(int).values
    p = preds['y_pred'].values

    bs = brier_score_loss(y, p)
    ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    auc = roc_auc_score(y, p)
    acc = float(((p > 0.5) == y).mean())

    tab = calibration_table(
        preds.rename(columns={'y_pred': 'elo_win_prob_home', 'y_true': 'home_win'})
    )

    lines = [
        "=" * 60,
        "Phase 4 — LightGBM Win Probability",
        "=" * 60,
        f"Seasons evaluated: {preds['season'].min()}-{preds['season'].max()}",
        f"Games evaluated:   {len(preds):,}",
        "",
        f"Brier score:       {bs:.4f}   (lower is better)",
        f"Log loss:          {ll:.4f}",
        f"AUC:               {auc:.4f}",
        f"Accuracy (p>0.5):  {acc:.3f}",
    ]

    if elo_baseline is not None:
        delta = elo_baseline - bs
        lines += [
            "",
            f"Elo baseline Brier: {elo_baseline:.4f}",
            f"Delta vs Elo:       {delta:+.4f}   ({'BEAT' if delta > 0 else 'WORSE THAN'} baseline)",
        ]

    lines += ["", "Calibration:", tab.to_string(index=False), "=" * 60]

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

    return {'brier': bs, 'log_loss': ll, 'auc': auc, 'accuracy': acc}


def report_regressor(preds, output_path=None, label='Margin'):
    y = preds['y_true'].values
    p = preds['y_pred'].values
    mae = mean_absolute_error(y, p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))

    # Baselines
    mae_zero = float(np.mean(np.abs(y)))
    mae_median = float(np.mean(np.abs(y - np.median(y))))

    lines = [
        "=" * 60,
        f"Phase 4 — LightGBM {label}",
        "=" * 60,
        f"Seasons evaluated: {preds['season'].min()}-{preds['season'].max()}",
        f"Games evaluated:   {len(preds):,}",
        "",
        f"MAE:               {mae:.3f}",
        f"RMSE:              {rmse:.3f}",
        f"  vs predict-zero  {mae_zero:.3f}",
        f"  vs predict-median {mae_median:.3f}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

    return {'mae': mae, 'rmse': rmse}


def train_final_models(feats, config, output_dir):
    """Train on ALL data with early stopping on last 20% (by season)."""
    features = select_feature_columns(feats)
    lgb_params = config['lightgbm']
    early_stop = lgb_params.get('early_stopping_rounds', 50)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _fit(target, obj):
        data = feats.dropna(subset=[target])
        tr_df, val_df = _split_train_val(data, target, val_frac=0.2)
        X_tr = tr_df[features].fillna(0); y_tr = tr_df[target].values
        X_val = val_df[features].fillna(0); y_val = val_df[target].values
        if obj == 'classifier':
            m = lgb.LGBMClassifier(**_lgb_kwargs(lgb_params))
            m.fit(X_tr, y_tr.astype(int), eval_set=[(X_val, y_val.astype(int))],
                  eval_metric='binary_logloss',
                  callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        else:
            m = lgb.LGBMRegressor(**_lgb_kwargs(lgb_params))
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='l1',
                  callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        return m

    clf = _fit('home_win', 'classifier')
    with open(f"{output_dir}/winprob_model.pkl", 'wb') as f:
        pickle.dump({'model': clf, 'features': features}, f)

    reg = _fit('margin', 'regressor')
    with open(f"{output_dir}/margin_model.pkl", 'wb') as f:
        pickle.dump({'model': reg, 'features': features}, f)

    print(f"[OK] Saved winprob_model.pkl + margin_model.pkl to {output_dir}")


def main(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    processed = config['data']['processed_dir']
    outputs = config['data']['outputs_dir']
    models_dir = config['data']['models_dir']

    print("Loading features...")
    feats = pd.read_parquet(f"{processed}/features.parquet")
    print(f"  {len(feats):,} games x {len(feats.columns)} cols")

    # Try to load Elo baseline Brier score for comparison
    elo_baseline = None
    elo_pred_path = f"{outputs}/elo_predictions.csv"
    if Path(elo_pred_path).exists():
        elo_preds = pd.read_csv(elo_pred_path)
        elo_baseline = brier_score_loss(elo_preds['home_win'].values,
                                        elo_preds['elo_win_prob_home'].values)
        print(f"  Elo baseline Brier: {elo_baseline:.4f}")

    print("\nWalk-forward: win probability classifier")
    winprob = walk_forward_train_eval(feats, config, target='home_win', model_type='classifier')
    winprob.to_csv(f"{outputs}/winprob_predictions.csv", index=False)
    report_classifier(winprob, elo_baseline=elo_baseline,
                      output_path=f"{outputs}/backtest_winprob.txt")

    print("\nWalk-forward: margin regressor")
    margin = walk_forward_train_eval(feats, config, target='margin', model_type='regressor')
    margin.to_csv(f"{outputs}/margin_predictions.csv", index=False)
    report_regressor(margin, output_path=f"{outputs}/backtest_margin.txt", label='Margin')

    print("\nTraining final models on all data...")
    train_final_models(feats, config, models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
