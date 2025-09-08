"""Retune forecasting models using all finalized games to date.

Pipeline steps:
1. Load historical game-level features & outcomes.
2. Split into train/validation by season/week cutoff (time-aware).
3. Train/update separate models for:
   - Win margin (regression)
   - Home score (regression)
   - Away score (regression)
4. Calibrate probability of home win from margin distribution (isotonic).
5. Persist models & calibration artifacts under models/.
6. Emit brief metrics JSON for dashboard consumption.

Assumptions:
- A consolidated feature CSV will be produced later; for now we derive minimal features from current season predictions with actual scores present.
- This is a bootstrap retune; future enhancement should use multi-year feature store.
"""
from __future__ import annotations
import os, json, math
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

RAW_PRED_FILE = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv'
FALLBACK_PRED_FILE = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced.csv'

OUT_PREFIX = 'rf_v1'  # default; can be overridden via --out-prefix

MIN_FINAL_GAMES = 80  # Require enough signal before overwriting models
RANDOM_STATE = 42

TARGETS = {
    'margin': ('actual_home_points', 'actual_away_points'),
    'home_pts': ('actual_home_points',),
    'away_pts': ('actual_away_points',),
}

FEATURE_COLS_BASE = [
    'predicted_home_points','predicted_away_points','predicted_total_points',
    'weather_temp','weather_wind','weather_adjustment','edge','confidence'
]

# Columns that will be engineered if absent
ENGINEER_IF_ABSENT = {
    'predicted_total_points': lambda df: df.get('predicted_home_points') + df.get('predicted_away_points'),
    'edge': lambda df: (df.get('predicted_home_points') - df.get('predicted_away_points')).abs(),
}

CALIBRATION_POINTS = 200


def _load_df() -> pd.DataFrame:
    if RAW_PRED_FILE.exists():
        df = pd.read_csv(RAW_PRED_FILE)
    else:
        df = pd.read_csv(FALLBACK_PRED_FILE) if FALLBACK_PRED_FILE.exists() else pd.DataFrame()
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Keep only rows with actual scores
    if not {'actual_home_points','actual_away_points'}.issubset(df.columns):
        return pd.DataFrame()
    df = df[df['actual_home_points'].notna() & df['actual_away_points'].notna()].copy()
    # Basic feature engineering
    for col, fn in ENGINEER_IF_ABSENT.items():
        if col not in df.columns and all(c in df.columns for c in ['predicted_home_points','predicted_away_points']):
            try:
                df[col] = fn(df)
            except Exception:
                df[col] = pd.NA
    if 'weather_temp' not in df.columns:
        df['weather_temp'] = pd.NA
    if 'weather_wind' not in df.columns:
        df['weather_wind'] = pd.NA
    if 'weather_adjustment' not in df.columns:
        df['weather_adjustment'] = 0.0
    if 'confidence' not in df.columns:
        df['confidence'] = 0.5
    # Margin target
    df['margin'] = df['actual_home_points'] - df['actual_away_points']
    # Time ordering for temporal split: use start_date or fallback to index
    if 'start_date' in df.columns:
        try:
            df['_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
        except Exception:
            df['_dt'] = pd.NaT
    else:
        df['_dt'] = pd.NaT
    # Fallback ordering
    df['_order'] = range(len(df))
    df = df.sort_values(by=['_dt','_order'])
    return df


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if df.empty:
        return df, df
    n = len(df)
    cut = max(int(n*(1-val_frac)), 1)
    train = df.iloc[:cut].copy()
    val = df.iloc[cut:].copy()
    return train, val


def _fit_rf(train: pd.DataFrame, target: str) -> RandomForestRegressor:
    feat_cols = [c for c in FEATURE_COLS_BASE if c in train.columns]
    X = train[feat_cols].fillna(0.0)
    y = train[target]
    model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y)
    return model


def _evaluate(model, df: pd.DataFrame, target: str) -> dict:
    if df.empty:
        return {'mae': None, 'rmse': None, 'count': 0}
    feat_cols = [c for c in FEATURE_COLS_BASE if c in df.columns]
    X = df[feat_cols].fillna(0.0)
    y = df[target]
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = math.sqrt(mean_squared_error(y, preds))
    return {'mae': mae, 'rmse': rmse, 'count': len(df)}


def _calibrate_home_win_prob(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    # Use margin distribution to approximate home win probability; calibrate via isotonic
    df = pd.concat([train, val], ignore_index=True)
    if df.empty:
        return {'status': 'skipped', 'reason': 'no_data'}
    if 'margin' not in df.columns:
        return {'status': 'skipped', 'reason': 'no_margin'}
    # Empirical probability: home win if margin > 0
    df['home_win'] = (df['margin'] > 0).astype(int)
    # Feature: predicted margin proxy = predicted_home_points - predicted_away_points
    if not {'predicted_home_points','predicted_away_points'}.issubset(df.columns):
        return {'status': 'skipped', 'reason': 'no_pred_columns'}
    df['pred_margin_pred'] = df['predicted_home_points'] - df['predicted_away_points']
    X = df['pred_margin_pred'].astype(float).values
    y = df['home_win'].values
    # Sort by X for isotonic
    order = X.argsort()
    X_sorted = X[order]
    y_sorted = y[order]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(X_sorted, y_sorted)
    # Persist calibration table by sampling a grid
    grid = sorted(set([round(v,2) for v in list(pd.Series(X).quantile([i/20 for i in range(21)]))]))
    probs = iso.predict(grid)
    calib = pd.DataFrame({'pred_margin': grid, 'home_win_prob': probs})
    calib_path = MODELS_DIR / f'{OUT_PREFIX}_home_win_calibration.csv'
    calib.to_csv(calib_path, index=False)
    return {'status': 'ok', 'points': len(calib), 'path': calib_path.name}


def main():
    import argparse
    global OUT_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-prefix', help='Versioned model prefix to use when saving artifacts (e.g., rf_v2).')
    ap.add_argument('--min-final-games', type=int, default=MIN_FINAL_GAMES, help='Minimum finalized games required to retrain.')
    args = ap.parse_args()
    if args.out_prefix:
        OUT_PREFIX = args.out_prefix
    # Allow dynamic threshold
    min_required = args.min_final_games
    df = _prepare(_load_df())
    if len(df) < min_required:
        print(json.dumps({'status': 'skipped', 'reason': 'insufficient_final_games', 'count': len(df), 'min_required': min_required, 'model_prefix': OUT_PREFIX}))
        return
    train, val = _train_val_split(df)
    results = {'generated_at_utc': datetime.utcnow().isoformat()+'Z', 'counts': {'train': len(train), 'val': len(val)}}

    # Train models
    for tgt, cols in TARGETS.items():
        target_col = cols[0]  # first element
        if target_col not in train.columns:
            continue
        model = _fit_rf(train, target_col)
        metrics_train = _evaluate(model, train, target_col)
        metrics_val = _evaluate(model, val, target_col)
        out_path = MODELS_DIR / f'{OUT_PREFIX}_{tgt}.joblib'
        try:
            import joblib
            joblib.dump({'model': model, 'target': target_col, 'features': [c for c in FEATURE_COLS_BASE if c in train.columns]}, out_path)
            results[f'{tgt}_model'] = {'path': out_path.name, 'train': metrics_train, 'val': metrics_val}
        except Exception as e:
            results[f'{tgt}_model'] = {'error': str(e)}

    # Calibration
    results['home_win_calibration'] = _calibrate_home_win_prob(train, val)

    # Persist metrics
    metrics_path = MODELS_DIR / f'{OUT_PREFIX}_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({'status': 'ok', 'metrics_file': metrics_path.name, 'model_prefix': OUT_PREFIX, **results}))

if __name__ == '__main__':
    main()
