"""Build calibration artifacts for win probability & conference variance.

This script derives:
 1. Isotonic regression lookup table mapping predicted margin to calibrated home win probability.
 2. Conference-level empirical margin standard deviation (for potential UI spread bands).

Inputs:
  - Uses enhanced with scores CSV if available to gather actual results.
  - Requires columns: season, week, home_team, away_team, actual_home_points, actual_away_points,
    predicted_home_points, predicted_away_points

Outputs saved in models/ using provided --prefix (defaults to latest manifest prefix or rf_v1):
  - {prefix}_home_win_calibration.csv
  - conference_sigma_overrides.csv (written to data/)

This can be invoked via /api/build-calibration route.
"""
from __future__ import annotations
import os, json, math, argparse
from pathlib import Path
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from datetime import datetime, timezone

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)
WITH_SCORES = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv'
ENHANCED = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced.csv'
MANIFEST = MODELS_DIR / 'model_manifest.json'

def _suggest_prefix():
    if MANIFEST.exists():
        try:
            m = json.loads(MANIFEST.read_text(encoding='utf-8'))
            mp = m.get('model_prefix')
            if isinstance(mp, str) and mp:
                return mp
        except Exception:
            pass
    return 'rf_v1'

def load_df():
    if WITH_SCORES.exists():
        try:
            return pd.read_csv(WITH_SCORES)
        except Exception:
            pass
    if ENHANCED.exists():
        try:
            return pd.read_csv(ENHANCED)
        except Exception:
            pass
    return pd.DataFrame()

def build_calibration(df: pd.DataFrame, prefix: str):
    out = {}
    if df.empty:
        return {'status':'no_data'}
    needed = {'actual_home_points','actual_away_points','home_team','away_team'}
    if not needed.issubset(df.columns):
        return {'status':'missing_columns', 'needed': sorted(needed - set(df.columns))}
    # Actual margin
    df = df.copy()
    df['actual_margin'] = df['actual_home_points'] - df['actual_away_points']
    # Pred margin proxy from predictions if present
    if {'predicted_home_points','predicted_away_points'}.issubset(df.columns):
        df['pred_margin_pred'] = df['predicted_home_points'] - df['predicted_away_points']
    else:
        # fallback to zero (uncalibrated) => skip
        return {'status':'no_pred_columns'}
    df = df.dropna(subset=['actual_margin','pred_margin_pred'])
    if df.empty:
        return {'status':'no_rows_after_drop'}
    try:
        X = df['pred_margin_pred'].astype(float).values
        y = (df['actual_margin'] > 0).astype(int).values
        order = X.argsort()
        Xs = X[order]; ys = y[order]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(Xs, ys)
        grid = sorted(set([round(v,2) for v in list(pd.Series(Xs).quantile([i/50 for i in range(51)]))]))
        probs = iso.predict(grid)
        calib_df = pd.DataFrame({'pred_margin': grid, 'home_win_prob': probs})
        calib_path = MODELS_DIR / f'{prefix}_home_win_calibration.csv'
        calib_df.to_csv(calib_path, index=False)
        out['calibration'] = {'points': len(calib_df), 'path': calib_path.name}
    except Exception as e:
        out['calibration_error'] = str(e)

    # Conference variance (std of actual margin)
    try:
        if 'home_conference' in df.columns:
            grp = df.groupby('home_conference')['actual_margin'].std().reset_index().rename(columns={'actual_margin':'sigma_margin'})
            conf_file = DATA_DIR / 'conference_sigma_overrides.csv'
            grp.to_csv(conf_file, index=False)
            out['conference_sigma_file'] = conf_file.name
    except Exception as e:
        out['conference_sigma_error'] = str(e)
    out['status'] = 'ok'
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', help='Model prefix (e.g., rf_v3). Default = manifest prefix or rf_v1.')
    args = ap.parse_args()
    prefix = args.prefix or _suggest_prefix()
    df = load_df()
    res = build_calibration(df, prefix)
    print(json.dumps({'generated_at_utc': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), 'model_prefix': prefix, **res}))

if __name__ == '__main__':
    main()
