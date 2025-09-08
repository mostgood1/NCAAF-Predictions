"""Regenerate season predictions overlaying latest model outputs.

Reads the existing enhanced predictions CSV, loads model artifacts (rf_v1_*),
adds model_* columns (points, total, margin, home_win_prob), and writes out a
new versioned CSV plus (optionally) an updated 'with_scores' file if actuals
present in source.

This does NOT retrain models; run src/modeling/retune_models.py first.
"""
from __future__ import annotations
import os, math, argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.data.weather_enrichment import enrich_dataframe

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

ENHANCED_FILE = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced.csv'
WITH_SCORES_FILE = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv'

OUT_STEM = 'college_football_schedule_2025_predicted_totals_enhanced'


def _load_models(prefix='rf_v1'):
    import joblib
    arts = {}
    for tgt in ['home_pts','away_pts','margin']:
        p = MODELS_DIR / f'{prefix}_{tgt}.joblib'
        if p.exists():
            try:
                arts[tgt] = joblib.load(p)
            except Exception as e:
                print(f"[warn] model load failed {tgt}: {e}")
    calib = MODELS_DIR / f'{prefix}_home_win_calibration.csv'
    if calib.exists():
        try:
            arts['calibration'] = pd.read_csv(calib)
        except Exception as e:
            print(f"[warn] calib load failed: {e}")
    return arts


def _interp_calib(calib_df: pd.DataFrame, margin_val: float):
    try:
        if calib_df is None or calib_df.empty or margin_val is None or math.isnan(margin_val):
            return None
        xs = calib_df['pred_margin'].values
        ys = calib_df['home_win_prob'].values
        if margin_val <= xs.min():
            return float(ys[xs.argmin()])
        if margin_val >= xs.max():
            return float(ys[xs.argmax()])
        import numpy as np
        idx = xs.searchsorted(margin_val)
        x0,x1 = xs[idx-1], xs[idx]
        y0,y1 = ys[idx-1], ys[idx]
        if x1 == x0:
            return float(y0)
        return float(y0 + (y1-y0)*((margin_val-x0)/(x1-x0)))
    except Exception:
        return None


def overlay(df: pd.DataFrame, arts: dict):
    if df.empty or not arts:
        return df
    feat_cols = ['predicted_home_points','predicted_away_points','predicted_total_points','weather_temp','weather_wind','weather_adjustment','edge','confidence']
    avail = [c for c in feat_cols if c in df.columns]
    if not avail:
        return df
    X = df[avail].fillna(0.0)
    def _apply(model_key, out_col):
        m = arts.get(model_key)
        if not m:
            return
        try:
            feats = [c for c in m['features'] if c in X.columns]
            df[out_col] = m['model'].predict(X[feats])
        except Exception as e:
            print(f"[warn] prediction failure {model_key}: {e}")
    _apply('home_pts','model_home_points')
    _apply('away_pts','model_away_points')
    if 'model_home_points' in df.columns and 'model_away_points' in df.columns:
        df['model_total_points'] = df['model_home_points'] + df['model_away_points']
    _apply('margin','model_margin')
    calib = arts.get('calibration')
    if 'model_margin' in df.columns and calib is not None:
        df['model_home_win_prob'] = df['model_margin'].apply(lambda v: _interp_calib(calib, v))
    # Derive model_edge & model_confidence
    try:
        if 'model_home_points' in df.columns and 'model_away_points' in df.columns:
            df['model_edge'] = (df['model_home_points'] - df['model_away_points']).abs()
        if 'model_home_win_prob' in df.columns:
            df['model_confidence_score'] = (df['model_home_win_prob'] - 0.5).abs() * 2.0
            def _tier(sc):
                try:
                    if sc is None or math.isnan(sc):
                        return None
                    if sc >= 0.40: return 'High'
                    if sc >= 0.25: return 'Medium'
                    if sc >= 0.15: return 'Low'
                    return 'Lean'
                except Exception:
                    return None
            df['model_confidence_tier'] = df['model_confidence_score'].apply(_tier)
            # Backfill legacy columns if missing or null
            if 'edge' in df.columns:
                mask = df['edge'].isna()
                if mask.any() and 'model_edge' in df.columns:
                    df.loc[mask, 'edge'] = df.loc[mask, 'model_edge']
            elif 'model_edge' in df.columns:
                df['edge'] = df['model_edge']
            if 'confidence' in df.columns:
                mask = df['confidence'].isna()
                if mask.any():
                    df.loc[mask, 'confidence'] = df.loc[mask, 'model_confidence_tier']
            else:
                df['confidence'] = df.get('model_confidence_tier')
    except Exception as e:
        print(f"[warn] model edge/confidence derivation failed: {e}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-prefix', default='rf_v1')
    ap.add_argument('--replace-predicted', action='store_true', help='Overwrite predicted_* columns with model_* values for front-end simplicity.')
    ap.add_argument('--out-suffix', default=None, help='Optional manual suffix for output filename (default uses timestamp).')
    args = ap.parse_args()

    if not ENHANCED_FILE.exists():
        print(json.dumps({'status':'error','reason':'missing_enhanced_file','path': str(ENHANCED_FILE)}))
        return
    df = pd.read_csv(ENHANCED_FILE)
    # Weather enrichment (fills weather_temp / weather_wind / weather_adjustment where missing)
    try:
        df = enrich_dataframe(df)
    except Exception as e:
        print(json.dumps({'status':'warn','phase':'enrichment','error':str(e)}))
    arts = _load_models(args.model_prefix)
    df = overlay(df, arts)
    if args.replace_predicted:
        # Only replace where model predictions exist to avoid wiping
        if 'model_home_points' in df.columns:
            df['predicted_home_points'] = df['model_home_points']
        if 'model_away_points' in df.columns:
            df['predicted_away_points'] = df['model_away_points']
        if 'model_total_points' in df.columns:
            df['predicted_total_points'] = df['model_total_points']
        if 'model_margin' in df.columns:
            df['predicted_win_margin'] = df['model_margin']
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    suffix = args.out_suffix or ts
    out_path = DATA_DIR / f'{OUT_STEM}_{suffix}.csv'
    df.to_csv(out_path, index=False)

    meta = {
        'status':'ok',
        'generated_at_utc': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        'rows': len(df),
        'output_file': out_path.name,
        'model_prefix': args.model_prefix,
        'replace_predicted': bool(args.replace_predicted),
    'missing_weather_rows': int((df['weather_temp'].isna() | df['weather_wind'].isna()).sum()) if {'weather_temp','weather_wind'}.issubset(df.columns) else None,
    'missing_edge_rows': int(df['edge'].isna().sum()) if 'edge' in df.columns else None,
    'missing_confidence_rows': int(df['confidence'].isna().sum()) if 'confidence' in df.columns else None,
    }
    print(json.dumps(meta))

if __name__ == '__main__':
    main()
