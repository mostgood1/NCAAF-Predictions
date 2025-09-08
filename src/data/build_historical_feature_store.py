"""Build a consolidated historical feature store without fabricating data.

Combines available multi-year feature/lines dataset with current season enhanced-with-scores
file (if present) into a normalized CSV: data/historical_feature_store.csv.

Rules:
 - Only include columns that already exist; never fabricate weather or predictions.
 - Safe derivation: predicted_total_points if both predicted_home_points & predicted_away_points exist.
 - Drop exact duplicate (season, week, home_team, away_team, start_date) rows keeping first.
 - Provide JSON summary to stdout.

Intended for model retraining expansion once richer multi-season engineered features exist.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUT_FILE = DATA_DIR / 'historical_feature_store.csv'

MULTI_YR_FILE = DATA_DIR / 'ncaa_games_last_15_years_features_with_lines.csv'
CURR_WITH_SCORES = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv'

KEY_COLS = ['season','week','home_team','away_team','start_date']
OPTIONAL_COLS = [
    'actual_home_points','actual_away_points',
    'predicted_home_points','predicted_away_points','predicted_total_points',
    'weather_temp','weather_wind','weather_adjustment','edge','confidence'
]

def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in KEY_COLS + OPTIONAL_COLS if c in df.columns]
    slim = df[cols].copy()
    # Derive predicted_total_points only if both components exist & total missing
    if 'predicted_total_points' not in slim.columns and {'predicted_home_points','predicted_away_points'}.issubset(slim.columns):
        try:
            slim['predicted_total_points'] = slim['predicted_home_points'] + slim['predicted_away_points']
        except Exception:
            pass
    return slim

def build():
    parts = []
    multi = _normalize(_load(MULTI_YR_FILE))
    if not multi.empty:
        parts.append(multi)
    curr = _normalize(_load(CURR_WITH_SCORES))
    if not curr.empty:
        parts.append(curr)
    if not parts:
        print(json.dumps({'status':'no_data','output_file': OUT_FILE.name}))
        return
    df = pd.concat(parts, ignore_index=True, copy=False)
    # Drop duplicates
    key_subset = [c for c in KEY_COLS if c in df.columns]
    before = len(df)
    if key_subset:
        df = df.sort_values(by=key_subset).drop_duplicates(subset=key_subset, keep='first')
    after = len(df)
    df.to_csv(OUT_FILE, index=False)
    summary = {
        'status': 'ok',
        'output_file': OUT_FILE.name,
        'rows': after,
        'dropped_duplicates': before - after,
        'columns_present': sorted(df.columns.tolist()),
        'missing_weather_rows': int((df['weather_temp'].isna() | df['weather_wind'].isna()).sum()) if {'weather_temp','weather_wind'}.issubset(df.columns) else None,
    }
    print(json.dumps(summary))

if __name__ == '__main__':
    build()
