"""
Train calibrated classifiers for ATS (against the spread) and Totals (Over/Under).

Data source:
- Uses data/college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv
  which contains frozen pregame predicted_* and actual scores for completed games.
- Uses data/college_football_betting_lines_2025.csv and/or embedded 'lines' column in
  the with_scores CSV to derive market spread/total. If multiple providers exist,
  the median spread and median total are used per game.

Outputs:
- models/ats_totals/
    ats_clf.joblib            Calibrated classifier for Home ATS cover probability
    totals_clf.joblib         Calibrated classifier for Over probability
    meta.json                 Feature list and training metadata

Notes:
- Pushes are removed from the training set (actual margin + line == 0 for ATS,
  actual total == OU for totals), as they refund.
- Features are selected from columns typically available in prediction frames; the
  script will gracefully subset to only present columns.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models' / 'ats_totals'
WITH_SCORES = DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv'
LINES_2025 = DATA_DIR / 'college_football_betting_lines_2025.csv'

# Ensure we can import app.py from repo root when running this script directly
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def _parse_lines_field(odds_str):
    """Parse serialized list-of-dicts from the 'lines' column."""
    try:
        if isinstance(odds_str, list):
            return odds_str
        if not isinstance(odds_str, str):
            return []
        s = odds_str.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            s2 = s.replace('""', '"')
            return json.loads(s2)
        except Exception:
            pass
        try:
            t = s
            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1]
            t = t.replace('""', '"')
            return json.loads(t)
        except Exception:
            pass
        try:
            import ast as _ast
            return _ast.literal_eval(s)
        except Exception:
            return []
    except Exception:
        return []


def _median_spread_total(lines_list: List[dict]) -> Tuple[float | None, float | None]:
    """Return (home_spread, total) as medians across providers.

    Spread definition: negative means home favored by abs(spread).
    """
    try:
        spreads = []
        totals = []
        for e in (lines_list or []):
            if not isinstance(e, dict):
                continue
            sp = e.get('spread')
            if sp is None:
                sp = e.get('line')
            ou = e.get('overUnder')
            if ou is None:
                ou = e.get('total')
            if isinstance(sp, (int, float)) and np.isfinite(sp):
                spreads.append(float(sp))
            if isinstance(ou, (int, float)) and np.isfinite(ou):
                totals.append(float(ou))
        h_spread = float(np.median(spreads)) if spreads else None
        tot = float(np.median(totals)) if totals else None
        return h_spread, tot
    except Exception:
        return None, None


def _norm_team_for_odds(name: str) -> str:
    try:
        s = str(name or '')
        s = s.strip().lower()
        s = s.replace('&', 'and')
        s = s.replace("ʻ", "'").replace("’", "'")
        import re as _re
        s = _re.sub(r"[^a-z0-9 '\-]", " ", s)
        s = s.replace("hawai'i", "hawaii")
        s = _re.sub(r"\s+", " ", s).strip()
        return s
    except Exception:
        return str(name or '')


FEATURE_CANDIDATES = [
    # Model-based
    'model_margin', 'model_total_points', 'model_home_points', 'model_away_points', 'model_confidence_score',
    # Baseline predictions
    'predicted_home_points', 'predicted_away_points', 'predicted_total_points',
    # Weather raw
    'weather_temp', 'weather_wind', 'weather_adjustment',
    # Engineered: line deltas and tempo proxies (added if computed below)
    'delta_total_pred', 'delta_total_model', 'abs_delta_total_pred', 'abs_delta_total_model',
    'tempo_proxy_pred', 'tempo_proxy_model', 'spread_abs',
    # Weather transforms/interactions
    'temp_center', 'temp2', 'wind2', 'cold', 'heat', 'temp_wind',
    # Context flags
    'is_neutral', 'is_conference', 'week',
    # Legacy features if present
    'edge', 'confidence',
]


def _prepare_frame() -> pd.DataFrame:
    if not WITH_SCORES.exists():
        raise RuntimeError(f"with_scores file missing: {WITH_SCORES}")
    df = pd.read_csv(WITH_SCORES)
    # Merge latest enhanced predictions snapshot to bring model_* and weather_* columns
    try:
        import glob as _glob
        cands = [p for p in _glob.glob(str(DATA_DIR / 'college_football_schedule_2025_predicted_totals_enhanced*.csv'))
                 if not p.endswith('with_scores.csv') and 'full_static' not in p]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p))
            enh = pd.read_csv(cands[-1])
            # Reduce to essential columns to avoid duplicate suffix issues
            keep_cols = [c for c in enh.columns if c in (
                'season','week','home_team','away_team',
                'model_home_points','model_away_points','model_total_points','model_margin','model_confidence_score',
                'weather_temp','weather_wind','weather_adjustment',
                'neutral_site','conference_game'
            )]
            enh_small = enh[keep_cols].copy()
            # Deduplicate by taking last occurrence
            enh_small = enh_small.drop_duplicates(subset=['season','week','home_team','away_team'], keep='last')
            df = df.merge(enh_small, on=['season','week','home_team','away_team'], how='left', suffixes=('', '_m'))
    except Exception:
        pass
    # Compute provider medians per game using 2025 lines CSV (primary source)
    df['_med_spread'] = np.nan
    df['_med_total'] = np.nan
    if LINES_2025.exists():
        try:
            ldf = pd.read_csv(LINES_2025)
            # Derive medians per event
            ms = []
            mt = []
            norm_home = []
            norm_away = []
            for _, r in ldf.iterrows():
                ls = _parse_lines_field(r.get('lines'))
                sp, ou = _median_spread_total(ls)
                ms.append(sp)
                mt.append(ou)
                norm_home.append(_norm_team_for_odds(r.get('homeTeam')))
                norm_away.append(_norm_team_for_odds(r.get('awayTeam')))
            ldf['_med_spread'] = ms
            ldf['_med_total'] = mt
            ldf['_home_norm'] = norm_home
            ldf['_away_norm'] = norm_away
            # Reduce to one row per matchup by median (in case of duplicates)
            grp = ldf.groupby(['year','week','_home_norm','_away_norm'], as_index=False)[['_med_spread','_med_total']].median()
            # Prepare join keys on df and merge (season on left vs year on right)
            df['_home_norm'] = df['home_team'].apply(_norm_team_for_odds)
            df['_away_norm'] = df['away_team'].apply(_norm_team_for_odds)
            df = df.merge(grp, left_on=['season','week','_home_norm','_away_norm'], right_on=['year','week','_home_norm','_away_norm'], how='left', suffixes=('', '_y'))
            # Backfill medians
            if '_med_spread_y' in df.columns:
                df['_med_spread'] = df['_med_spread'].fillna(df['_med_spread_y'])
            if '_med_total_y' in df.columns:
                df['_med_total'] = df['_med_total'].fillna(df['_med_total_y'])
            # Cleanup
            for c in ['_med_spread_y','_med_total_y','year']:
                if c in df.columns:
                    try:
                        df.drop(columns=[c], inplace=True)
                    except Exception:
                        pass
        except Exception:
            pass

    # Build actual stats
    df['_actual_margin'] = pd.to_numeric(df.get('actual_home_points'), errors='coerce') - pd.to_numeric(df.get('actual_away_points'), errors='coerce')
    df['_actual_total'] = pd.to_numeric(df.get('actual_home_points'), errors='coerce') + pd.to_numeric(df.get('actual_away_points'), errors='coerce')
    # Fallback: fill missing medians using app.get_betting_lines for completed games
    try:
        import app as webapp
        mask = df['_med_spread'].isna() | df['_med_total'].isna()
        # limit to completed games to speed up
        mask = mask & df['_actual_total'].notna()
        if mask.any():
            idxs = list(df[mask].index)
            for i in idxs:
                r = df.loc[i]
                try:
                    lines = webapp.get_betting_lines(int(r.get('season', 2025)), int(r.get('week', 0)), r.get('home_team'), r.get('away_team'))
                except Exception:
                    lines = []
                if lines:
                    sp = []
                    ou = []
                    for e in lines:
                        sv = e.get('spread')
                        tv = e.get('overUnder')
                        if isinstance(sv, (int,float)) and np.isfinite(sv):
                            sp.append(float(sv))
                        if isinstance(tv, (int,float)) and np.isfinite(tv):
                            ou.append(float(tv))
                    if sp:
                        df.at[i, '_med_spread'] = float(np.median(sp))
                    if ou:
                        df.at[i, '_med_total'] = float(np.median(ou))
    except Exception:
        pass
    return df
    


def _fit_calibrated_gbc(X: np.ndarray, y: np.ndarray, min_samples: int = 100):
    if X.shape[0] < max(min_samples, 20):
        return None
    base = GradientBoostingClassifier(random_state=42)
    # If very small sample, use sigmoid calibration to avoid isotonic overfit
    method = 'isotonic' if X.shape[0] >= 300 else 'sigmoid'
    try:
        clf = CalibratedClassifierCV(base, cv=3, method=method)
        clf.fit(X, y)
        return clf
    except Exception:
        try:
            base.fit(X, y)
            return base
        except Exception:
            return None


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare_frame()

    # --- Engineer additional features for Totals performance (safe if columns missing) ---
    def _num(x):
        try:
            v = float(x)
            if not np.isfinite(v):
                return np.nan
            return v
        except Exception:
            return np.nan
    # Line deltas
    if '_med_total' in df.columns:
        if 'predicted_total_points' in df.columns:
            df['delta_total_pred'] = pd.to_numeric(df['predicted_total_points'], errors='coerce') - pd.to_numeric(df['_med_total'], errors='coerce')
            df['abs_delta_total_pred'] = df['delta_total_pred'].abs()
        if 'model_total_points' in df.columns:
            df['delta_total_model'] = pd.to_numeric(df['model_total_points'], errors='coerce') - pd.to_numeric(df['_med_total'], errors='coerce')
            df['abs_delta_total_model'] = df['delta_total_model'].abs()
    # Spread magnitude
    if '_med_spread' in df.columns:
        df['spread_abs'] = pd.to_numeric(df['_med_spread'], errors='coerce').abs()
    # Tempo proxies using OU and spread magnitude
    try:
        if 'predicted_total_points' in df.columns and '_med_spread' in df.columns:
            df['tempo_proxy_pred'] = pd.to_numeric(df['predicted_total_points'], errors='coerce') - 0.5 * pd.to_numeric(df['_med_spread'], errors='coerce').abs()
        if 'model_total_points' in df.columns and '_med_spread' in df.columns:
            df['tempo_proxy_model'] = pd.to_numeric(df['model_total_points'], errors='coerce') - 0.5 * pd.to_numeric(df['_med_spread'], errors='coerce').abs()
    except Exception:
        pass
    # Weather transforms
    if 'weather_temp' in df.columns:
        t = pd.to_numeric(df['weather_temp'], errors='coerce')
        df['temp_center'] = t - 60.0
        df['temp2'] = (t - 60.0) ** 2
        df['cold'] = (50.0 - t).clip(lower=0)
        df['heat'] = (t - 80.0).clip(lower=0)
    if 'weather_wind' in df.columns:
        w = pd.to_numeric(df['weather_wind'], errors='coerce')
        df['wind2'] = w ** 2
    if 'weather_temp' in df.columns and 'weather_wind' in df.columns:
        t = pd.to_numeric(df['weather_temp'], errors='coerce')
        w = pd.to_numeric(df['weather_wind'], errors='coerce')
        df['temp_wind'] = t * w
    # Context flags (if present)
    for col, out in [('neutral_site','is_neutral'), ('conference_game','is_conference')]:
        if col in df.columns:
            df[out] = df[col].astype(float).fillna(0.0)
    # Ensure week is numeric if present
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce')

    # Feature matrix
    feat_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not feat_cols:
        raise RuntimeError("No usable feature columns found for ATS/Totals training")

    # Diagnostics: coverage
    cov = {
        'rows': int(len(df)),
        'with_spread_any': int(df['_med_spread'].notna().sum()),
        'with_total_any': int(df['_med_total'].notna().sum()),
        'completed': int((df['_actual_total'].notna()).sum()),
        'completed_with_spread': int((df['_actual_total'].notna() & df['_med_spread'].notna()).sum()),
        'completed_with_total': int((df['_actual_total'].notna() & df['_med_total'].notna()).sum()),
    }

    # ATS dataset: remove rows without spread or actuals; drop pushes
    ats_df = df[df['_med_spread'].notna() & df['_actual_margin'].notna()].copy()
    ats_df['_ats_margin'] = ats_df['_actual_margin'] + ats_df['_med_spread']
    ats_df = ats_df[ats_df['_ats_margin'] != 0]
    X_ats = ats_df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y_ats = (ats_df['_ats_margin'] > 0).astype(int).values  # 1 = Home covers

    # Totals dataset: remove rows without OU or actuals; drop pushes
    tot_df = df[df['_med_total'].notna() & df['_actual_total'].notna()].copy()
    tot_df = tot_df[tot_df['_actual_total'] != tot_df['_med_total']]
    X_tot = tot_df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y_tot = (tot_df['_actual_total'] > tot_df['_med_total']).astype(int).values  # 1 = Over

    results = {
        'features': feat_cols,
        'sizes': {
            'ats': int(len(X_ats)),
            'totals': int(len(X_tot)),
            'total_rows': int(len(df)),
        },
        'coverage': cov,
        'notes': 'Probabilities: ATS -> P(Home covers), Totals -> P(Over). Pushes dropped.'
    }

    # Fit models if sufficient data exists
    ats_clf = _fit_calibrated_gbc(X_ats, y_ats, min_samples=80)
    totals_clf = _fit_calibrated_gbc(X_tot, y_tot, min_samples=80)

    if ats_clf is not None:
        joblib.dump(ats_clf, MODELS_DIR / 'ats_clf.joblib')
        results['ats_model'] = 'saved'
    else:
        results['ats_model'] = 'insufficient_data'

    if totals_clf is not None:
        joblib.dump(totals_clf, MODELS_DIR / 'totals_clf.joblib')
        results['totals_model'] = 'saved'
    else:
        results['totals_model'] = 'insufficient_data'

    # Persist metadata
    with open(MODELS_DIR / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
