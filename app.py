import os

# -------------------- Build / Version Introspection --------------------
import time as _time
def _get_git_commit() -> str:
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            cand = os.path.join(base, '.git')
            if os.path.isdir(cand):
                head_path = os.path.join(cand, 'HEAD')
                if os.path.exists(head_path):
                    with open(head_path, 'r', encoding='utf-8') as f:
                        ref = f.read().strip()
                    if ref.startswith('ref:'):
                        ref_rel = ref.split(' ', 1)[1].strip()
                        ref_file = os.path.join(cand, *ref_rel.split('/'))
                        if os.path.exists(ref_file):
                            with open(ref_file, 'r', encoding='utf-8') as rf:
                                return rf.read().strip()[:40]
                    return ref[:40]
            new_base = os.path.dirname(base)
            if new_base == base:
                break
            base = new_base
    except Exception:
        pass
    return 'unknown'

BUILD_TIME = _time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime())
BUILD_COMMIT = _get_git_commit()
from flask import Flask, render_template_string, request, redirect, url_for, jsonify, make_response
import ast
import unicodedata
import pytz
from datetime import datetime
import pandas as pd
import os
import math
import json
import subprocess
from datetime import timezone
import sys
import re
import threading
import time
import joblib
from pathlib import Path
import glob

app = Flask(__name__)

# --- Early trivial health route to test server wiring even if later code errors ---
@app.route('/api/ping')
def api_ping():  # pragma: no cover
    return {'pong': True, 'commit': BUILD_COMMIT, 'build_time': BUILD_TIME}

# Resolve paths relative to this file, so it works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Load simple .env files (no external dependency) before reading environment variables
def _load_env_file(path: str):
    try:
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '=' not in s:
                    continue
                k, v = s.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        pass

# Load .env (if present) and a local override file
_load_env_file(os.path.join(BASE_DIR, '.env'))
_load_env_file(os.path.join(BASE_DIR, '.env.local'))

# Secrets fallback (non-committed): load odds api key from secrets/ if env var absent
try:
    if 'ODDS_API_KEY' not in os.environ:
        skey_path = os.path.join(BASE_DIR, 'secrets', 'odds_api_key.txt')
        if os.path.exists(skey_path):
            with open(skey_path, 'r', encoding='utf-8') as f:
                val = f.read().strip()
            if val:
                os.environ['ODDS_API_KEY'] = val
except Exception:
    pass

def _ensure_cfbd_key():
    """Ensure CFBD_API_KEY (or compatible token) is present; if missing, re-read .env files.
    This function previously became corrupted during a large patch; restored to a minimal safe helper.
    """
    try:
        if os.environ.get('CFBD_API_KEY') or os.environ.get('CFBD_TOKEN') or os.environ.get('CFBD'):
            return
        # Retry loading env files silently
        _load_env_file(os.path.join(BASE_DIR, '.env'))
        _load_env_file(os.path.join(BASE_DIR, '.env.local'))
    except Exception:
        pass

# Core data directory & prediction file paths (added after corruption fix)
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
pred_path_enh = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced.csv')
pred_path_scores = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv')
PRED_SOURCE = 'unknown'

# UI/env toggles & locks
HIDE_REFRESH = os.environ.get('HIDE_REFRESH', '0').lower() in ('1','true','yes')
_REFRESH_LOCK = threading.Lock()

# Simple in-memory cache for game cards API (invalidated on prediction reload)
GAME_CARDS_CACHE = {}

def _calibrate_win_prob(p):
    """Fallback calibration (identity clamp) if real calibration artifacts absent due to earlier truncation."""
    try:
        x = float(p)
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        return x
    except Exception:
        return None

# Optional: Conference-level sigma overrides for margin
_CONF_SIGMA = None
try:
    _dfc = pd.read_csv(os.path.join(DATA_DIR, "conference_sigma_overrides.csv"))
    if {'conference','sigma_margin'}.issubset(set(_dfc.columns)):
        _CONF_SIGMA = { str(r['conference']).strip(): float(r['sigma_margin']) for _, r in _dfc.iterrows() if pd.notnull(r['sigma_margin']) }
except Exception:
    _CONF_SIGMA = None

def _is_constant_predictions(df: pd.DataFrame) -> bool:
    try:
        sub = df[df['season'] == 2025]
        if sub.empty:
            sub = df
        # Check uniqueness and variance across predictions
        uh = sub['predicted_home_points'].nunique(dropna=True) if 'predicted_home_points' in sub.columns else None
        ua = sub['predicted_away_points'].nunique(dropna=True) if 'predicted_away_points' in sub.columns else None
        if uh is not None and ua is not None:
            if uh <= 2 and ua <= 2:
                return True
        # Additional guard: very low std on totals
        if 'predicted_total_points' in sub.columns:
            try:
                if float(pd.to_numeric(sub['predicted_total_points'], errors='coerce').std(skipna=True)) < 0.5:
                    return True
            except Exception:
                pass
    except Exception:
        return False
    return False

def _apply_week0_label(df: pd.DataFrame) -> pd.DataFrame:
    """For 2025, relabel early kickoff games before the main Thursday slate as Week 0.
    This helps align with CFBD (and user expectations) so completed Week 0 games are easy to filter.
    """
    try:
        if 'season' not in df.columns or 'week' not in df.columns or 'start_date' not in df.columns:
            return df
        mask_2025 = df['season'] == 2025
        if not mask_2025.any():
            return df
        # Threshold: 2025-08-28 00:00:00+00:00 (first big Thursday)
        threshold = pd.Timestamp('2025-08-28T00:00:00+00:00')
        def _pdt(x):
            try:
                s = str(x)
                if not s:
                    return pd.NaT
                return pd.to_datetime(s)
            except Exception:
                return pd.NaT
        sdt = df.loc[mask_2025, 'start_date'].apply(_pdt)
        to_week0 = sdt.notna() & (sdt < threshold)
        if to_week0.any():
            idx = sdt[to_week0].index
            df.loc[idx, 'week'] = 0
        # Try to keep week as int where possible
        try:
            df['week'] = pd.to_numeric(df['week'], errors='coerce')
            if df['week'].notna().all():
                df['week'] = df['week'].astype(int)
        except Exception:
            pass
        return df
    except Exception:
        return df

def _load_predictions_df() -> pd.DataFrame:
    """Load predictions preferring enhanced (to preserve weather), then merge in actuals from with_scores if present."""
    global PRED_SOURCE
    df_enh = None
    df_scores = None
    actuals_df = None
    # Try reading both files if available
    try:
        if pred_path_enh and os.path.exists(pred_path_enh):
            df_enh = pd.read_csv(pred_path_enh)
            # Align week labels (Week 0 vs 1) before any merges
            try:
                df_enh = _apply_week0_label(df_enh)
            except Exception:
                pass
            # If base enhanced file lacks model / odds columns, attempt to auto-upgrade
            try:
                have_model = any(c.startswith('model_') for c in df_enh.columns)
                if not have_model:
                    pattern = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced_*.csv')
                    cand_files = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
                    for cf in cand_files:
                        # Skip with scores (handled separately) and the base file itself
                        base_name = os.path.basename(cf)
                        if base_name.endswith('with_scores.csv') or base_name == os.path.basename(pred_path_enh):
                            continue
                        try:
                            tmp_df = pd.read_csv(cf)
                        except Exception:
                            continue
                        if any(c.startswith('model_') for c in tmp_df.columns):
                            df_enh = tmp_df
                            print(f"[load] Upgraded predictions source to latest model overlay file: {base_name}")
                            break
            except Exception as _upgrade_e:
                print(f"[load] upgrade scan failed: {_upgrade_e}")
    except Exception as e:
        print(f"[app] Failed to read enhanced: {e}")
    try:
        if pred_path_scores and os.path.exists(pred_path_scores):
            df_scores = pd.read_csv(pred_path_scores)
            if all(col in df_scores.columns for col in ['season','week','home_team','away_team','actual_home_points','actual_away_points']):
                # Prefer including start_date when available so we can week-align on actuals as well
                cols = ['season','week','home_team','away_team','actual_home_points','actual_away_points']
                if 'start_date' in df_scores.columns:
                    cols.append('start_date')
                if 'start_date_api' in df_scores.columns:
                    cols.append('start_date_api')
                actuals_df = df_scores[cols].copy()
                # Create a usable start_date column if only API format exists
                try:
                    if 'start_date' not in actuals_df.columns and 'start_date_api' in actuals_df.columns:
                        tmp = pd.to_datetime(actuals_df['start_date_api'], errors='coerce')
                        actuals_df['start_date'] = tmp.dt.tz_localize(None).astype(str)
                except Exception:
                    pass
                # Align week 0/1 labels in actuals for 2025 before merge
                try:
                    actuals_df = _apply_week0_label(actuals_df)
                except Exception:
                    pass
    except Exception as e:
        print(f"[app] Failed to read with_scores: {e}")

    # Preferred path: have enhanced; merge in actuals if available
    if df_enh is not None and isinstance(df_enh, pd.DataFrame) and not df_enh.empty:
        df = df_enh.copy()
        # Early duplicate removal
        try:
            if {'season','week','home_team','away_team','start_date'}.issubset(df.columns):
                before_ct = len(df)
                df = df.sort_values(by=['season','week','start_date','home_team','away_team']).drop_duplicates(subset=['season','week','home_team','away_team','start_date'], keep='first')
                if len(df) != before_ct:
                    print(f"[load] Dropped {before_ct-len(df)} duplicate base rows")
            elif {'season','week','home_team','away_team'}.issubset(df.columns):
                before_ct = len(df)
                df = df.sort_values(by=['season','week','home_team','away_team']).drop_duplicates(subset=['season','week','home_team','away_team'], keep='first')
                if len(df) != before_ct:
                    print(f"[load] Dropped {before_ct-len(df)} duplicate base rows (no start_date)")
        except Exception:
            pass
        # Ensure required columns exist
        for col in ['actual_home_points','actual_away_points','start_date_api']:
            if col not in df.columns:
                df[col] = pd.NA
        if actuals_df is not None and not actuals_df.empty:
            try:
                df = df.merge(actuals_df, on=['season','week','home_team','away_team'], how='left', suffixes=('', '_from_scores'))
                for col in ['actual_home_points','actual_away_points','start_date_api']:
                    alt = f"{col}_from_scores"
                    if alt in df.columns:
                        df[col] = df[col].where(df[col].notna(), df[alt])
                df = df.drop(columns=[c for c in df.columns if c.endswith('_from_scores')])
                for col in ['actual_home_points','actual_away_points']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Missing actuals fallback merge without week
                try:
                    missing_mask = (df.get('season',0)==2025) & (df['actual_home_points'].isna() | df['actual_away_points'].isna())
                except Exception:
                    missing_mask = pd.Series([False]*len(df))
                if missing_mask.any():
                    try:
                        no_wk_cols = ['season','home_team','away_team','actual_home_points','actual_away_points']
                        actuals_nowk = actuals_df[[c for c in no_wk_cols if c in actuals_df.columns]].copy()
                        actuals_nowk = actuals_nowk.sort_values(by=[c for c in ['actual_home_points','actual_away_points'] if c in actuals_nowk.columns], ascending=False)
                        actuals_nowk = actuals_nowk.drop_duplicates(subset=['season','home_team','away_team'], keep='first')
                        left = df[missing_mask].merge(actuals_nowk, on=['season','home_team','away_team'], how='left', suffixes=('', '_nw'))
                        for col in ['actual_home_points','actual_away_points']:
                            alt = f"{col}_nw"
                            if alt in left.columns:
                                left[col] = left[col].where(left[col].notna(), left[alt])
                        df.loc[missing_mask,['actual_home_points','actual_away_points']] = left[['actual_home_points','actual_away_points']].values
                    except Exception:
                        pass
                try:
                    df = _apply_week0_label(df)
                except Exception:
                    pass
                PRED_SOURCE = 'enhanced+scores'
            except Exception as _merge_e:
                print(f"[app] Merge actuals into enhanced failed: {_merge_e}")
                PRED_SOURCE = 'enhanced'
        else:
            try:
                df = _apply_week0_label(df)
            except Exception:
                pass
            PRED_SOURCE = 'enhanced'
    elif df_scores is not None and isinstance(df_scores, pd.DataFrame) and not df_scores.empty:
        df = df_scores.copy()
        for col in ['actual_home_points','actual_away_points']:
            if col not in df.columns:
                df[col] = pd.NA
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        try:
            df = _apply_week0_label(df)
        except Exception:
            pass
        PRED_SOURCE = 'scores_only'
    else:
        df = pd.DataFrame(columns=['season','week','home_team','away_team'])
        PRED_SOURCE = 'none'

    # Final duplicate cull
    try:
        dup_keys = [c for c in ['season','week','start_date','home_team','away_team'] if c in df.columns]
        if len(dup_keys) >= 4:
            before = len(df)
            df = df.sort_values(by=dup_keys).drop_duplicates(subset=dup_keys, keep='first')
            if len(df) != before:
                print(f"[load] Post-merge duplicate removal: {before-len(df)} rows dropped")
        elif {'season','week','home_team','away_team'}.issubset(df.columns):
            before = len(df)
            df = df.sort_values(by=['season','week','home_team','away_team']).drop_duplicates(subset=['season','week','home_team','away_team'], keep='first')
            if len(df) != before:
                print(f"[load] Post-merge duplicate removal (no start_date): {before-len(df)} rows dropped")
    except Exception:
        pass
    return df

pred_df = _load_predictions_df()

# Coerce core columns to numeric where applicable for reliable filtering/sorting
try:
    if 'week' in pred_df.columns:
        pred_df['week'] = pd.to_numeric(pred_df['week'], errors='coerce')
    if 'season' in pred_df.columns:
        pred_df['season'] = pd.to_numeric(pred_df['season'], errors='coerce')
except Exception:
    pass

try:
    team_conf_df = pd.read_csv(os.path.join(DATA_DIR, "team_conferences.csv"))
    team_conf_df['school_norm'] = team_conf_df['school'].str.strip().str.lower().str.replace('&', 'and').str.replace('  ', ' ')
except Exception:
    # Safe fallback: empty mapping so conferences default to Unknown
    team_conf_df = pd.DataFrame(columns=['school', 'conference', 'school_norm'])

# Add conference info to predictions
def norm(name):
    return str(name).strip().lower().replace('&', 'and').replace('  ', ' ')
conf_map = dict(zip(team_conf_df['school_norm'], team_conf_df['conference']))
pred_df['home_conference'] = pred_df['home_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
pred_df['away_conference'] = pred_df['away_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))

# Load win margin confidence intervals and build a fast lookup index
WIN_MARGIN_CONF_INDEX = {}
try:
    win_margin_conf_df = pd.read_csv(os.path.join(DATA_DIR, "win_margin_predictions_with_confidence.csv"))
    win_margin_conf_df.columns = win_margin_conf_df.columns.str.strip()
    required = {'season','week','home_team','away_team','conf_interval_lower','conf_interval_upper','conf_std'}
    if required.issubset(set(win_margin_conf_df.columns)):
        # Normalize keys once
        def _norm_simple(x):
            return str(x).strip().lower().replace('&','and').replace('  ',' ')
        try:
            win_margin_conf_df['season'] = pd.to_numeric(win_margin_conf_df['season'], errors='coerce').astype('Int64')
            win_margin_conf_df['week'] = pd.to_numeric(win_margin_conf_df['week'], errors='coerce').astype('Int64')
        except Exception:
            pass
        for _, r in win_margin_conf_df.iterrows():
            try:
                key = (int(r['season']), int(r['week']), _norm_simple(r['home_team']), _norm_simple(r['away_team']))
            except Exception:
                continue
            WIN_MARGIN_CONF_INDEX[key] = {
                'lower': r.get('conf_interval_lower', None),
                'upper': r.get('conf_interval_upper', None),
                'std': r.get('conf_std', None)
            }
    else:
        win_margin_conf_df = None
except Exception:
    win_margin_conf_df = None

# Load team assets
try:
    assets_df = pd.read_csv(os.path.join(DATA_DIR, "team_assets.csv"))
except Exception:
    # Safe fallback: empty assets to avoid startup crash
    assets_df = pd.DataFrame(columns=['school', 'logo', 'color', 'alt_color'])
def get_team_asset(team_name):
    row = assets_df[assets_df['school'] == team_name]
    if not row.empty:
        def clean(v):
            try:
                import pandas as _pd
                return '' if _pd.isna(v) else v
            except Exception:
                return v if v is not None else ''
        return {
            'logo': clean(row.iloc[0].get('logo', '')),
            'color': clean(row.iloc[0].get('color', '')),
            'alt_color': clean(row.iloc[0].get('alt_color', ''))
        }
    return {'logo': '', 'color': '', 'alt_color': ''}

# Load betting lines
lines_df = pd.read_csv(os.path.join(DATA_DIR, "college_football_betting_lines_last_15_years.csv")) if os.path.exists(os.path.join(DATA_DIR, "college_football_betting_lines_last_15_years.csv")) else pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
lines_index = {}
lines_index_norm = {}
def _norm_team_for_odds(name: str) -> str:
    try:
        s = str(name or '')
        s = s.strip().lower()
        s = s.replace('&', 'and')
        # unify apostrophes and remove punctuation except spaces
        s = s.replace("ʻ", "'").replace("’", "'")
        s = re.sub(r"[^a-z0-9 '\-]", " ", s)
        s = s.replace("hawai'i", "hawaii")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    except Exception:
        return str(name)

# Common alias map to improve matching odds provider names to our schedule team names
# Keys and values are both in normalized form (post _norm_team_for_odds)
TEAM_ALIASES = {
    'miami oh': 'miami ohio',
    'miami ohio': 'miami ohio',
    'ole miss': 'mississippi',
    'miss st': 'mississippi state',
    'utsa': 'texas san antonio',
    'ut san antonio': 'texas san antonio',
    'ucf': 'central florida',
    'usf': 'south florida',
    'app state': 'appalachian state',
    'appalachian st': 'appalachian state',
    'southern miss': 'southern mississippi',
    "hawai'i": 'hawaii',
    'hawaii': 'hawaii',
    'la lafayette': 'louisiana',
    'louisiana lafayette': 'louisiana',
    'la monroe': 'louisiana monroe',
    'umass': 'massachusetts',
    'uconn': 'connecticut',
    'utsa roadrunners': 'texas san antonio',
    'byu cougars': 'byu',
    # Extended mascot / full-name forms (keep normalized)
    'ole miss rebels': 'mississippi',
    'arkansas razorbacks': 'arkansas',
    'umass minutemen': 'massachusetts',
    'notre dame fighting irish': 'notre dame',
    'texas a m aggies': 'texas am', 'texas a and m aggies': 'texas am', 'texas am aggies': 'texas am',
    'miami hurricanes': 'miami',
    'florida gators': 'florida', 'florida state seminoles': 'florida state',
    'georgia bulldogs': 'georgia', 'alabama crimson tide': 'alabama',
    'penn state nittany lions': 'penn state', 'oregon ducks': 'oregon',
    'nebraska cornhuskers': 'nebraska', 'michigan wolverines': 'michigan',
    'lsu tigers': 'lsu', 'uconn huskies': 'connecticut', 'delaware blue hens': 'delaware',
}

def _canon_team(name: str) -> str:
    n = _norm_team_for_odds(name)
    return TEAM_ALIASES.get(n, n)

def _build_lines_index(df):
    idx = {}
    idx_norm = {}
    try:
        for _, row in df.iterrows():
            try:
                y = int(row['year'])
                w = int(row['week'])
                ht = row['homeTeam']
                at = row['awayTeam']
                key = (y, w, ht, at)
                odds_str = row.get('lines','')
                odds = []
                # Robustly parse odds stored as JSON or Python literal strings
                if isinstance(odds_str, str):
                    parsed = None
                    try:
                        parsed = json.loads(odds_str)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(odds_str)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, list):
                        odds = parsed
                    else:
                        odds = []
                else:
                    odds = odds_str if isinstance(odds_str, list) else []
                idx[key] = odds
                # normalized fallback keys (raw-normalized and canonical-normalized)
                n_ht = _norm_team_for_odds(ht)
                n_at = _norm_team_for_odds(at)
                c_ht = TEAM_ALIASES.get(n_ht, n_ht)
                c_at = TEAM_ALIASES.get(n_at, n_at)
                idx_norm[(y, w, n_ht, n_at)] = odds
                idx_norm[(y, w, c_ht, c_at)] = odds
            except Exception:
                continue
    except Exception:
        idx = {}
        idx_norm = {}
    return idx, idx_norm
lines_index, lines_index_norm = _build_lines_index(lines_df)

def _overlay_lines_2025_if_present():
    """Overlay 2025 lines CSV into lines_df and rebuild indexes. Safe if missing."""
    global lines_df, lines_index, lines_index_norm
    try:
        lines_2025_path = os.path.join(DATA_DIR, 'college_football_betting_lines_2025.csv')
        # If a copy exists under NCAFCompare/src/data or src/data, consider merging it into DATA_DIR
        nested = [
            os.path.join(BASE_DIR, 'NCAFCompare', 'src', 'data', 'college_football_betting_lines_2025.csv'),
            os.path.join(BASE_DIR, 'src', 'data', 'college_football_betting_lines_2025.csv'),
        ]
        def _read_df(path: str) -> pd.DataFrame:
            try:
                if os.path.exists(path):
                    return pd.read_csv(path)
            except Exception:
                pass
            return pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])

        # Merge strategy:
        # - If DATA_DIR file is missing: copy the largest available nested file.
        # - If DATA_DIR file exists: union keys (year,week,homeTeam,awayTeam), prefer nested rows for
        #   the weeks present in nested, but never drop other weeks. Back up DATA_DIR copy first.
        if not os.path.exists(lines_2025_path):
            # Choose the largest existing nested file
            best = None
            best_size = -1
            for n in nested:
                try:
                    if os.path.exists(n):
                        sz = os.path.getsize(n)
                        if sz > best_size:
                            best = n; best_size = sz
                except Exception:
                    continue
            if best:
                import shutil
                os.makedirs(os.path.dirname(lines_2025_path), exist_ok=True)
                shutil.copy2(best, lines_2025_path)
        else:
            # Attempt a safe merge with any newer nested files
            main_df = _read_df(lines_2025_path)
            merged = main_df.copy()
            for n in nested:
                try:
                    if not os.path.exists(n):
                        continue
                    # Only consider merging if nested is newer OR we have zero rows for 2025 in memory
                    do_merge = False
                    try:
                        do_merge = os.path.getmtime(n) > os.path.getmtime(lines_2025_path)
                    except Exception:
                        do_merge = True
                    if not do_merge:
                        continue
                    other = _read_df(n)
                    if other is None or other.empty:
                        continue
                    # Backup existing DATA_DIR file before merging
                    try:
                        bdir = os.path.join(os.path.dirname(lines_2025_path), 'backups', 'lines_2025')
                        os.makedirs(bdir, exist_ok=True)
                        ts = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
                        bpath = os.path.join(bdir, f"college_football_betting_lines_2025_{ts}.csv")
                        main_df.to_csv(bpath, index=False)
                    except Exception:
                        pass
                    # Merge by key, prefer 'other' for overlapping rows
                    try:
                        main_keys = set((int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam'])) for _, r in merged.iterrows())
                    except Exception:
                        main_keys = set()
                    rows = []
                    for _, r in other.iterrows():
                        try:
                            k = (int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam']))
                        except Exception:
                            continue
                        # Remove any existing row with this key
                        if k in main_keys:
                            merged = merged[~((merged.get('year',0)==k[0]) & (merged.get('week',-1)==k[1]) & (merged.get('homeTeam','')==k[2]) & (merged.get('awayTeam','')==k[3]))]
                        rows.append(r.to_dict())
                    if rows:
                        merged = pd.concat([merged, pd.DataFrame(rows)], ignore_index=True)
                except Exception:
                    # Skip problematic nested file and continue
                    continue
            # Write merged back
            try:
                if not merged.equals(main_df):
                    merged.to_csv(lines_2025_path, index=False)
            except Exception:
                pass
        if not os.path.exists(lines_2025_path):
            return
        new_lines = _read_df(lines_2025_path)
        # If Week 1 odds exist in the historical file but are missing from new_lines, bring them forward
        try:
            hist_path = os.path.join(DATA_DIR, 'college_football_betting_lines_last_15_years.csv')
            if os.path.exists(hist_path):
                hist = pd.read_csv(hist_path)
                if not hist.empty and 'year' in hist.columns and 'week' in hist.columns:
                    w1_hist = hist[(hist.get('year',0)==2025) & (hist.get('week',-1)==1)]
                    if not w1_hist.empty:
                        # Build keys present in new_lines (avoid dupes)
                        try:
                            keys_new = set((int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam'])) for _, r in new_lines.iterrows())
                        except Exception:
                            keys_new = set()
                        add_rows = []
                        for _, r in w1_hist.iterrows():
                            try:
                                k=(int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam']))
                            except Exception:
                                continue
                            if k not in keys_new:
                                add_rows.append(r.to_dict())
                        if add_rows:
                            new_lines = pd.concat([new_lines, pd.DataFrame(add_rows)], ignore_index=True)
        except Exception:
            pass
        # Keep non-2025 from current df, replace 2025 rows with new file (preserving any non-overlapping entries)
        if not isinstance(lines_df, pd.DataFrame) or lines_df.empty:
            base_non_2025 = pd.DataFrame(columns=new_lines.columns)
        else:
            base_non_2025 = lines_df[lines_df.get('year', 0) != 2025] if 'year' in lines_df.columns else pd.DataFrame()
        # Build preserved old 2025 rows that are not in new set
        if isinstance(lines_df, pd.DataFrame) and 'year' in lines_df.columns:
            old_2025 = lines_df[lines_df['year'] == 2025]
        else:
            old_2025 = pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
        try:
            new_keys = set((int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam'])) for _, r in new_lines.iterrows())
        except Exception:
            new_keys = set()
        to_keep = []
        for _, r in old_2025.iterrows():
            try:
                k = (int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam']))
            except Exception:
                continue
            if k not in new_keys:
                to_keep.append(r)
        preserved = pd.DataFrame(to_keep) if to_keep else pd.DataFrame(columns=new_lines.columns)
        lines_df_local = pd.concat([base_non_2025, preserved, new_lines], ignore_index=True)
        # Rebuild indexes from merged df
        lines_idx, lines_idx_norm = _build_lines_index(lines_df_local)
        # Commit globals only after successful build
        lines_df = lines_df_local
        lines_index, lines_index_norm = lines_idx, lines_idx_norm
    except Exception:
        # Do not crash app on overlay failure
        pass

# Attempt to load 2025 lines immediately so weeks 0-2 have odds without manual refresh
_overlay_lines_2025_if_present()
def get_betting_lines(year, week, home_team, away_team):
    def _try(y, w, ht, at):
        k = (int(y), int(w), ht, at)
        o = lines_index.get(k)
        if o is not None:
            return o
        # Try multiple normalized variants, including canonical and flipped orientation
        n_ht = _norm_team_for_odds(ht); n_at = _norm_team_for_odds(at)
        c_ht = _canon_team(ht); c_at = _canon_team(at)
        for a,b in ((n_ht, n_at), (c_ht, c_at)):
            v = lines_index_norm.get((int(y), int(w), a, b))
            if v: return v
        # try flipped
        for a,b in ((n_at, n_ht), (c_at, c_ht)):
            v = lines_index_norm.get((int(y), int(w), a, b))
            if v: return v
        return []

    y = int(year)
    w = int(week)
    odds = _try(y, w, home_team, away_team)
    if odds:
        return odds
    # Week 0/1 labeling mismatch fallback for 2025
    if y == 2025 and w in (0, 1):
        alt_w = 1 if w == 0 else 0
        odds_alt = _try(y, alt_w, home_team, away_team)
        if odds_alt:
            return odds_alt
    return []

def _build_game_card(game_row: pd.Series) -> dict:
    def r2(val):
        try:
            return f"{float(val):.2f}"
        except Exception:
            return val
    home_asset = get_team_asset(game_row['home_team'])
    away_asset = get_team_asset(game_row['away_team'])
    # Matchup classification (FBS vs FBS or FBS vs Non-FBS)
    try:
        _fbs_conf_set = {
            'acc','sec','big ten','big 12','pac 12','american','mountain west','sun belt','mac','conference usa','independent','independents','fbs independents','independent (fbs)'
        }
        _fbs_independents = {'notre dame','army','navy','umass','uconn','new mexico state'}
        def _is_fbs(team, conf):
            try:
                t = str(team or '').strip().lower()
                c = str(conf or '').strip().lower()
                return c in _fbs_conf_set or t in _fbs_independents
            except Exception:
                return False
        home_is_fbs = _is_fbs(game_row.get('home_team'), game_row.get('home_conference'))
        away_is_fbs = _is_fbs(game_row.get('away_team'), game_row.get('away_conference'))
        if home_is_fbs and away_is_fbs:
            matchup_level = 'FBS_vs_FBS'
        elif home_is_fbs or away_is_fbs:
            matchup_level = 'FBS_vs_NonFBS'
        else:
            matchup_level = 'NonFBS_vs_NonFBS'
    except Exception:
        home_is_fbs = away_is_fbs = False
        matchup_level = 'Unknown'
    betting_lines = get_betting_lines(
        year=int(game_row['season']),
        week=int(game_row['week']),
        home_team=game_row['home_team'],
        away_team=game_row['away_team']
    )
    # Synthetic fallback odds have been disabled by default (user request to exclude).
    # To re-enable, set environment variable ALLOW_SYNTHETIC_ODDS=1.
    if not betting_lines and os.environ.get('ALLOW_SYNTHETIC_ODDS','0') == '1':
        try:
            mh = _safe_float(game_row.get('model_home_points'))
            ma = _safe_float(game_row.get('model_away_points'))
            mtot = _safe_float(game_row.get('model_total_points'))
            mmargin = _safe_float(game_row.get('model_margin'))
            if mh is not None and ma is not None:
                if mtot is None:
                    mtot = mh + ma
                if mmargin is None:
                    mmargin = mh - ma
                spread = round(mmargin, 1)
                ou = round(mtot, 1) if mtot is not None else None
                p_home = _safe_float(game_row.get('model_home_win_prob'))
                def _fair_ml(p):
                    try:
                        if p is None or p <= 0 or p >= 1:
                            return None, None
                        if p >= 0.5:
                            home_ml = int(round(-100 * p / (1 - p)))
                            away_ml = int(round(100 * (1 - p) / p))
                        else:
                            away_ml = int(round(-100 * (1 - p) / p))
                            home_ml = int(round(100 * p / (1 - p)))
                        return home_ml, away_ml
                    except Exception:
                        return None, None
                home_ml, away_ml = _fair_ml(p_home)
                betting_lines = [{
                    'provider': 'ModelImplied',
                    'spread': spread,
                    'formattedSpread': f"{game_row['home_team']} {spread:+.1f}",
                    'overUnder': ou,
                    'homeMoneyline': home_ml,
                    'awayMoneyline': away_ml,
                    'synthetic': True
                }]
        except Exception:
            pass
    # Time handling and sort key (robust to NaN/NaT/None)
    def _is_bad_date_val(v):
        try:
            if v is None:
                return True
            if isinstance(v, float) and math.isnan(v):
                return True
            if isinstance(v, str):
                s = v.strip().lower()
                return s in ('', 'nan', 'nat', 'none', 'null')
            return False
        except Exception:
            return True
    raw_api = game_row.get('start_date_api', None)
    raw_sd = game_row.get('start_date', None)
    val_api = None if _is_bad_date_val(raw_api) else str(raw_api).strip()
    val_sd = None if _is_bad_date_val(raw_sd) else str(raw_sd).strip()
    display_time_fallback = val_sd or val_api or ''
    start_iso = ''
    sort_ts = None
    candidates = [c for c in [val_api, val_sd] if c]
    for c in candidates:
        try:
            s = c
            # Normalize space to 'T'
            if 'T' not in s and ' ' in s:
                s = s.replace(' ', 'T')
            # Parse with fromisoformat; if fails, try pandas
            dt_obj = None
            try:
                if s.endswith('Z'):
                    dt_obj = datetime.fromisoformat(s.replace('Z', '+00:00'))
                else:
                    dt_obj = datetime.fromisoformat(s)
            except Exception:
                try:
                    dt_obj = pd.to_datetime(c, errors='coerce').to_pydatetime() if c else None
                except Exception:
                    dt_obj = None
            if not dt_obj:
                continue
            # Ensure timezone-aware UTC
            if getattr(dt_obj, 'tzinfo', None) is None:
                dt_utc = dt_obj.replace(tzinfo=pytz.UTC)
            else:
                dt_utc = dt_obj.astimezone(pytz.UTC)
            sort_ts = dt_utc.timestamp()
            start_iso = dt_utc.isoformat().replace('+00:00', 'Z')
            try:
                display_time_fallback = dt_utc.strftime('%a, %b %d, %Y, %I:%M %p UTC')
            except Exception:
                display_time_fallback = start_iso
            break
        except Exception:
            continue
    # Confidence bounds
    conf_lower = conf_upper = conf_std = None
    try:
        week_val = int(game_row.get('week', 0))
        season_val = int(game_row.get('season', 0))
        key = (season_val, week_val, norm(game_row.get('home_team','')), norm(game_row.get('away_team','')))
        if key in WIN_MARGIN_CONF_INDEX:
            ent = WIN_MARGIN_CONF_INDEX[key]
            conf_lower = r2(ent.get('lower', None))
            conf_upper = r2(ent.get('upper', None))
            conf_std = r2(ent.get('std', None))
    except Exception:
        pass
    # Preds/actuals
    actual_home = _safe_float(game_row.get('actual_home_points', None))
    actual_away = _safe_float(game_row.get('actual_away_points', None))
    def _is_valid_num(x):
        try:
            return x is not None and not (isinstance(x, float) and math.isnan(x))
        except Exception:
            return x is not None
    # Base (legacy) predictions
    predicted_home = _safe_float(game_row.get('predicted_home_points', None))
    predicted_away = _safe_float(game_row.get('predicted_away_points', None))
    # Prefer model predictions if present
    model_home = _safe_float(game_row.get('model_home_points', None))
    model_away = _safe_float(game_row.get('model_away_points', None))
    if model_home is not None:
        predicted_home = model_home
    if model_away is not None:
        predicted_away = model_away
    predicted_winner = None
    actual_winner = None
    correct_prediction = None
    if predicted_home is not None and predicted_away is not None:
        if predicted_home > predicted_away:
            predicted_winner = game_row['home_team']
        elif predicted_home < predicted_away:
            predicted_winner = game_row['away_team']
    # Win probability: prefer calibrated model probability if available
    p_home_win = _safe_float(game_row.get('model_home_win_prob'))
    if p_home_win is None:
        try:
            if predicted_home is not None and predicted_away is not None:
                pred_margin_tmp = _safe_float(game_row.get('model_margin'))
                if pred_margin_tmp is None:
                    pred_margin_tmp = _safe_float(game_row.get('predicted_win_margin'), predicted_home - predicted_away)
                sigma_tmp = _get_conf_std_for_game(game_row)
                p_home_win = _phi(pred_margin_tmp / sigma_tmp)
        except Exception:
            p_home_win = None
    if _is_valid_num(actual_home) and _is_valid_num(actual_away):
        if actual_home > actual_away:
            actual_winner = game_row['home_team']
        elif actual_home < actual_away:
            actual_winner = game_row['away_team']
    if predicted_winner and actual_winner:
        correct_prediction = (predicted_winner == actual_winner)
    actual_total_points = None
    predicted_total_points = None
    total_points_diff = None
    if _is_valid_num(actual_home) and _is_valid_num(actual_away):
        try:
            s = actual_home + actual_away
            if isinstance(s, float) and math.isnan(s):
                actual_total_points = None
            else:
                actual_total_points = s
        except Exception:
            actual_total_points = None
    if predicted_home is not None and predicted_away is not None:
        predicted_total_points = predicted_home + predicted_away
    if actual_total_points is not None and predicted_total_points is not None:
        total_points_diff = actual_total_points - predicted_total_points
    # Weather
    wx_temp = _safe_float(game_row.get('wx_temp_f', None))
    wx_wind = _safe_float(game_row.get('wx_wind_mph', None))
    wx_adj = _safe_float(game_row.get('wx_adjust_total', None), None)
    pred_total_adj_num = _safe_float(game_row.get('predicted_total_points', None), None)
    pred_total_pre_num = None
    if pred_total_adj_num is not None:
        try:
            pred_total_pre_num = pred_total_adj_num - (wx_adj if wx_adj is not None else 0.0)
        except Exception:
            pred_total_pre_num = None
    else:
        if predicted_home is not None and predicted_away is not None:
            pred_total_pre_num = predicted_home + predicted_away
            pred_total_adj_num = pred_total_pre_num
    # O/U
    ou_line = None
    ou_model_lean = None
    ou_edge = None
    ou_actual_result = None
    ou_correct = None
    try:
        ou_values = []
        if betting_lines:
            for bl in betting_lines:
                ou = (
                    bl.get('overUnder') or bl.get('total') or bl.get('over_under') or bl.get('OU') or bl.get('o_u')
                )
                try:
                    if ou is not None and ou != '':
                        ou_values.append(float(ou))
                except Exception:
                    continue
        if ou_values:
            ou_line = sum(ou_values) / len(ou_values)
    except Exception:
        ou_line = None
    if ou_line is not None and predicted_total_points is not None:
        if predicted_total_points > ou_line:
            ou_model_lean = 'Over'
        elif predicted_total_points < ou_line:
            ou_model_lean = 'Under'
        else:
            ou_model_lean = 'Push'
        ou_edge = predicted_total_points - ou_line
    if ou_line is not None and actual_total_points is not None:
        if actual_total_points > ou_line:
            ou_actual_result = 'Over'
        elif actual_total_points < ou_line:
            ou_actual_result = 'Under'
        else:
            ou_actual_result = 'Push'
        if ou_model_lean in ('Over', 'Under') and ou_actual_result in ('Over', 'Under'):
            ou_correct = (ou_model_lean == ou_actual_result)
        else:
            ou_correct = None
    # ATS
    ats_home_line = None
    ats_model_lean = None
    ats_edge = None
    ats_actual_result = None
    ats_correct = None
    try:
        spread_vals = []
        if betting_lines:
            for bl in betting_lines:
                s_fmt = bl.get('formattedSpread')
                s_raw = bl.get('spread')
                val = None
                if isinstance(s_fmt, str) and s_fmt:
                    try:
                        if 'Home' in s_fmt or 'Away' in s_fmt:
                            num = float(s_fmt.replace('Home','').replace('Away','').strip())
                            if 'Home' in s_fmt:
                                val = num
                            else:
                                val = -num
                        else:
                            m = re.match(r"^(.*)\s+([+-]?[0-9]*\.?[0-9]+)$", s_fmt.strip())
                            if m:
                                team_label = m.group(1).strip()
                                num = float(m.group(2))
                                home_name = str(game_row.get('home_team','')).strip().lower()
                                away_name = str(game_row.get('away_team','')).strip().lower()
                                lbl = team_label.strip().lower()
                                is_home_labeled = (home_name in lbl) and not (away_name in lbl)
                                is_away_labeled = (away_name in lbl) and not (home_name in lbl)
                                if is_home_labeled:
                                    val = num
                                elif is_away_labeled:
                                    val = -num
                                else:
                                    val = None
                    except Exception:
                        val = None
                    # Fallback: if we couldn't infer team from formatted label (e.g., abbreviations),
                    # use numeric spread directly when available.
                    if val is None:
                        try:
                            if s_raw is not None and s_raw != '':
                                val = float(s_raw)
                        except Exception:
                            val = None
                else:
                    try:
                        if s_raw is not None and s_raw != '':
                            val = float(s_raw)
                    except Exception:
                        val = None
                if val is not None:
                    spread_vals.append(val)
        if spread_vals:
            ats_home_line = sum(spread_vals) / len(spread_vals)
    except Exception:
        ats_home_line = None
    if ats_home_line is not None and predicted_home is not None and predicted_away is not None:
        pred_margin = predicted_home - predicted_away
        comp = pred_margin + ats_home_line
        if comp > 0:
            ats_model_lean = 'Home'
        elif comp < 0:
            ats_model_lean = 'Away'
        else:
            ats_model_lean = 'Push'
        ats_edge = comp
    if ats_home_line is not None and _is_valid_num(actual_home) and _is_valid_num(actual_away):
        actual_margin = actual_home - actual_away
        comp_a = actual_margin + ats_home_line
        if comp_a > 0:
            ats_actual_result = 'Home'
        elif comp_a < 0:
            ats_actual_result = 'Away'
        else:
            ats_actual_result = 'Push'
        if ats_model_lean in ('Home','Away') and ats_actual_result in ('Home','Away'):
            ats_correct = (ats_model_lean == ats_actual_result)
        else:
            ats_correct = None
    def _format_ats_line(v):
        if v is None:
            return None
        return f"Home {float(v):+0.1f}".replace('+', '+').replace('-0.0', '0.0')
    return {
        'home_team': game_row['home_team'],
        'away_team': game_row['away_team'],
    'home_conference': game_row.get('home_conference', ''),
    'away_conference': game_row.get('away_conference', ''),
    'home_is_fbs': bool(home_is_fbs),
    'away_is_fbs': bool(away_is_fbs),
    'matchup_level': matchup_level,
        'venue': game_row.get('venue', ''),
        'game_time': display_time_fallback,
        'start_iso': start_iso,
        'sort_ts': sort_ts,
        'predicted_total_points': r2(predicted_total_points),
        'pred_total_adj': r2(pred_total_adj_num) if pred_total_adj_num is not None else None,
        'pred_total_pre': r2(pred_total_pre_num) if pred_total_pre_num is not None else None,
        'actual_total_points': r2(actual_total_points),
        'total_points_diff': r2(total_points_diff) if total_points_diff is not None else None,
        'predicted_home_points': r2(predicted_home),
        'predicted_away_points': r2(predicted_away),
        'actual_home_points': r2(actual_home) if _is_valid_num(actual_home) else None,
        'actual_away_points': r2(actual_away) if _is_valid_num(actual_away) else None,
    # Convenience boolean for template/API so status display isn't dependent on inline set logic
    'is_final': (_is_valid_num(actual_home) and _is_valid_num(actual_away)),
    'predicted_win_margin': r2(_safe_float(game_row.get('model_margin'), game_row.get('predicted_win_margin', ''))),
        'home_win_prob_pct': f"{p_home_win*100:.1f}%" if p_home_win is not None else None,
        'away_win_prob_pct': f"{(1-p_home_win)*100:.1f}%" if p_home_win is not None else None,
        'home_win_prob': p_home_win,
        'win_margin_conf_lower': conf_lower,
        'win_margin_conf_upper': conf_upper,
        'win_margin_conf_std': conf_std,
        'home_logo': home_asset['logo'],
        'home_color': home_asset['color'],
        'home_alt_color': home_asset['alt_color'],
        'away_logo': away_asset['logo'],
        'away_color': away_asset['color'],
        'away_alt_color': away_asset['alt_color'],
        'betting_lines': betting_lines,
        'ou_line': r2(ou_line) if ou_line is not None else None,
        'ou_model_lean': ou_model_lean,
        'ou_edge': r2(ou_edge) if ou_edge is not None else None,
        'ou_edge_num': ou_edge,
        'ou_actual_result': ou_actual_result,
        'ou_correct': ou_correct,
        'ats_line': _format_ats_line(ats_home_line),
        'ats_model_lean': ats_model_lean,
        'ats_edge': r2(ats_edge) if ats_edge is not None else None,
        'ats_edge_num': ats_edge,
        'ats_actual_result': ats_actual_result,
        'ats_correct': ats_correct,
        'wx_temp_f': r2(wx_temp) if wx_temp is not None else None,
        'wx_wind_mph': r2(wx_wind) if wx_wind is not None else None,
        'wx_adjust_total': r2(wx_adj) if wx_adj is not None else None,
        'predicted_winner': predicted_winner,
        'actual_winner': actual_winner,
        'correct_prediction': correct_prediction,
    }

# --- Helper functions for analysis and betting ---
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _phi(z):
    # Standard normal CDF using error function
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

def american_to_decimal(odds):
    # Return decimal odds and net b (decimal-1)
    if odds is None:
        return None, None
    try:
        o = float(odds)
    except Exception:
        return None, None
    if o > 0:
        dec = 1 + (o / 100.0)
    else:
        dec = 1 + (100.0 / abs(o))
    return dec, dec - 1.0

def kelly_fraction(p, dec_odds):
    # p is win probability, dec_odds is decimal odds
    if dec_odds is None:
        return 0.0
    b = dec_odds - 1.0
    q = 1 - p
    numer = b * p - q
    if b <= 0:
        return 0.0
    return max(0.0, numer / b)

def _get_conf_std_for_game(row):
    # Try to fetch per-game std from win_margin_conf_df; fallback to default
    try:
        if win_margin_conf_df is not None:
            wk = int(row.get('week', 0))
            szn = int(row.get('season', 0))
            home = str(row.get('home_team', '')).strip().lower().replace('&','and').replace('  ',' ')
            away = str(row.get('away_team', '')).strip().lower().replace('&','and').replace('  ',' ')
            sel = win_margin_conf_df[
                (win_margin_conf_df['week'] == wk) &
                (win_margin_conf_df['season'] == szn) &
                (win_margin_conf_df['home_team'].str.strip().str.lower().str.replace('&','and').str.replace('  ',' ') == home) &
                (win_margin_conf_df['away_team'].str.strip().str.lower().str.replace('&','and').str.replace('  ',' ') == away)
            ]
            if not sel.empty:
                val = _safe_float(sel.iloc[0].get('conf_std', None))
                if val and val > 0:
                    return val
    except Exception:
        pass
    # Conference-level override (average home/away conf sigma if available)
    try:
        if _CONF_SIGMA is not None:
            hc = str(row.get('home_conference','')).strip()
            ac = str(row.get('away_conference','')).strip()
            vals = []
            if hc in _CONF_SIGMA:
                vals.append(_CONF_SIGMA[hc])
            if ac in _CONF_SIGMA:
                vals.append(_CONF_SIGMA[ac])
            if vals:
                v = float(sum(vals) / len(vals))
                if v > 4.0:  # sanity lower bound
                    return v
    except Exception:
        pass
    return 14.0  # reasonable default std for margin

RECS_PATH = os.path.join(DATA_DIR, "recommendations_2025.csv")


def _ensure_recs_file():
    if not os.path.exists(RECS_PATH):
        cols = [
            'timestamp','season','week','home_team','away_team','market','side','price_american','line','provider',
            'model_prob','implied_prob','edge','kelly_f','bankroll','stake','status','result','pnl'
        ]
        pd.DataFrame(columns=cols).to_csv(RECS_PATH, index=False)

# -------------------- Auto Refresh Infrastructure --------------------
_AUTO_REFRESH_LAST_MTIME = None
_AUTO_REFRESH_LOCK = threading.Lock()

def _current_scores_mtime():
    global pred_path_scores
    try:
        p = pred_path_scores
        if p and os.path.exists(p):
            return os.path.getmtime(p)
    except Exception:
        return None
    return None

def _settle_recommendations() -> dict:
    """Settle any open recommendations whose games are now final. Returns summary similar to performance endpoint but light."""
    try:
        if not os.path.exists(RECS_PATH):
            return {'skipped': 'no_recs_file'}
        recs_df = pd.read_csv(RECS_PATH)
        if recs_df.empty:
            return {'skipped': 'empty'}
    except Exception as e:
        return {'error': f'load_failed: {e}'}

    merged = recs_df.merge(
        pred_df[['season','week','home_team','away_team','actual_home_points','actual_away_points']],
        on=['season','week','home_team','away_team'], how='left'
    )
    pnl_updates = []
    wins = losses = pushes = 0
    for idx, r in merged.iterrows():
        if r.get('status','open') != 'open':
            continue
        ah = r.get('actual_home_points'); aa = r.get('actual_away_points')
        if pd.isna(ah) or pd.isna(aa):
            continue
        market = r['market']
        side = r['side']
        price = _safe_float(r.get('price_american', -110), -110)
        stake = _safe_float(r.get('stake', 0), 0)
        dec, _ = american_to_decimal(price)
        result = 'pending'; pnl = 0.0; push = False; win = False
        if market == 'ML':
            winner = 'Home' if ah > aa else ('Away' if aa > ah else 'Tie')
            push = (winner == 'Tie')
            win = (winner == side)
        elif market == 'Spread':
            line = _safe_float(r.get('line', None))
            if line is None:
                push = True
            else:
                margin = ah - aa
                if side == 'Home':
                    win = margin > line; push = abs(margin - line) < 1e-9
                else:
                    win = margin < line; push = abs(margin - line) < 1e-9
        elif market == 'Total':
            line = _safe_float(r.get('line', None))
            tot = ah + aa
            if line is None:
                push = True
            else:
                if side == 'Over':
                    win = tot > line; push = abs(tot - line) < 1e-9
                else:
                    win = tot < line; push = abs(tot - line) < 1e-9
        if push:
            result = 'push'; pnl = 0.0; pushes += 1
        elif win:
            result = 'win'; pnl = stake * (dec - 1); wins += 1
        else:
            result = 'loss'; pnl = -stake; losses += 1
        pnl_updates.append((idx, result, pnl))
    if pnl_updates:
        # apply to original dataframe (recs_df) using index alignment
        for idx, result, pnl in pnl_updates:
            recs_df.loc[idx, 'status'] = 'closed'
            recs_df.loc[idx, 'result'] = result
            recs_df.loc[idx, 'pnl'] = round(pnl, 2)
        try:
            recs_df.to_csv(RECS_PATH, index=False)
        except Exception:
            pass
    return {
        'settled': len(pnl_updates),
        'wins': wins, 'losses': losses, 'pushes': pushes,
        'open_remaining': int((recs_df['status'] == 'open').sum()) if 'status' in recs_df.columns else 0
    }

def _auto_refresh_loop():
    global _AUTO_REFRESH_LAST_MTIME
    interval = float(os.environ.get('AUTO_REFRESH_INTERVAL_SEC', '60'))
    while True:
        try:
            m = _current_scores_mtime()
            if m and (_AUTO_REFRESH_LAST_MTIME is None or m > _AUTO_REFRESH_LAST_MTIME):
                with _AUTO_REFRESH_LOCK:
                    _reload_predictions()
                    try:
                        _overlay_lines_2025_if_present()
                    except Exception:
                        pass
                    settle = _settle_recommendations()
                    _AUTO_REFRESH_LAST_MTIME = m
                    print(f"[auto-refresh] Reloaded predictions & settled recs: {settle}")
        except Exception as e:
            try:
                print(f"[auto-refresh] error: {e}")
            except Exception:
                pass
        time.sleep(interval)


def _reload_predictions():
    global pred_df
    # Invalidate API cache
    try:
        GAME_CARDS_CACHE.clear()
    except Exception:
        pass
    # Re-read predictions with validation
    new_df = _load_predictions_df()
    new_df['home_conference'] = new_df['home_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
    new_df['away_conference'] = new_df['away_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
    # Attempt to overlay model-based point predictions if model artifacts exist
    try:
        _overlay_model_predictions(new_df)
    except Exception as e:
        try:
            print(f"[model-overlay] failed: {e}")
        except Exception:
            pass
    pred_df = new_df

def _update_scores_with_cfbd(week: int | None = None, overwrite: bool = False) -> dict:
    """Update actual scores in the with_scores CSV using CFBD API.
    - Reads CFBD_API_KEY from environment; if missing, returns a skipped result.
    - If with_scores CSV doesn't exist but enhanced does, creates it by copying enhanced and adding actuals cols.
    - Matches games by (home_team, away_team) for season 2025 and optional week, with light normalization.
    Returns a small summary dict with counts and path touched.
    """
    try:
        import requests  # lightweight dep, widely available
    except Exception:
        return {'step': 'cfbd_update', 'skipped': 'requests_not_available'}

    api_key = os.environ.get('CFBD_API_KEY') or os.environ.get('CFBD_TOKEN') or os.environ.get('CFBD')
    use_espn_only = (os.environ.get('USE_ESPN_ONLY', '0') == '1') or (not api_key)
    use_espn_first = os.environ.get('USE_ESPN_FIRST', '1') != '0'
    cfbd_primary = os.environ.get('USE_CFBD_PRIMARY', '0') == '1'

    # Ensure we have a target CSV path to update
    global pred_path_scores, pred_path_enh
    target_path = pred_path_scores
    try:
        if not target_path:
            # If scores file missing, but enhanced exists, initialize scores file from enhanced
            if pred_path_enh and os.path.exists(pred_path_enh):
                target_path = os.path.join(DATA_DIR, "college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv")
                base_df = pd.read_csv(pred_path_enh)
                for col in ['actual_home_points','actual_away_points','start_date_api']:
                    if col not in base_df.columns:
                        base_df[col] = pd.NA
                base_df.to_csv(target_path, index=False)
                pred_path_scores = target_path
            else:
                return {'step': 'cfbd_update', 'skipped': 'no_scores_or_enhanced_file'}
        elif not os.path.exists(target_path):
            return {'step': 'cfbd_update', 'skipped': 'scores_path_not_found'}
    except Exception as e:
        return {'step': 'cfbd_update', 'error': f'prep_failed: {e}'}

    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        return {'step': 'cfbd_update', 'error': f'read_failed: {e}'}

    if df.empty or 'season' not in df.columns or 'home_team' not in df.columns or 'away_team' not in df.columns:
        return {'step': 'cfbd_update', 'skipped': 'scores_df_invalid'}

    def _norm_team_base(name: str) -> str:
        """Shared normalizer: lowercase, diacritics stripped, unify apostrophes, &->and, trim and collapse whitespace."""
        try:
            s = str(name or '').strip().lower()
            # Unify symbols
            s = s.replace('&', 'and')
            s = s.replace("ʻ", "'").replace("’", "'")
            # Fix common cases
            s = s.replace("hawai'i", "hawaii")
            # Strip diacritics (e.g., José -> Jose)
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            # Normalize punctuation -> space for broad match
            s = re.sub(r"[^a-z0-9 '\-\(\)]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s
        except Exception:
            return str(name)

    def _norm_team_cfbd(name: str) -> str:
        return _norm_team_base(name)

    # Cleanup: clear any accidental 0-0 writes (no 0-0 finals exist in college football)
    try:
        cleared = 0
        for i, r in df.iterrows():
            try:
                if int(r.get('season', 0)) != 2025:
                    continue
                ah = r.get('actual_home_points')
                aa = r.get('actual_away_points')
                if pd.notna(ah) and pd.notna(aa):
                    try:
                        ahv = float(ah)
                        aav = float(aa)
                    except Exception:
                        # handle strings like '0'
                        try:
                            ahv = float(str(ah).strip())
                            aav = float(str(aa).strip())
                        except Exception:
                            continue
                    if ahv == 0.0 and aav == 0.0:
                        df.at[i, 'actual_home_points'] = pd.NA
                        df.at[i, 'actual_away_points'] = pd.NA
                        cleared += 1
            except Exception:
                continue
        if cleared > 0:
            try:
                df.to_csv(target_path, index=False)
            except Exception:
                pass
    except Exception:
        pass

    # Build a lookup of games from CFBD
    base_url = 'https://api.collegefootballdata.com/games'
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    if week is None:
        weeks = [0, 1, 2]
    else:
        try:
            wv = int(week)
        except Exception:
            wv = None
        if wv in (0, 1):
            weeks = [wv, (1 if wv == 0 else 0)]
        elif wv is not None:
            weeks = [wv]
        else:
            weeks = [0, 1, 2]
    updated = 0
    fetched = 0
    games_map = {}       # keyed by (week, ht, at)
    games_map_nowk = {}  # keyed by (ht, at) for ESPN-only matching
    http_notes = []
    # Try a few parameter variants to avoid missing data due to filters
    def _variants(wk: int):
        base = {'year': 2025, 'week': wk}
        # Try multiple combos to cover FBS/FCS and unscoped queries
        combos = [
            {'seasonType': 'regular', 'division': 'fbs'},
            {'seasonType': 'regular', 'division': 'fcs'},
            {'division': 'fbs'},
            {'division': 'fcs'},
            {'seasonType': 'regular'},
            {'classification': 'fbs'},
            {'classification': 'fcs'},
            {'seasonType': 'regular', 'classification': 'fbs'},
            {'seasonType': 'regular', 'classification': 'fcs'},
            {},
        ]
        statuses = [None, 'completed', 'final']
        vars = []
        for c in combos:
            for st in statuses:
                pr = dict(base)
                pr.update(c)
                if st:
                    pr['status'] = st
                vars.append(pr)
        return vars
    # Optionally try ESPN first or exclusively
    def _fetch_espn_nowk():
        nonlocal games_map_nowk
        try:
            unique_dates = set()
            try:
                if 'start_date' in df.columns:
                    for _, r in df.iterrows():
                        try:
                            if int(r.get('season', 0)) != 2025:
                                continue
                            rw = int(r.get('week')) if pd.notna(r.get('week')) else None
                        except Exception:
                            rw = None
                        if week is not None and rw is not None and rw != int(week):
                            continue
                        d = pd.to_datetime(r.get('start_date'), errors='coerce')
                        if pd.notna(d):
                            unique_dates.add(d.date().isoformat())
            except Exception:
                pass
            # Always ensure broad Week 0/1 window (union), covers Labor Day Monday
            try:
                if week is None or int(week) in (0, 1):
                    unique_dates.update({'2025-08-23','2025-08-28','2025-08-29','2025-08-30','2025-08-31','2025-09-01','2025-09-02'})
            except Exception:
                unique_dates.update({'2025-08-23','2025-08-28','2025-08-29','2025-08-30','2025-08-31','2025-09-01','2025-09-02'})
            def _norm_team_generic(s):
                return _norm_team_base(s)
            for dstr in sorted(unique_dates):
                ymd = dstr.replace('-','')
                es_urls = [
                    f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={ymd}&groups=80',
                    f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={ymd}',
                ]
                for u in es_urls:
                    try:
                        r = requests.get(u, timeout=20, headers={'Accept':'application/json','User-Agent':'Mozilla/5.0'})
                        http_notes.append({'espn': True, 'date': dstr, 'url': u, 'status': getattr(r,'status_code',None), 'len': len(getattr(r,'content',b''))})
                        if r.status_code != 200:
                            continue
                        j = r.json() or {}
                        events = j.get('events') or []
                        for ev in events:
                            comps = (ev.get('competitions') or [{}])[0]
                            comps_list = comps.get('competitors') or []
                            if len(comps_list) != 2:
                                continue
                            home = next((c for c in comps_list if c.get('homeAway')=='home'), comps_list[0])
                            away = next((c for c in comps_list if c.get('homeAway')=='away'), comps_list[-1])
                            hteam = home.get('team') or {}
                            ateam = away.get('team') or {}
                            def _cands(t):
                                return [
                                    _norm_team_generic(t.get('location')),
                                    _norm_team_generic(t.get('shortDisplayName')),
                                    _norm_team_generic(t.get('displayName')),
                                    _norm_team_generic(t.get('abbreviation')),
                                ]
                            hcands = [c for c in _cands(hteam) if c]
                            acands = [c for c in _cands(ateam) if c]
                            try:
                                hp = int(home.get('score')) if home.get('score') is not None else None
                                ap = int(away.get('score')) if away.get('score') is not None else None
                            except Exception:
                                hp = home.get('score')
                                ap = away.get('score')
                            stype = (ev.get('status') or {}).get('type') or {}
                            state = str(stype.get('state','')).lower()  # 'pre', 'in', 'post'
                            nameu = str(stype.get('name','')).upper()
                            comp = (stype.get('completed') is True) or (state in ('post','final')) or (nameu in ('STATUS_FINAL','STATUS_FULL_TIME'))
                            if hp is None or ap is None:
                                continue
                            # Never index 0-0 from ESPN; no such finals exist
                            try:
                                if int(hp) == 0 and int(ap) == 0:
                                    continue
                            except Exception:
                                pass
                            if not comp:
                                # Skip non-final games to avoid premature writes
                                continue
                            for ht in hcands:
                                for at in acands:
                                    # Map in both orientations: (home, away) and (away, home) with swapped points
                                    games_map_nowk[(ht, at)] = {'home_points': hp, 'away_points': ap, 'completed': True}
                                    games_map_nowk[(at, ht)] = {'home_points': ap, 'away_points': hp, 'completed': True}
                        if games_map_nowk:
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    def _fetch_ncaa_nowk():
        """Lightweight NCAA.com scoreboard scrape (JSON) as alternative final score source.
        Controlled by env USE_NCAA_ALT=1. Fills games_map_nowk similar to ESPN fetch.
        URL pattern: https://data.ncaa.com/casablanca/scoreboard/football/fbs/YYYY/MM/DD/scoreboard.json
        Only records games with non-null scores and a final/completed status marker.
        """
        if os.environ.get('USE_NCAA_ALT','0') != '1':
            return
        nonlocal games_map_nowk
        try:
            unique_dates = set()
            try:
                if 'start_date' in df.columns:
                    for _, r in df.iterrows():
                        try:
                            if int(r.get('season',0)) != 2025:
                                continue
                            rw = int(r.get('week')) if pd.notna(r.get('week')) else None
                        except Exception:
                            rw = None
                        if week is not None and rw is not None and rw != int(week):
                            continue
                        d = pd.to_datetime(r.get('start_date'), errors='coerce')
                        if pd.notna(d):
                            unique_dates.add(d.date())
            except Exception:
                pass
            for d in sorted(unique_dates):
                try:
                    y = d.year; m = f"{d.month:02d}"; dd = f"{d.day:02d}"
                    url = f"https://data.ncaa.com/casablanca/scoreboard/football/fbs/{y}/{m}/{dd}/scoreboard.json"
                    r = requests.get(url, timeout=15, headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'})
                    http_notes.append({'ncaa': True, 'date': d.isoformat(), 'status': getattr(r,'status_code',None)})
                    if r.status_code != 200:
                        continue
                    j = r.json() or {}
                    games = j.get('scoreboard') or j.get('games') or j.get('events') or []
                    if isinstance(games, dict):
                        # sometimes nested key
                        games = games.get('games') or []
                    for g in games:
                        try:
                            home = g.get('home') or {}
                            away = g.get('away') or {}
                            hp = home.get('score') or home.get('homeScore') or home.get('runs')
                            ap = away.get('score') or away.get('awayScore') or away.get('runs')
                            status_txt = (g.get('status') or g.get('finalMessage') or '').lower()
                            if hp in (None,'') or ap in (None,''):
                                continue
                            try:
                                hp_i = int(hp); ap_i = int(ap)
                                if hp_i == 0 and ap_i == 0:
                                    # Avoid 0-0 placeholders
                                    continue
                            except Exception:
                                pass
                            is_final = any(s in status_txt for s in ('final','completed'))
                            if not is_final:
                                continue
                            def _cand_names(obj):
                                return [
                                    _norm_team_base(obj.get('names',{}).get('short')), _norm_team_base(obj.get('names',{}).get('display')),
                                    _norm_team_base(obj.get('shortname')), _norm_team_base(obj.get('name')),
                                    _norm_team_base(obj.get('abbrev'))
                                ]
                            hcands = [c for c in _cand_names(home) if c]
                            acands = [c for c in _cand_names(away) if c]
                            for ht in hcands:
                                for at in acands:
                                    games_map_nowk[(ht, at)] = {'home_points': hp, 'away_points': ap, 'completed': True}
                                    games_map_nowk[(at, ht)] = {'home_points': ap, 'away_points': hp, 'completed': True}
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass

    # Fetch order control: allow choosing CFBD as primary when USE_CFBD_PRIMARY=1
    if use_espn_only:
        _fetch_espn_nowk(); _fetch_ncaa_nowk()
    else:
        if cfbd_primary:
            # Defer ESPN until after CFBD attempt to favor CFBD data where both exist
            pass
        else:
            if use_espn_first:
                _fetch_espn_nowk(); _fetch_ncaa_nowk()

    # CFBD fetch only if not ESPN-only and if we still need more
    # CFBD fetch: when CFBD is primary we always attempt it (even if ESPN returned);
    # otherwise mimic original behavior (only if ESPN did not produce any finals yet)
    if not use_espn_only and (cfbd_primary or not games_map_nowk):
        try:
            for wk in weeks:
                for pr in _variants(wk):
                    try:
                        resp = requests.get(base_url, headers=headers, params=pr, timeout=25)
                        http_notes.append({'week': wk, 'status': getattr(resp, 'status_code', None), 'len': len(getattr(resp, 'content', b''))})
                        if resp.status_code != 200:
                            continue
                        data = resp.json() or []
                        if not data:
                            continue
                        fetched += len(data)
                        for g in data:
                            ht = _norm_team_cfbd(g.get('home_team'))
                            at = _norm_team_cfbd(g.get('away_team'))
                            hp = g.get('home_points')
                            ap = g.get('away_points')
                            comp = g.get('completed')
                            if hp is None or ap is None:
                                continue
                            # Skip 0-0 non-finals or erroneous entries
                            try:
                                if int(hp) == 0 and int(ap) == 0:
                                    continue
                            except Exception:
                                pass
                            games_map[(wk, ht, at)] = {'home_points': hp, 'away_points': ap, 'completed': comp}
                        if any(k[0] == wk for k in games_map.keys()):
                            break
                    except Exception:
                        continue
        except Exception:
            return {'step': 'cfbd_update', 'error': 'fetch_failed'}

    if not games_map and not use_espn_only and not games_map_nowk:
        # Fallback 1: CFBD scoreboard endpoint which sometimes surfaces scores earlier
        try:
            sb_url = 'https://api.collegefootballdata.com/scoreboard'
            def _score_of(obj, key_candidates):
                for k in key_candidates:
                    if k in obj and obj[k] is not None:
                        return obj[k]
                return None
            for wk in weeks:
                for pr in _variants(wk):
                    params = {k:v for k,v in pr.items() if k in ('year','week','seasonType','division','classification','status')}
                    resp = requests.get(sb_url, headers=headers, params=params, timeout=25)
                    http_notes.append({'scoreboard': True, 'week': wk, 'status': getattr(resp, 'status_code', None), 'len': len(getattr(resp, 'content', b''))})
                    if resp.status_code != 200:
                        continue
                    data = resp.json() or {}
                    games = data.get('games') or data.get('events') or data.get('data') or []
                    if not games:
                        continue
                    for g in games:
                        ht = _norm_team_cfbd(g.get('home_team') or g.get('home') or g.get('homeTeam') or g.get('home_name'))
                        at = _norm_team_cfbd(g.get('away_team') or g.get('away') or g.get('awayTeam') or g.get('away_name'))
                        hp = _score_of(g, ['home_points','home_points_total','home_score','homeScore'])
                        ap = _score_of(g, ['away_points','away_points_total','away_score','awayScore'])
                        comp = g.get('completed') or g.get('status') in ('completed','final','Final')
                        if hp is None or ap is None:
                            continue
                        # Skip 0-0 entries
                        try:
                            if int(hp) == 0 and int(ap) == 0:
                                continue
                        except Exception:
                            pass
                        games_map[(wk, ht, at)] = {'home_points': hp, 'away_points': ap, 'completed': comp}
                    if any(k[0] == wk for k in games_map.keys()):
                        break
        except Exception:
            pass
    # ESPN fetch already handled above in ESPN-first path
    # If CFBD was primary and ESPN not yet fetched, try ESPN now to fill gaps
    if (cfbd_primary and not use_espn_only):
        try:
            if not games_map_nowk:
                _fetch_espn_nowk(); _fetch_ncaa_nowk()
        except Exception:
            pass

    if not games_map and not games_map_nowk:
        return {'step': 'cfbd_update', 'skipped': 'no_games_from_api', 'notes': http_notes, 'cfbd_primary': cfbd_primary}

    # Apply updates into our CSV
    try:
        changed_rows = 0
        # Apply rows
        for i, r in df.iterrows():
            try:
                if int(r.get('season', 0)) != 2025:
                    continue
            except Exception:
                continue
            # Optional week filter
            try:
                rw = int(r.get('week')) if pd.notna(r.get('week')) else None
            except Exception:
                rw = None
            if week is not None and rw is not None and rw != int(week):
                continue
            # Skip if already has actuals
            if not overwrite:
                if pd.notna(r.get('actual_home_points')) and pd.notna(r.get('actual_away_points')):
                    continue
            ht = _norm_team_cfbd(r.get('home_team'))
            at = _norm_team_cfbd(r.get('away_team'))
            # Try exact week key first, then try the other label (0<->1) for 2025 mismatch tolerance
            keys = []
            if rw is not None:
                keys.append((rw, ht, at))
                if rw in (0,1):
                    keys.append(((1 if rw == 0 else 0), ht, at))
            else:
                for wk in weeks:
                    keys.append((wk, ht, at))
            hit = None
            for k in keys:
                if k in games_map:
                    hit = games_map[k]
                    break
            if hit is None and games_map_nowk:
                # Fall back to ESPN no-week map using normalized team names
                hit = games_map_nowk.get((ht, at))
            if hit is None:
                continue
            if not hit.get('completed'):
                continue
            hp = hit['home_points']
            ap = hit['away_points']
            try:
                if overwrite:
                    if hp is not None:
                        df.at[i, 'actual_home_points'] = int(hp)
                    if ap is not None:
                        df.at[i, 'actual_away_points'] = int(ap)
                else:
                    if pd.isna(r.get('actual_home_points')) and hp is not None:
                        df.at[i, 'actual_home_points'] = int(hp)
                    if pd.isna(r.get('actual_away_points')) and ap is not None:
                        df.at[i, 'actual_away_points'] = int(ap)
                changed_rows += 1
            except Exception:
                continue
        if changed_rows > 0:
            df.to_csv(target_path, index=False)
            updated = changed_rows
    except Exception as e:
        return {'step': 'cfbd_update', 'error': f'apply_failed: {e}'}

    return {'step': 'cfbd_update', 'updated_rows': int(updated), 'fetched_games': int(fetched), 'path': target_path, 'notes': http_notes}

def _geocode_needed(base_dir: str, selected_week: int | None) -> bool:
    try:
        sched_path = os.path.join(DATA_DIR, 'college_football_schedule_2025.csv')
        cache_path = os.path.join(DATA_DIR, 'venue_coordinates_cache.csv')
        if not os.path.exists(sched_path):
            return False
        df = pd.read_csv(sched_path)
        if selected_week is not None and 'week' in df.columns:
            try:
                df = df[df['week'] == int(selected_week)].copy()
            except Exception:
                pass
        if df.empty:
            return False
        if 'venue' not in df.columns or 'home_team' not in df.columns:
            return False
        df['venue_norm'] = df['venue'].astype(str).str.strip().str.lower()
        req = set((vn, ht) for vn, ht in zip(df['venue_norm'], df['home_team']))
        if not os.path.exists(cache_path):
            return True
        cache = pd.read_csv(cache_path)
        if cache.empty:
            return True
        if 'venue_norm' not in cache.columns or 'home_team' not in cache.columns:
            return True
        cache['venue_norm'] = cache['venue_norm'].astype(str).str.strip().str.lower()
        have = set((vn, ht) for vn, ht in zip(cache['venue_norm'], cache['home_team']))
        missing = req - have
        # If more than 0 missing, we need geocoding
        return len(missing) > 0
    except Exception:
        # On any error, be conservative and run geocode once
        return True

@app.route('/api/debug-actuals')
def debug_actuals():
    try:
        sub = pred_df[(pred_df['season'] == 2025)].copy()
        have = sub[sub['actual_home_points'].notna() & sub['actual_away_points'].notna()]
        sample = []
        for _, r in have.head(10).iterrows():
            sample.append({
                'week': int(r.get('week', 0)) if pd.notna(r.get('week')) else None,
                'home_team': r.get('home_team'),
                'away_team': r.get('away_team'),
                'actual_home_points': r.get('actual_home_points'),
                'actual_away_points': r.get('actual_away_points'),
            })
        return {
            'rows_2025': int(len(sub)),
            'completed_count': int(len(have)),
            'sample': sample,
            'pred_source': PRED_SOURCE,
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/debug-pred-source')
def debug_pred_source():
    try:
        sub = pred_df[pred_df['season'] == 2025]
        uh = sub['predicted_home_points'].nunique(dropna=True) if 'predicted_home_points' in sub.columns else None
        ua = sub['predicted_away_points'].nunique(dropna=True) if 'predicted_away_points' in sub.columns else None
        ut = sub['predicted_total_points'].nunique(dropna=True) if 'predicted_total_points' in sub.columns else None
        return {
            'pred_source': PRED_SOURCE,
            'rows': int(len(pred_df)),
            'rows_2025': int(len(sub)),
            'unique_home_preds': int(uh) if uh is not None else None,
            'unique_away_preds': int(ua) if ua is not None else None,
            'unique_total_preds': int(ut) if ut is not None else None,
        }, 200
    except Exception as e:
        return {'error': str(e), 'pred_source': PRED_SOURCE}, 500


@app.route('/api/debug-week-counts')
def debug_week_counts():
    try:
        df = pred_df[pred_df.get('season', 0) == 2025].copy()
        if df.empty:
            return {'message': 'No 2025 rows loaded', 'pred_source': PRED_SOURCE}, 200
        weeks = sorted([int(w) for w in pd.to_numeric(df['week'], errors='coerce').dropna().unique()]) if 'week' in df.columns else []
        counts = {}
        for w in weeks:
            sub = df[df['week'] == w]
            counts[int(w)] = int(len(sub))
        # sample a few from week 0 and 1
        def sample_week(w):
            if 'start_date' in df.columns:
                cols = ['week','away_team','home_team','start_date']
            else:
                cols = ['week','away_team','home_team']
            sub = df[df['week'] == w]
            return sub[cols].head(5).to_dict(orient='records') if not sub.empty else []
        return {
            'pred_source': PRED_SOURCE,
            'weeks': weeks,
            'counts': counts,
            'sample_w0': sample_week(0),
            'sample_w1': sample_week(1),
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/debug-missing-odds')
def debug_missing_odds():
    """List games missing real bookmaker odds (ignores synthetic ModelImplied). Accepts ?season=&week=."""
    try:
        season = request.args.get('season', default=2025, type=int)
        week = request.args.get('week', default=None, type=int)
        df = pred_df[pred_df.get('season', 0) == season].copy()
        if week is not None:
            df = df[df.get('week', 0) == week]
        out = []
        for _, r in df.iterrows():
            lines = get_betting_lines(r['season'], r['week'], r['home_team'], r['away_team'])
            if not any(not l.get('synthetic') for l in lines):
                out.append({
                    'season': int(r['season']),
                    'week': int(r['week']),
                    'home': r['home_team'],
                    'away': r['away_team'],
                    'model_spread': _safe_float(r.get('model_margin')),
                    'model_total': _safe_float(r.get('model_total_points')),
                })
        return {'count': len(out), 'games': out[:1000]}, 200
    except Exception as e:
        return {'error': str(e)}, 500


def compute_recommendations(week=None, bankroll=1000.0, kelly_factor=0.5, ev_threshold=0.02):
    """Core engine to compute EV+ recommendations, reused by API and UI."""
    df = pred_df[(pred_df['season'] == 2025)].copy()
    # Upcoming only
    df = df[df['actual_home_points'].isna() & df['actual_away_points'].isna()]
    if week is not None:
        try:
            df = df[df['week'] == int(week)]
        except Exception:
            pass
    recs = []
    kelly_cap = 0.10  # never stake >10% per bet
    longshot_cap_odds = 4.0  # decimal (>4.0 ~= +300)
    min_prob_for_longshot = 0.30
    for _, row in df.iterrows():
        odds_list = get_betting_lines(int(row['season']), int(row['week']), row['home_team'], row['away_team'])
        if not odds_list:
            continue
        # Prefer model-based point predictions
        pred_home = _safe_float(row.get('model_home_points'))
        if pred_home is None:
            pred_home = _safe_float(row.get('predicted_home_points'))
        pred_away = _safe_float(row.get('model_away_points'))
        if pred_away is None:
            pred_away = _safe_float(row.get('predicted_away_points'))
        if pred_home is None or pred_away is None:
            continue
        pred_total = pred_home + pred_away
        pred_margin = _safe_float(row.get('model_margin'))
        if pred_margin is None:
            pred_margin = _safe_float(row.get('predicted_win_margin'))
        if pred_margin is None:
            pred_margin = pred_home - pred_away
        sigma_m = _get_conf_std_for_game(row)
        sigma_t = 12.0
        for odds in odds_list:
            provider = odds.get('provider')
            # ML
            home_ml = _safe_float(odds.get('homeMoneyline'))
            away_ml = _safe_float(odds.get('awayMoneyline'))
            if home_ml is not None:
                p_home = _safe_float(row.get('model_home_win_prob'))
                if p_home is None:
                    p_home = _phi(pred_margin / sigma_m)
                dec, _ = american_to_decimal(home_ml)
                if dec:
                    kf = None
                    # Skip extreme longshots unless model prob decent
                    if not (dec > longshot_cap_odds and p_home < min_prob_for_longshot):
                        kf = kelly_fraction(p_home, dec)
                        kf = min(kf, kelly_cap)
                        # Scale down for longshots
                        if dec > longshot_cap_odds:
                            kf *= 0.25
                    ev = p_home * (dec - 1) - (1 - p_home)
                    if ev > ev_threshold and kf is not None and kf > 0:
                        stake = round(bankroll * kf * kelly_factor, 2)
                        recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'ML', 'side': 'Home', 'provider': provider, 'price_american': int(home_ml), 'model_prob': round(p_home,4), 'implied_prob': round(1/(dec),4), 'edge': round(ev,4), 'kelly_f': round(kf,4), 'stake': stake})
            if away_ml is not None:
                p_home_tmp = _safe_float(row.get('model_home_win_prob'))
                if p_home_tmp is None:
                    p_home_tmp = _phi(pred_margin / sigma_m)
                p_away = max(0.0, 1.0 - p_home_tmp)
                dec, _ = american_to_decimal(away_ml)
                if dec:
                    kf = None
                    if not (dec > longshot_cap_odds and p_away < min_prob_for_longshot):
                        kf = kelly_fraction(p_away, dec)
                        kf = min(kf, kelly_cap)
                        if dec > longshot_cap_odds:
                            kf *= 0.25
                    ev = p_away * (dec - 1) - (1 - p_away)
                    if ev > ev_threshold and kf is not None and kf > 0:
                        stake = round(bankroll * kf * kelly_factor, 2)
                        recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'ML', 'side': 'Away', 'provider': provider, 'price_american': int(away_ml), 'model_prob': round(p_away,4), 'implied_prob': round(1/(dec),4), 'edge': round(ev,4), 'kelly_f': round(kf,4), 'stake': stake})
            # Spread / Totals
            spread = odds.get('spread')
            try:
                spread_val = float(str(spread).replace('Home','').replace('Away','').strip()) if spread is not None else None
            except Exception:
                spread_val = None
            ou_val = _safe_float(odds.get('overUnder'))
            dec_110 = 1 + (100/110)
            if spread_val is not None:
                p_home_cover = _phi((pred_margin - spread_val) / sigma_m)
                ev_home = p_home_cover * (dec_110 - 1) - (1 - p_home_cover)
                kf_home = min(kelly_fraction(p_home_cover, dec_110), kelly_cap)
                if ev_home > ev_threshold and kf_home > 0:
                    stake = round(bankroll * kf_home * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Spread', 'side': 'Home', 'provider': provider, 'price_american': -110, 'model_prob': round(p_home_cover,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_home,4), 'kelly_f': round(kf_home,4), 'stake': stake, 'line': spread_val})
                p_away_cover = 1 - p_home_cover
                ev_away = p_away_cover * (dec_110 - 1) - (1 - p_away_cover)
                kf_away = min(kelly_fraction(p_away_cover, dec_110), kelly_cap)
                if ev_away > ev_threshold and kf_away > 0:
                    stake = round(bankroll * kf_away * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Spread', 'side': 'Away', 'provider': provider, 'price_american': -110, 'model_prob': round(p_away_cover,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_away,4), 'kelly_f': round(kf_away,4), 'stake': stake, 'line': spread_val})
            if ou_val is not None:
                p_over = 1 - _phi((ou_val - pred_total) / sigma_t)
                ev_over = p_over * (dec_110 - 1) - (1 - p_over)
                kf_over = min(kelly_fraction(p_over, dec_110), kelly_cap)
                if ev_over > ev_threshold and kf_over > 0:
                    stake = round(bankroll * kf_over * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Total', 'side': 'Over', 'provider': provider, 'price_american': -110, 'model_prob': round(p_over,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_over,4), 'kelly_f': round(kf_over,4), 'stake': stake, 'line': ou_val})
                p_under = 1 - p_over
                ev_under = p_under * (dec_110 - 1) - (1 - p_under)
                kf_under = min(kelly_fraction(p_under, dec_110), kelly_cap)
                if ev_under > ev_threshold and kf_under > 0:
                    stake = round(bankroll * kf_under * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Total', 'side': 'Under', 'provider': provider, 'price_american': -110, 'model_prob': round(p_under,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_under,4), 'kelly_f': round(kf_under,4), 'stake': stake, 'line': ou_val})
    recs.sort(key=lambda x: x['edge'], reverse=True)
    return recs

def _confidence_tier(edge: float | None, kelly_f: float | None, model_prob: float | None) -> tuple[str, int]:
    try:
        e = float(edge) if edge is not None else 0.0
    except Exception:
        e = 0.0
    try:
        k = float(kelly_f) if kelly_f is not None else 0.0
    except Exception:
        k = 0.0
    try:
        p = float(model_prob) if model_prob is not None else 0.0
    except Exception:
        p = 0.0
    score = 0
    # Edge thresholds
    if e >= 0.05:
        score += 2
    elif e >= 0.03:
        score += 1
    # Kelly fraction thresholds (scaled stake sizing)
    if k >= 0.03:
        score += 2
    elif k >= 0.015:
        score += 1
    # Model probability threshold
    if p >= 0.60:
        score += 1
    tier = 'Low'
    if score >= 3:
        tier = 'High'
    elif score >= 2:
        tier = 'Medium'
    return tier, score

def _parse_start_ts(row: pd.Series) -> tuple[str, float | None, str]:
    """Return (start_iso, sort_ts, display_time) from a predictions row."""
    # Local helper matches logic in _build_game_card for robust parsing
    def _is_bad_date_val(v):
        try:
            if v is None:
                return True
            if isinstance(v, float) and math.isnan(v):
                return True
            if isinstance(v, str):
                s = v.strip().lower()
                return s in ('', 'nan', 'nat', 'none', 'null')
            return False
        except Exception:
            return True
    raw_api = row.get('start_date_api', None)
    raw_sd = row.get('start_date', None)
    val_api = None if _is_bad_date_val(raw_api) else str(raw_api).strip()
    val_sd = None if _is_bad_date_val(raw_sd) else str(raw_sd).strip()
    display_time = val_sd or val_api or ''
    start_iso = ''
    sort_ts = None
    candidates = [c for c in [val_api, val_sd] if c]
    for c in candidates:
        try:
            s = c
            if 'T' not in s and ' ' in s:
                s = s.replace(' ', 'T')
            dt_obj = None
            try:
                if s.endswith('Z'):
                    dt_obj = datetime.fromisoformat(s.replace('Z', '+00:00'))
                else:
                    dt_obj = datetime.fromisoformat(s)
            except Exception:
                try:
                    dt_obj = pd.to_datetime(c, errors='coerce').to_pydatetime() if c else None
                except Exception:
                    dt_obj = None
            if not dt_obj:
                continue
            # Ensure timezone-aware UTC
            if getattr(dt_obj, 'tzinfo', None) is None:
                dt_utc = dt_obj.replace(tzinfo=pytz.UTC)
            else:
                dt_utc = dt_obj.astimezone(pytz.UTC)
            sort_ts = dt_utc.timestamp()
            start_iso = dt_utc.isoformat().replace('+00:00', 'Z')
            try:
                display_time = dt_utc.strftime('%a, %b %d, %Y, %I:%M %p UTC')
            except Exception:
                display_time = start_iso
            break
        except Exception:
            continue
    return start_iso, sort_ts, display_time

@app.route('/api/recommendations', methods=['GET'])
def recommendations_api():
    """Return EV+ betting recommendations for a given week with confidence and sorting.
    Query params:
      - week: int (default: auto-upcoming from predictions)
      - bankroll: float (default 1000)
      - kelly: float scale factor (default 0.5)
      - ev: float minimum EV threshold (default 0.02)
      - market: optional filter in {ML, Spread, Total}
      - sort: one of {time, confidence_desc, market, edge_desc, stake_desc, prob_desc} (default edge_desc)
      - limit: int cap number of results
    """
    try:
        week_q = request.args.get('week')
        bankroll = float(request.args.get('bankroll', 1000.0))
        kelly_factor = float(request.args.get('kelly', 0.5))
        ev_threshold = float(request.args.get('ev', 0.02))
        market_filter = request.args.get('market')
        sort_key = request.args.get('sort', 'edge_desc')
        limit = request.args.get('limit')
        limit = int(limit) if (limit and str(limit).isdigit()) else None

        week_val = int(week_q) if (week_q and str(week_q).isdigit()) else None
        recs = compute_recommendations(week=week_val, bankroll=bankroll, kelly_factor=kelly_factor, ev_threshold=ev_threshold)
        # Attach timing and confidence
        out = []
        # Build a quick index by (season,week,home,away) to get start time
        idx = {}
        try:
            df2025 = pred_df[(pred_df.get('season', 0) == 2025)].copy()
            for _, r in df2025.iterrows():
                key = (int(r['season']), int(r['week']), str(r['home_team']), str(r['away_team']))
                idx[key] = r
        except Exception:
            idx = {}
        for rec in recs:
            tier, score = _confidence_tier(rec.get('edge'), rec.get('kelly_f'), rec.get('model_prob'))
            key = (rec['season'], rec['week'], rec['home_team'], rec['away_team'])
            row = idx.get(key)
            start_iso, sort_ts, display_time = _parse_start_ts(row) if row is not None else ('', None, '')
            try:
                ha = get_team_asset(rec['home_team'])
            except Exception:
                ha = {'logo': '', 'color': '', 'alt_color': ''}
            try:
                aa = get_team_asset(rec['away_team'])
            except Exception:
                aa = {'logo': '', 'color': '', 'alt_color': ''}
            ent = {
                **rec,
                'confidence': tier,
                'confidence_score': score,
                'start_iso': start_iso,
                'sort_ts': sort_ts,
                'game_time': display_time,
                'home_logo': ha.get('logo',''),
                'away_logo': aa.get('logo',''),
                'home_color': ha.get('color',''),
                'away_color': aa.get('color',''),
            }
            if market_filter and ent.get('market') != market_filter:
                continue
            out.append(ent)
        # Sorting
        try:
            if sort_key == 'time':
                out.sort(key=lambda x: (x.get('sort_ts') is None, x.get('sort_ts') or 0.0))
            elif sort_key == 'confidence_desc':
                out.sort(key=lambda x: (x.get('confidence_score') or 0, x.get('edge') or 0.0), reverse=True)
            elif sort_key == 'market':
                out.sort(key=lambda x: (str(x.get('market','')), x.get('sort_ts') or 0.0))
            elif sort_key == 'stake_desc':
                out.sort(key=lambda x: (x.get('stake') or 0.0), reverse=True)
            elif sort_key == 'prob_desc':
                out.sort(key=lambda x: (x.get('model_prob') or 0.0), reverse=True)
            else:  # edge_desc
                out.sort(key=lambda x: (x.get('edge') or 0.0), reverse=True)
        except Exception:
            pass
        if limit is not None and limit > 0:
            out = out[:limit]
        return jsonify({'count': len(out), 'week': week_val, 'sort': sort_key, 'results': out}), 200
    except Exception as e:
        return {'error': str(e)}, 500
@app.route('/api/build-calibration', methods=['POST'])
def build_calibration():
    """Generate isotonic LUT and conference sigma CSV from historical data."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(base_dir, 'src', 'modeling', 'build_calibration_artifacts.py')
        out = subprocess.run([sys.executable, script], capture_output=True, text=True, check=False)
        return {'returncode': out.returncode, 'stdout': out.stdout[-8000:], 'stderr': out.stderr[-8000:]}, 200
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/', methods=['GET', 'POST'])
def index():
    # Fast path for platform HEAD probes
    if request.method == 'HEAD':
        return '', 200
    # Default to upcoming games for the current week if not POST
    import datetime as dt
    weeks = sorted(pred_df['week'].dropna().unique())
    selected_week = weeks[0] if weeks else None
    if request.method == 'GET':
        # Query param driven (read-only)
        filter_type = request.args.get('filter_type', 'all')
        # Week selection
        week_q = request.args.get('week')
        try:
            if week_q is not None and week_q.isdigit():
                w_int = int(week_q)
                if w_int in weeks:
                    selected_week = w_int
        except Exception:
            pass
        if weeks and selected_week is None:  # auto-pick most recent with finals
            try:
                finals_counts = {}
                for w in weeks:
                    subw = pred_df[pred_df['week']==w]
                    finals_counts[w] = int((subw['actual_home_points'].notna() & subw['actual_away_points'].notna()).sum())
                done_weeks = [w for w,c in finals_counts.items() if c>0]
                selected_week = max(done_weeks) if done_weeks else min(weeks)
            except Exception:
                selected_week = weeks[0]
        week_games = pred_df[pred_df['week']==int(selected_week)].copy() if selected_week is not None else pred_df.copy()
        week_games['date_only'] = week_games.get('start_date','').astype(str).str[:10]
        all_dates = sorted([d for d in week_games['date_only'].dropna().unique() if d])
        selected_date = request.args.get('date','')
        selected_conference = request.args.get('conference','')
        show_all = request.args.get('show_all','0').lower() in ('1','true','yes')
        include_non_fbs = request.args.get('include_non_fbs','0').lower() in ('1','true','yes')
        hide_both_unknown = not include_non_fbs  # deprecated flag retained for minimal downstream condition usage
        want_full = request.args.get('full','0').lower() in ('1','true','yes')
        sort_by = request.args.get('sort_by','time')
        filtered_games = week_games.copy()
        if selected_date and not show_all:
            filtered_games = filtered_games[filtered_games['date_only']==selected_date]
        if selected_conference:
            filtered_games = filtered_games[(filtered_games['home_conference']==selected_conference) | (filtered_games['away_conference']==selected_conference)]
        eff_filter = (filter_type if not show_all else 'all')
        if eff_filter == 'completed':
            filtered_games = filtered_games[(filtered_games['actual_home_points'].notna()) & (filtered_games['actual_away_points'].notna())]
        elif eff_filter == 'upcoming':
            filtered_games = filtered_games[(filtered_games['actual_home_points'].isna()) & (filtered_games['actual_away_points'].isna())]
        if hide_both_unknown:
            filtered_games = filtered_games[~((filtered_games['home_conference']=='Unknown') & (filtered_games['away_conference']=='Unknown'))]
        try:
            if {'week','home_team','away_team'}.issubset(filtered_games.columns):
                filtered_games = filtered_games.sort_values(by=['start_date','home_team','away_team']).drop_duplicates(subset=['week','home_team','away_team'], keep='first')
        except Exception:
            pass
        if not want_full and not show_all:
            try:
                finals_mask = filtered_games['actual_home_points'].notna() & filtered_games['actual_away_points'].notna()
                finals_df = filtered_games[finals_mask]
                upcoming_df = filtered_games[~finals_mask]
                filtered_games = pd.concat([finals_df, upcoming_df.head(80)], ignore_index=True)
            except Exception:
                filtered_games = filtered_games.head(80)
    else:
        # POST: Use form data to filter games
        filter_type = request.form.get('filter_type', 'all')
        selected_week = int(request.form.get('week', weeks[0] if weeks else 1))
        selected_date = request.form.get('date', '')
        selected_conference = request.form.get('conference', '')
        show_all = bool(request.form.get('show_all'))
        include_non_fbs = bool(request.form.get('include_non_fbs'))
        hide_both_unknown = not include_non_fbs
        sort_by = request.form.get('sort_by', 'time')
        week_games = pred_df[pred_df['week'] == int(selected_week)].copy() if selected_week else pred_df.copy()
        week_games['date_only'] = week_games.get('start_date','').astype(str).str[:10]
        all_dates = sorted([d for d in week_games['date_only'].dropna().unique() if d])
        filtered_games = week_games.copy()
        if selected_date and not show_all:
            filtered_games = filtered_games[filtered_games['date_only'] == selected_date]
        if selected_conference:
            filtered_games = filtered_games[(filtered_games['home_conference'] == selected_conference) | (filtered_games['away_conference'] == selected_conference)]
        effective_filter = (filter_type if not show_all else 'all')
        if effective_filter == 'completed':
            filtered_games = filtered_games[(filtered_games['actual_home_points'].notnull()) & (filtered_games['actual_away_points'].notnull())]
        elif effective_filter == 'upcoming':
            filtered_games = filtered_games[(filtered_games['actual_home_points'].isnull()) & (filtered_games['actual_away_points'].isnull())]
        if hide_both_unknown:
            try:
                filtered_games = filtered_games[~((filtered_games['home_conference'] == 'Unknown') & (filtered_games['away_conference'] == 'Unknown'))]
            except Exception:
                pass
        if show_all and len(filtered_games) <= 1 and 'season' in pred_df.columns:
            try:
                filtered_games = pred_df[pred_df['season'] == 2025].copy()
            except Exception:
                pass
        # Do not cap POST results; user explicitly filtered

    # Compute finals banner metrics (based on full week dataset, not filtered slice cap)
    try:
        week_scope_df = pred_df[pred_df['week']==selected_week] if selected_week is not None else pred_df
        finals_count_week = int(((week_scope_df['actual_home_points'].notna()) & (week_scope_df['actual_away_points'].notna())).sum())
        total_games_week = int(len(week_scope_df))
        finals_pct_week = (f"{(finals_count_week/total_games_week*100):.1f}%" if total_games_week>0 else '—')
        unknown_pending = int(((week_scope_df['home_conference']=='Unknown') & (week_scope_df['away_conference']=='Unknown') & (week_scope_df['actual_home_points'].isna()) & (week_scope_df['actual_away_points'].isna())).sum())
    except Exception:
        finals_count_week = 0; total_games_week = 0; finals_pct_week='—'; unknown_pending=0

    # Prepare game cards for all filtered games (fixed loop)
    game_cards = []
    try:
        print('[index] building cards: week', selected_week, 'rows', len(filtered_games))
        print('[index] sample rows:', [f"{r.away_team} at {r.home_team}" for r in filtered_games.head(3).itertuples()])
    except Exception:
        pass
    for _, game_row in filtered_games.iterrows():
        try:
            card = _build_game_card(game_row)
            # TEMP DEBUG: log first few to inspect actual vs predicted vs is_final
            if len(game_cards) < 5:
                try:
                    print('[debug-card]', card['away_team'], 'at', card['home_team'], 'ah=', card.get('actual_home_points'), 'aa=', card.get('actual_away_points'), 'is_final=', card.get('is_final'))
                except Exception:
                    pass
            game_cards.append(card)
        except Exception:
            # Skip any problematic row but continue rendering others
            continue
    # Summary metrics for this view
    summary = { 'winners': {'correct':0,'total':0}, 'ou': {'correct':0,'push':0,'total':0}, 'ats': {'correct':0,'push':0,'total':0} }
    for g in game_cards:
        if g.get('correct_prediction') is not None:
            summary['winners']['total'] += 1
            if g['correct_prediction']:
                summary['winners']['correct'] += 1
        if g.get('ou_actual_result'):
            if g['ou_actual_result'] == 'Push':
                summary['ou']['push'] += 1
            else:
                summary['ou']['total'] += 1
                if g.get('ou_correct') is True:
                    summary['ou']['correct'] += 1
        if g.get('ats_actual_result'):
            if g['ats_actual_result'] == 'Push':
                summary['ats']['push'] += 1
            else:
                summary['ats']['total'] += 1
                if g.get('ats_correct') is True:
                    summary['ats']['correct'] += 1

    # Compute summary percentages
    try:
        for key in ('winners','ou','ats'):
            corr = summary[key].get('correct', 0) or 0
            tot = summary[key].get('total', 0) or 0
            summary[key]['pct'] = (f"{(corr/tot*100):.1f}%" if tot > 0 else '—')
    except Exception:
        for key in ('winners','ou','ats'):
            summary[key]['pct'] = '—'

    # Sorting
    try:
        if sort_by == 'winprob_desc':
            game_cards.sort(key=lambda g: (g.get('home_win_prob') or 0.0), reverse=True)
        elif sort_by == 'ou_edge_desc':
            game_cards.sort(key=lambda g: abs(g.get('ou_edge_num') or 0.0), reverse=True)
        elif sort_by == 'ats_edge_desc':
            game_cards.sort(key=lambda g: abs(g.get('ats_edge_num') or 0.0), reverse=True)
        else:
            game_cards.sort(key=lambda g: (g.get('sort_ts') is None, g.get('sort_ts') or 0.0))
    except Exception:
        pass

    page_html = render_template_string('''
    <style>
        html, body { height:100%; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0 0 40px; }
        .container { max-width: 1100px; margin: 16px auto 0; background: #fff; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); padding: 20px 22px 28px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        form { display: flex; flex-direction: column; gap: 16px; margin-bottom: 32px; }
        /* Centered filter bar */
        .filterbar { position: static; z-index: 1; background: #fff; margin: 4px auto 12px; padding: 8px 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: row; flex-wrap: wrap; gap: 8px 14px; align-items: center; justify-content: center; max-width: 1000px; }
        label { font-weight: 500; color: #34495e; }
        select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 1em; }
        button { background: #2980b9; color: #fff; border: none; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #3498db; }

    .banner { background:#eef6ff; border:1px solid #c9e2ff; padding:8px 14px; border-radius:8px; font-size:0.93em; color:#1d4567; margin: 8px 0 14px; }
    .card { background: #f9fafb; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); padding: 12px 14px 10px; margin: 8px 0 0; border-left: 6px solid #bdc3c7; }
        .card-header { display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .status { font-weight: 700; font-size: 0.85em; padding: 4px 8px; border-radius: 10px; }
        .status.final { background:#eafaf1; color:#1e8449; }
        .status.upcoming { background:#f4f6fa; color:#7f8c8d; }
        .when { color:#34495e; font-size: 0.95em; }

        .teams { display: grid; grid-template-columns: 1fr 60px 1fr; align-items: center; gap: 10px; }
        .team { text-align: center; }
        .team-logo { height: 52px; margin-bottom: 6px; }
        .team-name { font-weight: 700; font-size: 1.05em; padding: 4px 10px; border-radius: 6px; display: inline-block; margin-top: 2px; }
        .vs { font-size: 1.8em; color: #888; font-weight: 700; }

        .score-block { margin-top: 6px; }
        .score { font-size: 1.6em; font-weight: 800; color: #2c3e50; }
        .pred { font-size: 0.95em; color: #7f8c8d; }

    .rows { display:grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin-top: 8px; }
        .row { background:#fff; border:1px solid #eaecef; border-radius:8px; padding:8px 10px; font-size:0.95em; color:#2c3e50; }
        .row b { color:#2c3e50; }
        .badges { display:flex; flex-wrap:wrap; gap:6px; }
        .badge { padding:2px 8px; border-radius:12px; font-size:0.85em; font-weight:700; }
        .ok { background:#eafaf1; color:#1e8449; }
        .err { background:#fdecea; color:#c0392b; }
        .push { background:#f4f6fa; color:#7f8c8d; }
        .muted { color:#7f8c8d; }

        .odds-toggle { margin-top: 8px; text-align:center; }
        .odds-toggle button { background:#6c5ce7; }
        .odds { margin-top:8px; }
        .odds-table { width: 100%; border-collapse: collapse; background: #fff; }
        .odds-table th, .odds-table td { padding: 8px 10px; border: 1px solid #e0e0e0; text-align: center; }
        .odds-table th { background: #eaf1fb; color: #2c3e50; }
        .odds-table tr:nth-child(even) { background: #f4f6fa; }
        .no-odds { color: #888; font-style: italic; }

        .topbar { position: sticky; top: 0; z-index: 120; display:flex; justify-content: space-between; align-items:center; margin-bottom: 10px; padding: 10px 8px; background: rgba(255,255,255,0.92); border-bottom: 1px solid #eee; backdrop-filter: saturate(180%) blur(8px); border-top-left-radius: 12px; border-top-right-radius: 12px; }
        .links a { color:#2980b9; margin-left:12px; text-decoration: underline; }
        .summary { display:flex; gap:16px; justify-content:center; color:#2c3e50; font-weight:600; margin:10px 0 16px; }

        /* Responsive grid for cards */
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
    @media (min-width: 1300px) { .grid { grid-template-columns: 1fr 1fr 1fr; } }

        .filterbar .control { display: inline-flex; align-items: center; gap: 8px; }
        .filterbar .control label { font-size: 0.95em; margin: 0; color: #34495e; }
        .filterbar select, .filterbar button, .filterbar input[type="checkbox"] { font-size: 0.95em; padding: 6px 10px; }
        .filterbar button { height: 36px; }

        /* Back to top */
        #backToTop { position: fixed; right: 16px; bottom: 16px; padding: 8px 12px; border: none; border-radius: 18px; background: #2980b9; color: #fff; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: none; }
        #backToTop:hover { background: #3498db; }
    </style>
    <div class="container">
    <div class="topbar">
            <div class="links">
                <a href="/recommendations">Recommendations</a>
    <a href="/recommendations/performance">Performance</a>
        <a href="/analysis">Analysis</a>
                <a href="/win-totals">Win Totals</a>
                <a href="/conference-records">Conference Records</a>
                <a href="/team-schedules">Team Schedules</a>
            </div>
        {% if not HIDE_REFRESH %}
        <div style="display:flex; align-items:center; gap:8px;">
            <button type="button" id="refreshBtn" title="Click = full refresh; Shift+Click = quick (scores+odds)" onclick="if(window.refreshData){try{window.refreshData();}catch(e){alert('Refresh error: '+e);}}else{fetch('/api/refresh-data',{method:'POST'}).then(()=>location.reload()).catch(e=>alert('Refresh failed: '+e));}">Refresh Data</button>
            <span id="refreshStatus" style="font-size:0.95em; color:#555;"></span>
            <small>
                <a href="/api/refresh-data" target="_blank" style="color:#2980b9; text-decoration:underline;">manual</a>
                • <a href="/api/refresh-data?mode=quick" target="_blank" style="color:#27ae60; text-decoration:underline;">fast</a>
                • <a href="/refresh-status" target="_blank" style="color:#8e44ad; text-decoration:underline;">diagnostics</a>
            </small>
        </div>
    {% endif %}
    </div> <!-- end topbar -->
    <div class="banner">
        Week {{selected_week}}: <strong>{{finals_count_week}}</strong> finals / {{total_games_week}} games ({{finals_pct_week}} complete)
    </div>
        <h2>2025 NCAA Football Predictions</h2>
        <div class="summary">
            <div class="muted" style="align-self:center;">This view</div>
            <div class="muted" style="align-self:center;"><span id="sum-count-shown">{{ game_cards|length }}</span> shown</div>
            <div>Winners: <span id="sum-winners-correct">{{summary['winners']['correct']}}</span> / <span id="sum-winners-total">{{summary['winners']['total']}}</span> (<span id="sum-winners-pct">{{summary['winners']['pct']}}</span>)</div>
            <div>ATS: <span id="sum-ats-correct">{{summary['ats']['correct']}}</span> / <span id="sum-ats-total">{{summary['ats']['total']}}</span> (<span id="sum-ats-pct">{{summary['ats']['pct']}}</span>) +<span id="sum-ats-push">{{summary['ats']['push']}}</span> push</div>
            <div>Totals: <span id="sum-ou-correct">{{summary['ou']['correct']}}</span> / <span id="sum-ou-total">{{summary['ou']['total']}}</span> (<span id="sum-ou-pct">{{summary['ou']['pct']}}</span>) +<span id="sum-ou-push">{{summary['ou']['push']}}</span> push</div>
        </div>
        <div style="text-align:center; margin:-2px 0 8px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
            <button type="button" id="toggleFinalsBtn" style="background:#8e44ad;">{{ 'Show All Games' if filter_type == 'completed' else 'Show Finals Only' }}</button>
            <button type="button" id="toggleScopeBtn" style="background:#16a085;">{{ 'Include Non-FBS' if hide_both_unknown else 'Hide Non-FBS' }}</button>
        </div>
        <div style="text-align:center; margin:-6px 0 10px;">
            <label style="font-size:0.95em;color:#34495e;"><input type="checkbox" id="toggleWxTotals" checked> Show weather-adjusted totals</label>
        </div>
    <form method="post" id="mainForm" class="filterbar">
            <div class="control">
                <label for="week">Week</label>
                <select name="week" id="week" onchange="document.getElementById('mainForm').submit();">
                    {% for w in weeks %}
                    <option value="{{w}}" {% if w == selected_week|int %}selected{% endif %}>Week {{w}}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="control">
                <label for="date">Date</label>
                <select name="date" id="date">
                    <option value="">All Dates (Local)</option>
                    {% for d in all_dates %}
                    <option value="{{d}}" {% if d == selected_date %}selected{% endif %}>{{d}}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="control">
                <label for="conference">Conference</label>
                <select name="conference" id="conference" onchange="document.getElementById('mainForm').submit();">
                    <option value="">All Conferences</option>
                    {% for conf in all_conferences %}
                    <option value="{{conf}}" {% if conf == selected_conference %}selected{% endif %}>{{conf}}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="control">
                <label for="filter_type">Show</label>
                <select name="filter_type" id="filter_type" onchange="document.getElementById('mainForm').submit();">
                    <option value="all" {% if filter_type == 'all' %}selected{% endif %}>All games</option>
                    <option value="completed" {% if filter_type == 'completed' %}selected{% endif %}>Completed games</option>
                    <option value="upcoming" {% if filter_type == 'upcoming' %}selected{% endif %}>Upcoming games</option>
                </select>
            </div>
            <div class="control">
                <label for="sort_by">Sort by</label>
                <select name="sort_by" id="sort_by">
                    <option value="time" {% if sort_by == 'time' %}selected{% endif %}>Time</option>
                    <option value="winprob_desc" {% if sort_by == 'winprob_desc' %}selected{% endif %}>Home Win Prob (desc)</option>
                    <option value="ou_edge_desc" {% if sort_by == 'ou_edge_desc' %}selected{% endif %}>O/U Edge |abs| (desc)</option>
                    <option value="ats_edge_desc" {% if sort_by == 'ats_edge_desc' %}selected{% endif %}>ATS Edge |abs| (desc)</option>
                </select>
            </div>
            <label class="control"><input type="checkbox" name="show_all" {% if show_all %}checked{% endif %} onchange="document.getElementById('mainForm').submit();"> Show all games for week</label>

            <!-- Removed obsolete hide-both-unknown checkbox (superseded by Non-FBS toggle) -->
            <button type="submit">Submit</button>
        </form>
    <div class="grid">
    {% for game_info in game_cards %}
    <div class="card" data-sort-ts="{{game_info['sort_ts'] or 0}}" data-home-win-prob="{{game_info['home_win_prob'] or 0}}" data-ou-edge="{{game_info['ou_edge_num'] or 0}}" data-ats-edge="{{game_info['ats_edge_num'] or 0}}" data-home-conf="{{game_info['home_conference']}}" data-away-conf="{{game_info['away_conference']}}" data-ats-actual="{{game_info['ats_actual_result'] or ''}}" data-ats-correct="{% if game_info['ats_correct'] is not none %}{{ 'true' if game_info['ats_correct'] else 'false' }}{% else %}{% endif %}" data-ou-actual="{{game_info['ou_actual_result'] or ''}}" data-ou-correct="{% if game_info['ou_correct'] is not none %}{{ 'true' if game_info['ou_correct'] else 'false' }}{% else %}{% endif %}" data-winner-correct="{% if game_info['correct_prediction'] is not none %}{{ 'true' if game_info['correct_prediction'] else 'false' }}{% else %}{% endif %}" style="border-left-color: {% if game_info['is_final'] %}{% if game_info['correct_prediction'] is not none %}{% if game_info['correct_prediction'] %}#2ecc71{% else %}#e74c3c{% endif %}{% else %}#95a5a6{% endif %}{% else %}#bdc3c7{% endif %};">
        <div class="card-header">
            <div class="when">Venue: {{game_info['venue']}} • <span class="local-time" data-iso="{{game_info['start_iso']}}">{{game_info['game_time']}}</span></div>
        <div class="status {% if game_info['is_final'] %}final{% else %}upcoming{% endif %}">{% if game_info['is_final'] %}FINAL{% else %}UPCOMING{% endif %}</div>
        </div>
        <div class="teams">
            <div class="team">
                <img src="{{game_info['away_logo']}}" alt="{{game_info['away_team']}} logo" class="team-logo" onerror="this.onerror=null;this.src='';"><br>
                <span class="team-name" style="background:{{game_info['away_alt_color']}};color:{% if game_info['away_alt_color'] in ['#000','#111','#222','#333','#444','#1a1a1a','#232323','#2c3e50','#34495e'] %}#fff{% else %}#222{% endif %};">{{game_info['away_team']}}</span>
                <div class="score-block">
            {% if game_info['is_final'] %}
                        <div class="score">{{game_info['actual_away_points']}}</div>
                        <div class="pred">Model: {{game_info['predicted_away_points']}}</div>
                    {% else %}
                        <div class="score">{{game_info['predicted_away_points']}}</div>
                        <div class="pred muted">Projected</div>
                    {% endif %}
                </div>
            </div>
            <div class="vs">@</div>
            <div class="team">
                <img src="{{game_info['home_logo']}}" alt="{{game_info['home_team']}} logo" class="team-logo" onerror="this.onerror=null;this.src='';"><br>
                <span class="team-name" style="background:{{game_info['home_alt_color']}};color:{% if game_info['home_alt_color'] in ['#000','#111','#222','#333','#444','#1a1a1a','#232323','#2c3e50','#34495e'] %}#fff{% else %}#222{% endif %};">{{game_info['home_team']}}</span>
                <div class="score-block">
                    {% if game_info['is_final'] %}
                        <div class="score">{{game_info['actual_home_points']}}</div>
                        <div class="pred">Model: {{game_info['predicted_home_points']}}</div>
                    {% else %}
                        <div class="score">{{game_info['predicted_home_points']}}</div>
                        <div class="pred muted">Projected</div>
                    {% endif %}
                </div>
            </div>
        </div>

        <div class="rows">
            <div class="row">
                <b>Total (model):</b>
                <span class="total-adj"> {{game_info['pred_total_adj'] or game_info['predicted_total_points']}} </span>
                <span class="total-pre" style="display:none;"> {{game_info['pred_total_pre'] or game_info['predicted_total_points']}} </span>
                {% if game_info['is_final'] %}
                    {% if game_info['actual_total_points'] is not none %}
                        <br><b>Total (actual):</b> {{game_info['actual_total_points']}}
                        {% if game_info['total_points_diff'] is not none %}
                            <br><b>Diff:</b> <span style="font-weight:700; color:{% if game_info['total_points_diff']|float > 0 %}#0b84ff{% elif game_info['total_points_diff']|float < 0 %}#ff7f0e{% else %}#2c3e50{% endif %};">{{game_info['total_points_diff']}}</span>
                        {% endif %}
                    {% endif %}
                {% endif %}
            </div>
            <div class="row">
                {% if game_info['home_win_prob_pct'] %}
                    <b>Win Prob:</b> Away {{game_info['away_win_prob_pct']}} / Home {{game_info['home_win_prob_pct']}}
                {% else %}
                    <span class="muted">Win Prob: —</span>
                {% endif %}
                {% if game_info['is_final'] and game_info['correct_prediction'] is not none %}
                    <div class="badges" style="margin-top:6px;">
                        <span class="badge {% if game_info['correct_prediction'] %}ok{% else %}err{% endif %}">Winner {% if game_info['correct_prediction'] %}Correct{% else %}Wrong{% endif %}</span>
                    </div>
                {% endif %}
            </div>
            <div class="row">
                {% if game_info['ats_line'] %}
                    <b>Spread:</b> {{game_info['ats_line']}} • <b>Model:</b> {{game_info['ats_model_lean'] or '—'}}
                    {% if game_info['ats_edge'] %}<span class="muted"> (Edge {{game_info['ats_edge']}})</span>{% endif %}
                    {% if game_info['is_final'] and game_info['ats_actual_result'] %}
                        <br><b>ATS:</b> {{game_info['ats_actual_result']}}
                        {% if game_info['ats_correct'] is not none %}
                            <span class="badge {% if game_info['ats_correct'] %}ok{% else %}err{% endif %}" style="margin-left:6px;">{% if game_info['ats_correct'] %}Correct{% else %}Wrong{% endif %}</span>
                        {% elif game_info['ats_actual_result'] == 'Push' %}
                            <span class="badge push" style="margin-left:6px;">Push</span>
                        {% endif %}
                    {% endif %}
                {% else %}
                    <span class="muted">Spread: —</span>
                {% endif %}
            </div>
            <div class="row">
                {% if game_info['ou_line'] %}
                    <b>O/U:</b> {{game_info['ou_line']}} • <b>Model:</b> {{game_info['ou_model_lean'] or '—'}}
                    {% if game_info['ou_edge'] %}<span class="muted"> (Edge {{game_info['ou_edge']}})</span>{% endif %}
                    {% if game_info['is_final'] and game_info['ou_actual_result'] %}
                        <br><b>Totals:</b> {{game_info['ou_actual_result']}}
                        {% if game_info['ou_correct'] is not none %}
                            <span class="badge {% if game_info['ou_correct'] %}ok{% else %}err{% endif %}" style="margin-left:6px;">{% if game_info['ou_correct'] %}Correct{% else %}Wrong{% endif %}</span>
                        {% elif game_info['ou_actual_result'] == 'Push' %}
                            <span class="badge push" style="margin-left:6px;">Push</span>
                        {% endif %}
                    {% endif %}
                {% else %}
                    <span class="muted">O/U: —</span>
                {% endif %}
            </div>
            <div class="row">
                {% if game_info['wx_temp_f'] or game_info['wx_wind_mph'] or game_info['wx_adjust_total'] %}
                    <b>Weather:</b>
                    {% if game_info['wx_temp_f'] %} {{game_info['wx_temp_f']}}°F{% endif %}
                    {% if game_info['wx_wind_mph'] %} • {{game_info['wx_wind_mph']}} mph wind{% endif %}
                    {% if game_info['wx_adjust_total'] %} • Δ {{game_info['wx_adjust_total']}}{% endif %}
                {% else %}
                    <span class="muted">Weather: —</span>
                {% endif %}
            </div>
        </div>

        {% if game_info['betting_lines'] and game_info['betting_lines']|length > 0 %}
        <div class="odds-toggle"><button type="button" class="toggleOddsBtn">Show Odds</button></div>
        <div class="odds" style="display:none;">
            <table class="odds-table">
                <tr><th>Provider</th><th>Spread</th><th>Over/Under</th><th>Home ML</th><th>Away ML</th></tr>
                {% for odds in game_info['betting_lines'] %}
                <tr>
                    <td>{{ odds['provider'] }}</td>
                    <td>{{ odds['formattedSpread'] or odds['spread'] }}</td>
                    <td>{{ odds['overUnder'] }}</td>
                    <td>{{ odds['homeMoneyline'] }}</td>
                    <td>{{ odds['awayMoneyline'] }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% else %}
            <div class="no-odds">No betting odds available for this game.</div>
        {% endif %}
    </div>
    {% endfor %}
    </div>
    </div>
    <button id="backToTop" title="Back to top">Top</button>
        <script>
        (function(){
            async function doRefresh(ev){
                const btn = document.getElementById('refreshBtn');
                const statusEl = document.getElementById('refreshStatus');
                if(!btn) return;
                const original = btn.textContent;
                btn.disabled = true; btn.textContent = 'Refreshing…';
                const quick = !!(ev && ev.shiftKey);
                if(statusEl) { statusEl.textContent = quick ? 'Quick refresh (scores+odds)…' : 'Running full refresh in background… this can take ~60–180s'; }
                try{
                    // Start in background and poll progress
                    const startUrl = quick ? '/api/refresh-start?mode=quick' : '/api/refresh-start';
                    // Try POST first; if 409 (already running) or non-OK, tolerate and proceed to polling.
                    let startOk = false;
                    try{
                        const startRes = await fetch(startUrl, {method:'POST'});
                        if(startRes.ok || startRes.status === 409){
                            startOk = true;
                        }else{
                            // Some local servers may block POST; attempt GET as fallback
                            const getRes = await fetch(startUrl.replace('/api/refresh-start','/api/refresh-start'), {method:'GET'});
                            if(getRes.ok || getRes.status === 409){
                                startOk = true;
                            }else{
                                const txt = await startRes.text().catch(()=> '');
                                throw new Error('Failed to start refresh: ' + (txt || startRes.status));
                            }
                        }
                    }catch(e){
                        // If network error on start, still try to poll in case it actually started
                        startOk = true;
                        if(statusEl){ statusEl.textContent = 'Starting refresh… (retrying)'; }
                    }
                    let done = false; let tries = 0;
                    while(!done && tries < 180){ // up to ~3 minutes
                        await new Promise(r=>setTimeout(r, 1000));
                        tries++;
                        let p = null;
                        try{
                            const progRes = await fetch('/api/refresh-progress', {cache:'no-store'});
                            const ct = (progRes.headers.get('content-type')||'').toLowerCase();
                            if(!progRes.ok || ct.indexOf('application/json') === -1){
                                // Service may still be waking; skip this tick
                                continue;
                            }
                            p = await progRes.json();
                        }catch(_e){
                            // Non-JSON (Render wake page) or network hiccup — keep polling
                            continue;
                        }
                        if(p){
                            if(statusEl){
                                const elapsed = p.elapsed ? ` (${p.elapsed}s)` : '';
                                statusEl.textContent = `Refreshing${elapsed}…`;
                            }
                            if(p.status && p.status !== 'running'){
                                done = true;
                                if(statusEl){
                                    const secs = p.seconds_total ? ` in ${p.seconds_total}s` : '';
                                    statusEl.textContent = `Done (${p.mode})${secs}: ${p.status}. Source=${p.pred_source}`;
                                }
                                break;
                            }
                        }
                    }
                    // If not done after polling window, open diagnostics for details
                    if(!done){
                        const diagUrl = quick ? '/refresh-status?mode=quick' : '/refresh-status';
                        window.open(diagUrl, '_blank');
                    }
                    // Refresh page when finished
                    if(done) location.reload();
                }catch(e){
                    if(statusEl){ statusEl.textContent = 'Refresh failed: ' + e; } else { alert('Refresh failed: ' + e); }
                }finally{
                    btn.disabled = false; btn.textContent = original;
                }
            }
            // Expose globally and bind click
            window.refreshData = doRefresh;
            document.addEventListener('DOMContentLoaded', function(){
                const btn = document.getElementById('refreshBtn');
                if(btn){ btn.addEventListener('click', doRefresh); }

                // Render game times in user's local timezone
                try {
                    const opts = { weekday: 'short', month: 'short', day: '2-digit', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' };
                    document.querySelectorAll('.local-time').forEach(el => {
                        let iso = (el.getAttribute('data-iso') || '').trim();
                        // Fallback: parse the visible text if no data-iso present
                        if(!iso){
                            iso = (el.textContent || '').trim();
                            if(!iso) return;
                        }
                        let s = iso;
                        if(s.indexOf('T') === -1 && s.indexOf(' ') !== -1){ s = s.replace(' ', 'T'); }
                        // If no timezone provided, assume UTC (append Z)
                        if(!/[zZ]|[+-]\d{2}:?\d{2}$/.test(s)) s = s + 'Z';
                        let d = new Date(s);
                        if(isNaN(d)){
                            // Coerce common raw form: YYYY-MM-DD HH:MM:SS+00:00 -> ISO Z
                            const m = s.match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\+00:00)?$/);
                            if(m){ s = m[1] + 'T' + m[2] + 'Z'; d = new Date(s); }
                        }
                        if(!isNaN(d)){
                            el.textContent = d.toLocaleString(undefined, opts);
                            // Normalize attribute for consistency next time
                            el.setAttribute('data-iso', s);
                        }
                    });
                } catch(e) { /* no-op */ }

                // Build Date dropdown from cards (local dates)
                try {
                    const dateSel = document.getElementById('date');
                    const hideUnknownChk = document.getElementById('hideBothUnknown');
                    const grid = document.querySelector('.grid');
                    const dOpts = new Map(); // key: yyyy-mm-dd (local), val: Label
                    const cards = Array.from(document.querySelectorAll('.grid .card'));
                    const fmt2d = n=> (n<10?('0'+n):(''+n));
                    cards.forEach(c=>{
                        const el = c.querySelector('.local-time');
                        if(!el) return;
                        let iso = (el.getAttribute('data-iso')||'').trim();
                        if(!iso){ iso = (el.textContent||'').trim(); }
                        if(!iso) return;
                        let s = iso; if(s.indexOf('T')===-1 && s.indexOf(' ')!==-1) s = s.replace(' ','T');
                        if(!/[zZ]|[+-]\d{2}:?\d{2}$/.test(s)) s += 'Z';
                        const d = new Date(s); if(isNaN(d)) return;
                        const key = `${d.getFullYear()}-${fmt2d(d.getMonth()+1)}-${fmt2d(d.getDate())}`;
                        const label = d.toLocaleDateString(undefined, { weekday:'short', month:'short', day:'2-digit', year:'numeric' });
                        dOpts.set(key, label);
                        c.setAttribute('data-local-date', key);
                    });
                    function recalcSummary(){
                        try{
                            const visCards = Array.from(document.querySelectorAll('.grid .card')).filter(c=>c.style.display !== 'none');
                            // Count shown
                            const shown = visCards.length;
                            const put = (id, val)=>{ const el=document.getElementById(id); if(el) el.textContent = String(val); };
                            put('sum-count-shown', shown);
                            // Winners
                            let wTot=0, wCor=0;
                            // ATS
                            let aTot=0, aCor=0, aPush=0;
                            // OU
                            let oTot=0, oCor=0, oPush=0;
                            visCards.forEach(c=>{
                                const w = c.getAttribute('data-winner-correct');
                                if(w==="true" || w==="false"){ wTot += 1; if(w==="true") wCor += 1; }
                                const aAct = (c.getAttribute('data-ats-actual')||'').trim();
                                const aOK = c.getAttribute('data-ats-correct');
                                if(aAct === 'Home' || aAct === 'Away'){ aTot += 1; if(aOK === 'true') aCor += 1; }
                                else if(aAct === 'Push'){ aPush += 1; }
                                const oAct = (c.getAttribute('data-ou-actual')||'').trim();
                                const oOK = c.getAttribute('data-ou-correct');
                                if(oAct === 'Over' || oAct === 'Under'){ oTot += 1; if(oOK === 'true') oCor += 1; }
                                else if(oAct === 'Push'){ oPush += 1; }
                            });
                            // Update DOM
                            put('sum-winners-correct', wCor); put('sum-winners-total', wTot);
                            put('sum-ats-correct', aCor); put('sum-ats-total', aTot); put('sum-ats-push', aPush);
                            put('sum-ou-correct', oCor); put('sum-ou-total', oTot); put('sum-ou-push', oPush);
                            const pct = (c,t)=> (t>0? ((c/t*100).toFixed(1)+'%') : '—');
                            put('sum-winners-pct', pct(wCor, wTot));
                            put('sum-ats-pct', pct(aCor, aTot));
                            put('sum-ou-pct', pct(oCor, oTot));
                        }catch(e){ /* no-op */ }
                    }

                    function applyCombinedFilters(){
                        const cardsAll = Array.from(document.querySelectorAll('.grid .card'));
                        const dateVal = (dateSel && dateSel.value) ? dateSel.value : '';
                        const hideUnknown = !!(hideUnknownChk && hideUnknownChk.checked);
                        cardsAll.forEach(c=>{
                            const k = c.getAttribute('data-local-date') || '';
                            const hc = (c.getAttribute('data-home-conf')||'').trim();
                            const ac = (c.getAttribute('data-away-conf')||'').trim();
                            const dateOk = (!dateVal || dateVal===k);
                            const confOk = (!hideUnknown || !(hc==='Unknown' && ac==='Unknown'));
                            c.style.display = (dateOk && confOk) ? '' : 'none';
                        });
                        recalcSummary();
                    }

                    if(dateSel && dOpts.size){
                        const keep = dateSel.value; // server-provided selection (UTC-based)
                        // Clear and rebuild options
                        dateSel.innerHTML = '';
                        const optAll = document.createElement('option');
                        optAll.value = ''; optAll.textContent = 'All Dates (Local)';
                        dateSel.appendChild(optAll);
                        Array.from(dOpts.keys()).sort().forEach(k=>{
                            const o = document.createElement('option');
                            o.value = k; o.textContent = dOpts.get(k);
                            dateSel.appendChild(o);
                        });
                        // If a previous local key was stored, select it
                        const stored = sessionStorage.getItem('selectedLocalDate') || '';
                        if(stored && dOpts.has(stored)) dateSel.value = stored; else dateSel.value = '';
                        // Apply initial filters and bind events
                        dateSel.addEventListener('change', (ev)=>{
                            ev.preventDefault();
                            sessionStorage.setItem('selectedLocalDate', dateSel.value || '');
                            applyCombinedFilters();
                        });
                        if(hideUnknownChk){ hideUnknownChk.addEventListener('change', (ev)=>{ ev.preventDefault(); applyCombinedFilters(); }); }
                        applyCombinedFilters();
                    }
                } catch(e) { /* no-op */ }

                // Client-side sorting
                const sortSelect = document.getElementById('sort_by');
                const grid = document.querySelector('.grid');
                function getVal(card, mode){
                    const ts = parseFloat(card.getAttribute('data-sort-ts') || '0');
                    const p = parseFloat(card.getAttribute('data-home-win-prob') || '0');
                    const ou = Math.abs(parseFloat(card.getAttribute('data-ou-edge') || '0'));
                    const ats = Math.abs(parseFloat(card.getAttribute('data-ats-edge') || '0'));
                    if(mode==='winprob_desc') return isNaN(p)?0:p;
                    if(mode==='ou_edge_desc') return isNaN(ou)?0:ou;
                    if(mode==='ats_edge_desc') return isNaN(ats)?0:ats;
                    // default time asc; use large number when missing to push to bottom
                    return isNaN(ts)? Number.MAX_SAFE_INTEGER : ts;
                }
                function resort(mode){
                    if(!grid) return;
                    const cards = Array.from(grid.children).filter(el => el.classList.contains('card'));
                    if(!cards.length) return;
                    cards.sort((a,b)=>{
                        const va = getVal(a, mode);
                        const vb = getVal(b, mode);
                        if(mode==='winprob_desc' || mode==='ou_edge_desc' || mode==='ats_edge_desc'){
                            return (vb - va);
                        } else {
                            return (va - vb);
                        }
                    });
                    cards.forEach(c => grid.appendChild(c));
                }
                if(sortSelect){
                    sortSelect.addEventListener('change', function(ev){
                        // prevent form submission; client-side sort only
                        ev.preventDefault();
                        resort(sortSelect.value || 'time');
                    });
                    // initial align to current selection
                    resort(sortSelect.value || 'time');
                }

                // If only one card is rendered after server filters, ensure it's not due to DOM filtering
                try {
                    const gridCards = Array.from(document.querySelectorAll('.grid .card'));
                    if (gridCards.length <= 1) {
                        // No-op; optionally could display a hint
                        const summary = document.querySelector('.summary');
                        if(summary){
                            const hint = document.createElement('div');
                            hint.className = 'muted';
                            hint.textContent = '(Only one game matched the current filters)';
                            summary.appendChild(hint);
                        }
                    }
                } catch(_) {}

                // Toggle model vs adjusted totals
                const chk = document.getElementById('toggleWxTotals');
                function applyToggle(){
                    const showAdj = chk && chk.checked;
                    document.querySelectorAll('.total-adj').forEach(el => el.style.display = showAdj ? '' : 'none');
                    document.querySelectorAll('.total-pre').forEach(el => el.style.display = showAdj ? 'none' : '');
                }
                if(chk){ chk.addEventListener('change', applyToggle); applyToggle(); }

                // Date filtering is server-driven via form submit on change (keeps initial payload light)

                // Toggle per-card odds table
                document.querySelectorAll('.toggleOddsBtn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const card = btn.closest('.card');
                        if(!card) return;
                        const odds = card.querySelector('.odds');
                        if(!odds) return;
                        const isHidden = window.getComputedStyle(odds).display === 'none';
                        if(isHidden){
                            odds.style.display = '';
                            btn.textContent = 'Hide Odds';
                        } else {
                            odds.style.display = 'none';
                            btn.textContent = 'Show Odds';
                        }
                    });
                });

                // Back to top behavior
                const topBtn = document.getElementById('backToTop');
                function toggleTop(){
                    if(window.scrollY > 300){ topBtn.style.display = 'block'; } else { topBtn.style.display = 'none'; }
                }
                window.addEventListener('scroll', toggleTop);
                toggleTop();
                topBtn.addEventListener('click', function(){ window.scrollTo({ top: 0, behavior: 'smooth' }); });

                // Finals-only toggle (round-trips with query params)
                try {
                    const finalsBtn = document.getElementById('toggleFinalsBtn');
                    if(finalsBtn){
                        finalsBtn.addEventListener('click', ()=>{
                            const url = new URL(window.location.href);
                            if(url.searchParams.get('filter_type') === 'completed'){
                                url.searchParams.delete('filter_type');
                            } else {
                                url.searchParams.set('filter_type','completed');
                            }
                            if(!url.searchParams.get('week')){ url.searchParams.set('week','{{ selected_week }}'); }
                            window.location.href = url.toString();
                        });
                    }
                    const scopeBtn = document.getElementById('toggleScopeBtn');
                    if(scopeBtn){
                        scopeBtn.addEventListener('click', ()=>{
                            const url = new URL(window.location.href);
                            const inc = url.searchParams.get('include_non_fbs');
                            if(inc === '1'){
                                url.searchParams.delete('include_non_fbs'); // revert to hiding
                            } else {
                                url.searchParams.set('include_non_fbs','1');
                                // When including non-FBS, keep existing hide_both_unknown if user forced earlier
                            }
                            if(!url.searchParams.get('week')){ url.searchParams.set('week','{{ selected_week }}'); }
                            window.location.href = url.toString();
                        });
                    }
                } catch(e) { /* no-op */ }
            });
        })();
        </script>
    <div style="margin-top:30px; text-align:center; font-size:0.75em; color:#7f8c8d;">
        Build {{ BUILD_TIME }} • Commit {{ BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown' }} • Source {{ PRED_SOURCE }}
        • <a href="/version" style="color:#2980b9;">version JSON</a>
    </div>
    ''', weeks=weeks, selected_week=selected_week, all_dates=all_dates, selected_date=selected_date, show_all=show_all, hide_both_unknown=hide_both_unknown, all_conferences=pred_df['home_conference'].unique(), selected_conference=selected_conference, game_cards=game_cards, filter_type=filter_type, summary=summary, sort_by=sort_by, HIDE_REFRESH=HIDE_REFRESH, finals_count_week=finals_count_week, total_games_week=total_games_week, finals_pct_week=finals_pct_week, unknown_pending=unknown_pending, BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT, PRED_SOURCE=PRED_SOURCE)
    resp = make_response(page_html)
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# New route: Projected Conference Records for 2025
@app.route('/conference-records')
def conference_records():
    # Only use 2025 season games
    conf_games = pred_df[pred_df['season'] == 2025].copy()
    # Use merged conference info, skip non-conference games
    conf_games = conf_games[(conf_games['home_conference'] == conf_games['away_conference'])]
    conf_col = 'home_conference'
    # Determine winner for each game
    def get_winner(row):
        try:
            home_pts = float(row.get('predicted_home_points', 0))
            away_pts = float(row.get('predicted_away_points', 0))
            if home_pts > away_pts:
                return row['home_team']
            elif away_pts > home_pts:
                return row['away_team']
            else:
                return 'TIE'
        except Exception:
            return None
    conf_games['winner'] = conf_games.apply(get_winner, axis=1)
    # Aggregate records
    records = {}
    for _, row in conf_games.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']
        conference = row[conf_col] if conf_col else 'Unknown'
        for team in [home, away]:
            if team not in records:
                records[team] = {'conference': conference, 'W': 0, 'L': 0, 'T': 0}
        if winner == 'TIE':
            records[home]['T'] += 1
            records[away]['T'] += 1
        elif winner == home:
            records[home]['W'] += 1
            records[away]['L'] += 1
        elif winner == away:
            records[away]['W'] += 1
            records[home]['L'] += 1
    # Convert to DataFrame for conference records
    conf_rec_df = pd.DataFrame([
        {'Team': team, 'Conference': rec['conference'], 'Conf_Wins': rec['W'], 'Conf_Losses': rec['L'], 'Conf_Ties': rec['T']}
        for team, rec in records.items()
    ])
    conf_rec_df = conf_rec_df.sort_values(['Conference', 'Conf_Wins', 'Conf_Losses'], ascending=[True, False, True])

    # Calculate overall records for all teams (all games, not just conference)
    overall_records = {}
    all_games = pred_df[pred_df['season'] == 2025]
    for _, row in all_games.iterrows():
        home = row['home_team']
        away = row['away_team']
        try:
            home_pts = float(row.get('predicted_home_points', 0))
            away_pts = float(row.get('predicted_away_points', 0))
        except Exception:
            home_pts = away_pts = 0
        for team in [home, away]:
            if team not in overall_records:
                overall_records[team] = {'W': 0, 'L': 0, 'T': 0}
        if home_pts == away_pts:
            overall_records[home]['T'] += 1
            overall_records[away]['T'] += 1
        elif home_pts > away_pts:
            overall_records[home]['W'] += 1
            overall_records[away]['L'] += 1
        elif away_pts > home_pts:
            overall_records[away]['W'] += 1
            overall_records[home]['L'] += 1

    # Merge overall records into conference records
    conf_rec_df['Overall_Wins'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('W', 0))
    conf_rec_df['Overall_Losses'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('L', 0))
    conf_rec_df['Overall_Ties'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('T', 0))

    # Load conference logos
    conf_logo_df = pd.read_csv(os.path.join(DATA_DIR, "conference_logos.csv"))
    conf_logo_map = dict(zip(conf_logo_df['conference'], conf_logo_df['logo_url']))
    # Group by conference for cards
    conferences = conf_rec_df['Conference'].unique()
    conf_groups = {conf: conf_rec_df[conf_rec_df['Conference'] == conf] for conf in conferences}
    conf_logos = {conf: conf_logo_map.get(conf, '') for conf in conferences}
    # Load team assets for colors
    assets_df = pd.read_csv(os.path.join(DATA_DIR, "team_assets.csv"))
    team_color_map = dict(zip(assets_df['school'], assets_df['color']))
    team_alt_color_map = dict(zip(assets_df['school'], assets_df.get('alt_color', ['']*len(assets_df))))
    # Render as cards per conference
    return render_template_string('''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        .conf-card { background: #f8f8f8; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.07); padding: 24px; margin-bottom: 28px; }
        .conf-title { font-size: 1.3em; font-weight: bold; color: #2980b9; margin-bottom: 12px; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; background: #fff; }
        th, td { padding: 8px 10px; border: 1px solid #e0e0e0; text-align: center; }
        th { background: #eaf1fb; color: #2c3e50; }
        tr:nth-child(even) { background: #f4f6fa; }
    </style>
    <div class="container">
        <div style="text-align:right; margin: 12px 0 0 0;">
            <a href="/" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">&#8592; Back to Main Predictions</a>
        </div>
        <h2>2025 Projected Conference Records</h2>
        {% for conf, group in conf_groups.items() %}
        <div class="conf-card">
            <div class="conf-title">
                {% if conf_logos[conf] %}
                    <img src="{{ conf_logos[conf] }}" alt="{{ conf }} logo" style="height:38px;vertical-align:middle;margin-right:10px;">
                {% endif %}
                {{ conf }}
            </div>
            <table>
                <tr>
                    <th>Team</th>
                    <th>Conf W</th><th>Conf L</th><th>Conf T</th>
                    <th>Overall W</th><th>Overall L</th><th>Overall T</th>
                </tr>
                {% for _, row in group.iterrows() %}
                <tr>
                    <td>
                        <span style="color:{{ team_color_map.get(row['Team'], '#2c3e50') }};background:{{ team_alt_color_map.get(row['Team'], '#eaf1fb') }};padding:3px 10px;border-radius:6px;display:inline-block;">{{ row['Team'] }}</span>
                    </td>
                    <td>{{ row['Conf_Wins'] }}</td>
                    <td>{{ row['Conf_Losses'] }}</td>
                    <td>{{ row['Conf_Ties'] }}</td>
                    <td>{{ row['Overall_Wins'] }}</td>
                    <td>{{ row['Overall_Losses'] }}</td>
                    <td>{{ row['Overall_Ties'] }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}
    </div>
    ''', conf_groups=conf_groups, conf_logos=conf_logos, team_color_map=team_color_map, team_alt_color_map=team_alt_color_map)

# New route: Team Schedules
@app.route('/team-schedules', methods=['GET', 'POST'])
def team_schedules():
    # Get all conferences and teams
    all_teams_df = pred_df[pred_df['season'] == 2025].copy()
    all_conferences = sorted(set(list(all_teams_df['home_conference'].dropna()) + list(all_teams_df['away_conference'].dropna())))
    all_conferences = [conf for conf in all_conferences if conf != 'Unknown']
    
    selected_conference = request.form.get('conference', '')
    selected_team = request.form.get('team', '')
    
    available_teams = []
    team_schedule = None
    team_info = None
    if selected_conference:
        conf_teams = set()
        conf_games = all_teams_df[(all_teams_df['home_conference'] == selected_conference) | (all_teams_df['away_conference'] == selected_conference)]
        for _, row in conf_games.iterrows():
            if row['home_conference'] == selected_conference:
                conf_teams.add(row['home_team'])
            if row['away_conference'] == selected_conference:
                conf_teams.add(row['away_team'])
        available_teams = sorted(list(conf_teams))
    if selected_team:
        team_games = all_teams_df[(all_teams_df['home_team'] == selected_team) | (all_teams_df['away_team'] == selected_team)].copy()
        team_games = team_games.sort_values('week')
        schedule_data = []
        team_record = {'W': 0, 'L': 0, 'T': 0}
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == selected_team
            opponent = game['away_team'] if is_home else game['home_team']
            opp_asset = get_team_asset(opponent)
            try:
                home_pts = float(game.get('predicted_home_points', 0))
                away_pts = float(game.get('predicted_away_points', 0))
                team_pts = home_pts if is_home else away_pts
                opp_pts = away_pts if is_home else home_pts
                if team_pts > opp_pts:
                    result = 'W'
                    team_record['W'] += 1
                elif opp_pts > team_pts:
                    result = 'L'
                    team_record['L'] += 1
                else:
                    result = 'T'
                    team_record['T'] += 1
            except:
                result = '-'
                team_pts = opp_pts = 0
            game_date = game.get('start_date', '')[:10] if game.get('start_date') else ''
            schedule_data.append({
                'week': int(game['week']),
                'date': game_date,
                'opponent': opponent,
                'is_home': is_home,
                'venue': game.get('venue', ''),
                'team_pts': f"{team_pts:.1f}" if team_pts else '',
                'opp_pts': f"{opp_pts:.1f}" if opp_pts else '',
                'result': result,
                'opp_logo': opp_asset['logo'],
                'opp_color': opp_asset['color'],
                'opp_alt_color': opp_asset['alt_color']
            })
        team_schedule = schedule_data
        team_asset = get_team_asset(selected_team)
        team_info = {
            'name': selected_team,
            'logo': team_asset['logo'],
            'color': team_asset['color'],
            'alt_color': team_asset['alt_color'],
            'record': team_record
        }

    return render_template_string('''
    <style>
        .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        form { display: flex; flex-direction: column; gap: 16px; margin-bottom: 32px; }
        label { font-weight: 500; color: #34495e; }
        select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 1em; }
        button { background: #2980b9; color: #fff; border: none; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #3498db; }
        .team-header { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 24px; padding: 20px; background: #f8f8f8; border-radius: 10px; }
        .team-logo { height: 80px; }
        .team-name { font-size: 1.8em; font-weight: bold; padding: 8px 16px; border-radius: 8px; display: inline-block; }
        .record { font-size: 1.2em; color: #2c3e50; margin-top: 8px; }
        .schedule-table { width: 100%; border-collapse: collapse; margin-top: 16px; background: #fff; }
        .schedule-table th, .schedule-table td { padding: 12px 8px; border: 1px solid #e0e0e0; text-align: center; }
        .schedule-table th { background: #eaf1fb; color: #2c3e50; }
        .schedule-table tr:nth-child(even) { background: #f4f6fa; }
        .opponent { display: flex; align-items: center; justify-content: center; gap: 8px; }
        .opp-logo { height: 30px; }
        .opp-name { padding: 2px 8px; border-radius: 4px; font-weight: 500; }
        .result-w { color: #27ae60; font-weight: bold; }
        .result-l { color: #e74c3c; font-weight: bold; }
        .result-t { color: #f39c12; font-weight: bold; }
        .home-indicator { color: #2980b9; font-weight: bold; }
        .away-indicator { color: #7f8c8d; }
    </style>
    <div class="container">
        <div style="text-align:right; margin: 12px 0 0 0;">
            <a href="/" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">&#8592; Back to Main Predictions</a>
            <a href="/conference-records" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">View Conference Records</a>
        </div>
        <h2>2025 Team Schedules</h2>
        <form method="post" id="scheduleForm">
            <label for="conference">Select Conference:</label>
            <select name="conference" id="conference" onchange="document.getElementById('scheduleForm').submit();">
                <option value="">Choose a Conference</option>
                {% for conf in all_conferences %}
                <option value="{{conf}}" {% if conf == selected_conference %}selected{% endif %}>{{conf}}</option>
                {% endfor %}
            </select>
            {% if available_teams %}
            <label for="team">Select Team:</label>
            <select name="team" id="team" onchange="document.getElementById('scheduleForm').submit();">
                <option value="">Choose a Team</option>
                {% for team in available_teams %}
                <option value="{{team}}" {% if team == selected_team %}selected{% endif %}>{{team}}</option>
                {% endfor %}
            </select>
            {% endif %}
        </form>
        {% if team_info and team_schedule %}
        <div class="team-header">
            <img src="{{team_info['logo']}}" alt="{{team_info['name']}} logo" class="team-logo">
            <div>
                <div class="team-name" style="color:{{team_info['color']}};background:{{team_info['alt_color']}};">{{team_info['name']}}</div>
                <div class="record">Projected Record: {{team_info['record']['W']}}-{{team_info['record']['L']}}{% if team_info['record']['T'] > 0 %}-{{team_info['record']['T']}}{% endif %}</div>
            </div>
        </div>
        <table class="schedule-table">
            <tr>
                <th>Week</th>
                <th>Date</th>
                <th>Opponent</th>
                <th>Location</th>
                <th>Venue</th>
                <th>Projected Score</th>
                <th>Result</th>
            </tr>
            {% for game in team_schedule %}
            <tr>
                <td>{{game['week']}}</td>
                <td>{{game['date']}}</td>
                <td>
                    <div class="opponent">
                        <img src="{{game['opp_logo']}}" alt="{{game['opponent']}} logo" class="opp-logo">
                        <span class="opp-name" style="color:{{game['opp_color']}};background:{{game['opp_alt_color']}};">{{game['opponent']}}</span>
                    </div>
                </td>
                <td>
                    {% if game['is_home'] %}
                        <span class="home-indicator">HOME</span>
                    {% else %}
                        <span class="away-indicator">@ AWAY</span>
                    {% endif %}
                </td>
                <td>{{game['venue']}}</td>
                <td>
                    {% if game['team_pts'] and game['opp_pts'] %}
                        {{game['team_pts']}} - {{game['opp_pts']}}
                    {% else %}
                        -
                    {% endif %}
                </td>
                <td>
                    {% if game['result'] == 'W' %}
                        <span class="result-w">W</span>
                    {% elif game['result'] == 'L' %}
                        <span class="result-l">L</span>
                    {% elif game['result'] == 'T' %}
                        <span class="result-t">T</span>
                    {% else %}
                        -
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    ''', all_conferences=all_conferences, selected_conference=selected_conference, available_teams=available_teams, selected_team=selected_team, team_schedule=team_schedule, team_info=team_info)

# -------------------- Lightweight Diagnostics --------------------
@app.route('/version')
def version_info():
    try:
        wk = None
        finals = None
        total = None
        if 'week' in pred_df.columns:
            try:
                wk = int(pd.to_numeric(pred_df['week'], errors='coerce').dropna().max())
            except Exception:
                wk = None
        if wk is not None:
            sub = pred_df[pred_df['week'] == wk]
            if not sub.empty and 'actual_home_points' in sub.columns:
                finals = int(((~sub['actual_home_points'].isna()) & (~sub['actual_away_points'].isna())).sum())
                total = int(len(sub))
        # Ensure model artifacts/meta loaded for current prefix
        if not _MODEL_META:
            _load_model_artifacts()
        return jsonify({
            'build_time': BUILD_TIME,
            'commit': BUILD_COMMIT,
            'prediction_source': PRED_SOURCE,
            'latest_week': wk,
            'latest_week_finals': finals,
            'latest_week_total': total,
            'model_version': _MODEL_META.get('model_version'),
            'model_counts': _MODEL_META.get('counts'),
            'model_manifest_prefix': globals().get('_MODEL_PREFIX')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Model Artifacts Loading & Inference --------------------
_MODEL_ARTIFACTS = {}
_MODEL_META = {}
MODELS_DIR = Path(os.path.join(BASE_DIR, 'models'))
_MODEL_MANIFEST_PATH = MODELS_DIR / 'model_manifest.json'
_MODEL_PREFIX = 'rf_v1'  # default until manifest loaded

def _load_model_artifacts():
    global _MODEL_ARTIFACTS, _MODEL_META
    if not MODELS_DIR.exists():
        return
    # Resolve model prefix from manifest if present
    global _MODEL_PREFIX
    try:
        if _MODEL_MANIFEST_PATH.exists():
            import json as _json
            manifest = _json.loads(_MODEL_MANIFEST_PATH.read_text(encoding='utf-8'))
            mp = manifest.get('model_prefix')
            if isinstance(mp, str) and mp:
                _MODEL_PREFIX = mp
    except Exception as e:
        print(f"[model-load] manifest read error: {e}")
    # Metrics JSON defines version & metrics for chosen prefix
    try:
        import json as _json
        metrics_file = MODELS_DIR / f'{_MODEL_PREFIX}_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                _MODEL_META = _json.load(f)
        else:
            _MODEL_META = {}
        _MODEL_META['model_version'] = _MODEL_PREFIX
    except Exception as e:
        _MODEL_META = {'model_version': _MODEL_PREFIX, 'load_error': str(e)}
    # Load joblib models for dynamic prefix
    for tgt in ['margin','home_pts','away_pts']:
        jf = MODELS_DIR / f'{_MODEL_PREFIX}_{tgt}.joblib'
        if jf.exists():
            try:
                _MODEL_ARTIFACTS[tgt] = joblib.load(jf)
            except Exception as e:
                print(f"[model-load] failed {tgt}: {e}")
    # Load calibration table if present (prefix-specific)
    calib = MODELS_DIR / f'{_MODEL_PREFIX}_home_win_calibration.csv'
    if calib.exists():
        try:
            import pandas as _pd
            _MODEL_ARTIFACTS['calibration'] = _pd.read_csv(calib)
        except Exception as e:
            print(f"[model-load] calib read error: {e}")

def _predict_proba_from_calibration(margin_pred: float):
    try:
        calib = _MODEL_ARTIFACTS.get('calibration')
        if calib is None or calib.empty:
            return None
        # Simple linear interpolation between nearest points
        import numpy as _np
        xs = calib['pred_margin'].values
        ys = calib['home_win_prob'].values
        if margin_pred <= xs.min():
            return float(ys[xs.argmin()])
        if margin_pred >= xs.max():
            return float(ys[xs.argmax()])
        idx = xs.searchsorted(margin_pred)
        x0,x1 = xs[idx-1], xs[idx]
        y0,y1 = ys[idx-1], ys[idx]
        if x1 == x0:
            return float(y0)
        return float(y0 + (y1-y0)*((margin_pred-x0)/(x1-x0)))
    except Exception:
        return None

def _overlay_model_predictions(df):
    if not _MODEL_ARTIFACTS:
        _load_model_artifacts()
    if not _MODEL_ARTIFACTS:
        return
    # Feature columns used during training
    feat_cols = ['predicted_home_points','predicted_away_points','predicted_total_points','weather_temp','weather_wind','weather_adjustment','edge','confidence']
    # Ensure prerequisite prediction-derived cols
    try:
        if 'predicted_total_points' not in df.columns and {'predicted_home_points','predicted_away_points'}.issubset(df.columns):
            df['predicted_total_points'] = df['predicted_home_points'] + df['predicted_away_points']
        if 'edge' not in df.columns and {'predicted_home_points','predicted_away_points'}.issubset(df.columns):
            df['edge'] = (df['predicted_home_points'] - df['predicted_away_points']).abs()
        if 'confidence' not in df.columns:
            # Default neutral value used during training bootstrap
            df['confidence'] = 0.5
        for wcol in ['weather_temp','weather_wind','weather_adjustment']:
            if wcol not in df.columns:
                df[wcol] = 0.0
    except Exception:
        pass
    # Now all required features should exist
    baseX = df[feat_cols].fillna(0.0) if all(c in df.columns for c in feat_cols) else df[[c for c in feat_cols if c in df.columns]].fillna(0.0)
    # Predict new points if per-target models exist
    home_model = _MODEL_ARTIFACTS.get('home_pts')
    away_model = _MODEL_ARTIFACTS.get('away_pts')
    if home_model and away_model:
        try:
            df['model_home_points'] = home_model['model'].predict(baseX[[c for c in home_model['features'] if c in baseX.columns]])
            df['model_away_points'] = away_model['model'].predict(baseX[[c for c in away_model['features'] if c in baseX.columns]])
            df['model_total_points'] = df['model_home_points'] + df['model_away_points']
        except Exception as e:
            print(f"[model-overlay] point preds failed: {e}")
    margin_model = _MODEL_ARTIFACTS.get('margin')
    if margin_model:
        try:
            df['model_margin'] = margin_model['model'].predict(baseX[[c for c in margin_model['features'] if c in baseX.columns]])
        except Exception as e:
            print(f"[model-overlay] margin pred failed: {e}")
    # Calibrated win probability
    try:
        if 'model_margin' in df.columns:
            df['model_home_win_prob'] = df['model_margin'].apply(_predict_proba_from_calibration)
    except Exception:
        pass
    # Derive edge/confidence style metrics from model outputs (non-destructive: write to new columns)
    try:
        if 'model_home_points' in df.columns and 'model_away_points' in df.columns:
            df['model_edge'] = (df['model_home_points'] - df['model_away_points']).abs()
        if 'model_home_win_prob' in df.columns:
            # Confidence score centered around coin flip
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
            # Backfill legacy columns if missing or null (do not overwrite existing populated values)
            if 'edge' in df.columns:
                try:
                    mask = df['edge'].isna()
                    if mask.any() and 'model_edge' in df.columns:
                        df.loc[mask, 'edge'] = df.loc[mask, 'model_edge']
                except Exception:
                    pass
            else:
                if 'model_edge' in df.columns:
                    df['edge'] = df['model_edge']
            if 'confidence' in df.columns:
                try:
                    mask = df['confidence'].isna()
                    if mask.any():
                        df.loc[mask, 'confidence'] = df.loc[mask, 'model_confidence_tier']
                except Exception:
                    pass
            else:
                df['confidence'] = df.get('model_confidence_tier')
    except Exception as e:
        print(f"[model-overlay] edge/confidence derivation failed: {e}")

@app.route('/api/data-health')
def api_data_health():
    """Report completeness of required feature & model columns."""
    try:
        required_feats = ['predicted_home_points','predicted_away_points','predicted_total_points','weather_temp','weather_wind','weather_adjustment','edge','confidence']
        model_cols = ['model_home_points','model_away_points','model_total_points','model_margin','model_home_win_prob']
        summary = {}
        total_rows = int(len(pred_df))
        for col in required_feats:
            if col in pred_df.columns:
                missing = int(pred_df[col].isna().sum())
                summary[col] = {'present': True, 'missing': missing, 'pct_missing': round((missing/total_rows*100.0) if total_rows else 0.0, 2)}
            else:
                summary[col] = {'present': False, 'missing': total_rows, 'pct_missing': 100.0}
        for col in model_cols:
            if col in pred_df.columns:
                missing = int(pred_df[col].isna().sum())
                summary[col] = {'present': True, 'missing': missing, 'pct_missing': round((missing/total_rows*100.0) if total_rows else 0.0, 2)}
            else:
                summary[col] = {'present': False, 'missing': total_rows, 'pct_missing': 100.0}
        # Sample problematic rows (limit 10)
        prob_mask = None
        for col in ['weather_temp','weather_wind','edge','confidence']:
            if col in pred_df.columns:
                m = pred_df[col].isna()
                prob_mask = m if prob_mask is None else (prob_mask | m)
        sample_rows = []
        if prob_mask is not None:
            for _, r in pred_df[prob_mask].head(10).iterrows():
                sample_rows.append({
                    'home_team': r.get('home_team'),
                    'away_team': r.get('away_team'),
                    'week': r.get('week'),
                    'weather_temp': r.get('weather_temp'),
                    'weather_wind': r.get('weather_wind'),
                    'edge': r.get('edge'),
                    'confidence': r.get('confidence')
                })
        return jsonify({'rows': total_rows, 'columns': summary, 'sample_incomplete': sample_rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-metrics')
def api_model_metrics():
    if not _MODEL_META:
        _load_model_artifacts()
    if not _MODEL_META:
        return {'status':'no_model'}, 404
    meta = {k:v for k,v in _MODEL_META.items() if k not in ('home_pts_model','away_pts_model','margin_model')}
    return meta

@app.route('/api/week-status')
def api_week_status():
    try:
        week_arg = request.args.get('week')
        if week_arg is None:
            week = int(pd.to_numeric(pred_df['week'], errors='coerce').dropna().max()) if 'week' in pred_df.columns else None
        else:
            week = int(week_arg)
        if week is None:
            return jsonify({'error': 'no week data'}), 400
        sub = pred_df[pred_df['week'] == week]
        if sub.empty:
            return jsonify({'week': week, 'games': 0, 'finals': 0, 'pct_complete': 0.0})
        finals = 0
        if 'actual_home_points' in sub.columns:
            finals = int(((~sub['actual_home_points'].isna()) & (~sub['actual_away_points'].isna())).sum())
        pct = (float(finals)/float(len(sub))*100.0) if len(sub) else 0.0
        sample = []
        for _, r in sub.head(5).iterrows():
            sample.append({
                'home': r.get('home_team'),
                'away': r.get('away_team'),
                'ah': r.get('actual_home_points'),
                'aa': r.get('actual_away_points')
            })
        return jsonify({'week': week, 'games': int(len(sub)), 'finals': finals, 'pct_complete': round(pct,2), 'sample': sample})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/game')
def api_debug_game():
    """Return raw prediction/actual fields for a specific game (diagnose FINAL vs UPCOMING)."""
    home = request.args.get('home','')
    away = request.args.get('away','')
    week = request.args.get('week')
    season = request.args.get('season','2025')
    try:
        week_i = int(week) if week is not None else None
    except Exception:
        week_i = None
    try:
        season_i = int(season)
    except Exception:
        season_i = 2025
    df = pred_df
    q = df[(df['season'] == season_i)]
    if week_i is not None:
        q = q[q['week'] == week_i]
    if home:
        q = q[q['home_team'].str.lower() == home.lower()]
    if away:
        q = q[q['away_team'].str.lower() == away.lower()]
    out = []
    for _, r in q.iterrows():
        out.append({
            'season': int(r.get('season',0)) if pd.notna(r.get('season',None)) else None,
            'week': int(r.get('week',0)) if pd.notna(r.get('week',None)) else None,
            'home_team': r.get('home_team'),
            'away_team': r.get('away_team'),
            'actual_home_points': r.get('actual_home_points'),
            'actual_away_points': r.get('actual_away_points'),
            'predicted_home_points': r.get('predicted_home_points'),
            'predicted_away_points': r.get('predicted_away_points'),
            'start_date': r.get('start_date'),
            'is_final_calc': (pd.notna(r.get('actual_home_points')) and pd.notna(r.get('actual_away_points'))),
        })
    return jsonify({'count': len(out), 'games': out})

@app.route('/api/_routes')
def api_list_routes():  # simple debug list of routes
    try:
        from flask import url_for
        routes = []
        for rule in app.url_map.iter_rules():
            methods = sorted([m for m in rule.methods if m not in ('HEAD','OPTIONS')])
            routes.append({'rule': str(rule), 'endpoint': rule.endpoint, 'methods': methods})
        routes = sorted(routes, key=lambda r: r['rule'])
        return jsonify({'count': len(routes), 'routes': routes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/env-info')
def api_env_info():
    try:
        import platform, hashlib
        pid = os.getpid()
        cwd = os.getcwd()
        app_file = __file__
        data_file = pred_path_scores if os.path.exists(pred_path_scores) else pred_path_enh
        df_stats = None
        finals = 0
        total = 0
        week2_finals = None
        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            try:
                total = int(len(pred_df))
                if 'actual_home_points' in pred_df.columns:
                    finals = int(((pred_df['actual_home_points'].notna()) & (pred_df['actual_away_points'].notna())).sum())
                w2 = pred_df[pred_df.get('week') == 2]
                if not w2.empty and 'actual_home_points' in w2.columns:
                    week2_finals = int(((w2['actual_home_points'].notna()) & (w2['actual_away_points'].notna())).sum())
            except Exception:
                pass
        file_hash = None
        try:
            if data_file and os.path.exists(data_file):
                with open(data_file,'rb') as f:
                    file_hash = hashlib.md5(f.read(8192)).hexdigest()
        except Exception:
            pass
        return jsonify({
            'pid': pid,
            'platform': platform.platform(),
            'cwd': cwd,
            'app_file': app_file,
            'prediction_source': PRED_SOURCE,
            'data_file_used': data_file,
            'data_file_hash_head': file_hash,
            'rows_loaded': total,
            'finals_total': finals,
            'week2_finals': week2_finals,
            'build_commit': BUILD_COMMIT,
            'build_time': BUILD_TIME,
            'route_count': len(list(app.url_map.iter_rules()))
        })
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/analysis-2025', methods=['GET'])
def analysis_2025():
    # Analyze completed 2025 games: prediction vs actuals. Optional ?week= filter.
    week_param = request.args.get('week')
    df = pred_df[(pred_df['season'] == 2025) & pred_df['actual_home_points'].notna() & pred_df['actual_away_points'].notna()].copy()
    if week_param:
        try:
            df = df[df['week'] == int(week_param)]
        except Exception:
            pass
    if df.empty:
        msg = 'No completed 2025 games found yet.' if not week_param else f'No completed 2025 games found for week {week_param}.'
        return {'message': msg, 'count': 0}, 200

    # Compute metrics
    df['pred_home'] = df['predicted_home_points']
    df['pred_away'] = df['predicted_away_points']
    df['pred_total'] = df['pred_home'] + df['pred_away']
    df['act_total'] = df['actual_home_points'] + df['actual_away_points']
    df['pred_margin'] = df['pred_home'] - df['pred_away'] if 'predicted_win_margin' not in df.columns else df['predicted_win_margin'].fillna(df['pred_home'] - df['pred_away'])
    df['act_margin'] = df['actual_home_points'] - df['actual_away_points']

    total_mae = float((df['act_total'] - df['pred_total']).abs().mean())
    margin_mae = float((df['act_margin'] - df['pred_margin']).abs().mean())

    # Winner accuracy
    df['pred_winner'] = df.apply(lambda r: r['home_team'] if r['pred_home'] > r['pred_away'] else (r['away_team'] if r['pred_away'] > r['pred_home'] else 'TIE'), axis=1)
    df['act_winner'] = df.apply(lambda r: r['home_team'] if r['actual_home_points'] > r['actual_away_points'] else (r['away_team'] if r['actual_away_points'] > r['actual_home_points'] else 'TIE'), axis=1)
    winner_acc = float((df['pred_winner'] == df['act_winner']).mean())

    # Approx probability from margin normal model for Brier score
    sig = df.apply(_get_conf_std_for_game, axis=1)
    z = (df['pred_margin']) / sig
    p_home = z.apply(_phi)
    y_home = (df['act_winner'] == df['home_team']).astype(float)
    brier = float(((p_home - y_home) ** 2).mean())

    details = df[['season','week','home_team','away_team','pred_home','pred_away','actual_home_points','actual_away_points','pred_total','act_total','pred_margin','act_margin']].copy()
    details = details.to_dict(orient='records')

    return {
        'count': int(len(df)),
        'winner_accuracy': winner_acc,
        'total_mae': total_mae,
        'margin_mae': margin_mae,
        'brier_homewin': brier,
        'details_sample': details[:50]
    }, 200


@app.route('/analysis', methods=['GET'])
def analysis_page():
    # UI for model performance with a simple week selector
    completed = pred_df[(pred_df['season'] == 2025) & pred_df['actual_home_points'].notna() & pred_df['actual_away_points'].notna()].copy()
    completed_weeks = sorted(completed['week'].dropna().unique())
    selected_week = request.args.get('week')
    if selected_week:
        try:
            selected_week = int(selected_week)
        except Exception:
            selected_week = None
    if not selected_week:
        selected_week = completed_weeks[-1] if completed_weeks else None
    # Compute metrics (reuse logic similar to API)
    df = completed.copy()
    if selected_week is not None:
        df = df[df['week'] == selected_week]
    metrics = None
    rows = []
    classes = {}
    if not df.empty:
        df['pred_home'] = df['predicted_home_points']
        df['pred_away'] = df['predicted_away_points']
        df['pred_total'] = df['pred_home'] + df['pred_away']
        df['act_total'] = df['actual_home_points'] + df['actual_away_points']
        df['pred_margin'] = df['pred_home'] - df['pred_away'] if 'predicted_win_margin' not in df.columns else df['predicted_win_margin'].fillna(df['pred_home'] - df['pred_away'])
        df['act_margin'] = df['actual_home_points'] - df['actual_away_points']
        total_mae = float((df['act_total'] - df['pred_total']).abs().mean())
        margin_mae = float((df['act_margin'] - df['pred_margin']).abs().mean())
        df['pred_winner'] = df.apply(lambda r: r['home_team'] if r['pred_home'] > r['pred_away'] else (r['away_team'] if r['pred_away'] > r['pred_home'] else 'TIE'), axis=1)
        df['act_winner'] = df.apply(lambda r: r['home_team'] if r['actual_home_points'] > r['actual_away_points'] else (r['away_team'] if r['actual_away_points'] > r['actual_home_points'] else 'TIE'), axis=1)
        winner_acc = float((df['pred_winner'] == df['act_winner']).mean())
        sig = df.apply(_get_conf_std_for_game, axis=1)
        z = (df['pred_margin']) / sig
        p_home_raw = z.apply(_phi)
        p_home_cal = p_home_raw.apply(_calibrate_win_prob)
        y_home = (df['act_winner'] == df['home_team']).astype(float)
        brier_raw = float(((p_home_raw - y_home) ** 2).mean())
        brier_cal = float(((p_home_cal - y_home) ** 2).mean()) if p_home_cal is not None else brier_raw
        metrics = {
            'count': int(len(df)),
            'winner_accuracy': round(winner_acc, 4),
            'total_mae': round(total_mae, 3),
            'margin_mae': round(margin_mae, 3),
            'brier': round(brier_raw, 4),
            'brier_cal': round(brier_cal, 4),
            'brier_improve': round(brier_raw - brier_cal, 4)
        }
        # Traffic light classes
        def _cls(name, val):
            try:
                v = float(val)
            except Exception:
                return 'na'
            if name == 'winner_accuracy':
                return 'good' if v >= 0.60 else ('ok' if v >= 0.50 else 'bad')
            if name == 'total_mae':
                return 'good' if v <= 10 else ('ok' if v <= 12 else 'bad')
            if name == 'margin_mae':
                return 'good' if v <= 10 else ('ok' if v <= 14 else 'bad')
            if name == 'brier':
                return 'good' if v <= 0.20 else ('ok' if v <= 0.25 else 'bad')
            return 'na'
        classes = {
            'winner_accuracy': _cls('winner_accuracy', metrics['winner_accuracy']),
            'total_mae': _cls('total_mae', metrics['total_mae']),
            'margin_mae': _cls('margin_mae', metrics['margin_mae']),
            'brier': _cls('brier', metrics['brier']),
            'brier_cal': _cls('brier', metrics['brier_cal'])
        }
        show_cols = ['week','home_team','away_team','pred_home','pred_away','actual_home_points','actual_away_points','pred_total','act_total','pred_margin','act_margin']
        rows = df[show_cols].head(25).to_dict(orient='records')
    return render_template_string('''
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; }
            .container { max-width: 900px; margin: 30px auto; background:#fff; padding:24px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); }
            .nav { text-align:right; margin-bottom:8px; }
            select, button { padding:8px 10px; border:1px solid #ccc; border-radius:6px; }
            table { width:100%; border-collapse:collapse; background:#fff; margin-top:12px; }
            th,td { padding:8px 10px; border:1px solid #e0e0e0; text-align:center; }
            th { background:#eaf1fb; }
    .kpis { display:flex; gap:18px; justify-content:center; margin: 10px 0 18px; flex-wrap: wrap; }
        .kpi { background:#f8f8f8; padding:10px 14px; border-radius:10px; display:flex; align-items:center; gap:8px; }
        .dot { width:10px; height:10px; border-radius:50%; background:#bbb; display:inline-block; }
        .kpi.good .dot { background:#2ecc71; }
        .kpi.ok .dot { background:#f1c40f; }
        .kpi.bad .dot { background:#e74c3c; }
        </style>
        <div class="container">
            <div class="nav">
        <a href="/">Main</a> | <a href="/recommendations">Recommendations</a> | <a href="/recommendations/performance">Performance</a>
            </div>
            <h2>Model Performance (2025)</h2>
            <form method="get">
                <label>Week</label>
                <select name="week" onchange="this.form.submit()">
                    {% for w in weeks %}
                        <option value="{{w}}" {% if w == selected_week %}selected{% endif %}>Week {{w}}</option>
                    {% endfor %}
                </select>
            </form>
            {% if metrics %}
            <div class="kpis">
        <div class="kpi"><span class="dot"></span> Games: <b>{{metrics.count}}</b></div>
        <div class="kpi {{classes.winner_accuracy}}"><span class="dot"></span> Winner Acc: <b>{{metrics.winner_accuracy}}</b></div>
        <div class="kpi {{classes.total_mae}}"><span class="dot"></span> Total MAE: <b>{{metrics.total_mae}}</b></div>
        <div class="kpi {{classes.margin_mae}}"><span class="dot"></span> Margin MAE: <b>{{metrics.margin_mae}}</b></div>
    <div class="kpi {{classes.brier}}"><span class="dot"></span> Brier (raw): <b>{{metrics.brier}}</b></div>
    <div class="kpi {{classes.brier_cal}}"><span class="dot"></span> Brier (cal): <b>{{metrics.brier_cal}}</b></div>
    <div class="kpi"><span class="dot" style="background:#3498db"></span> Δ Brier: <b>{{metrics.brier_improve}}</b></div>
            </div>
            <table>
                <tr><th>Wk</th><th>Matchup</th><th>Pred (A-H)</th><th>Actual (A-H)</th><th>Pred Total</th><th>Actual Total</th><th>Pred Margin</th><th>Actual Margin</th></tr>
                {% for r in rows %}
                <tr>
                    <td>{{r['week']}}</td>
                    <td>{{r['away_team']}} @ {{r['home_team']}}</td>
                    <td>{{r['pred_away']}} - {{r['pred_home']}}</td>
                    <td>{{r['actual_away_points']}} - {{r['actual_home_points']}}</td>
                    <td>{{r['pred_total']}}</td>
                    <td>{{r['act_total']}}</td>
                    <td>{{r['pred_margin']}}</td>
                    <td>{{r['act_margin']}}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <div>No completed games found for the selected week.</div>
            {% endif %}
        </div>
    ''', weeks=completed_weeks, selected_week=selected_week, metrics=metrics, rows=rows, classes=classes)


@app.route('/api/recommendations/simple', methods=['GET'])
def recommendations_simple():
    # Recommend EV+ bets for upcoming games. Supports ?week=&bankroll=&kelly_factor=
    week = request.args.get('week')
    bankroll = _safe_float(request.args.get('bankroll', 1000), 1000)
    kelly_factor = _safe_float(request.args.get('kelly_factor', 0.5), 0.5)
    ev_threshold = _safe_float(request.args.get('ev_threshold', 0.02), 0.02)
    recs = compute_recommendations(week=week, bankroll=bankroll, kelly_factor=kelly_factor, ev_threshold=ev_threshold)
    top = recs[:100]

    # Optionally log recommendations
    if request.args.get('log', 'false').lower() == 'true' and top:
        _ensure_recs_file()
        ts = datetime.now(timezone.utc).isoformat()
        out = []
        for r in top:
            out.append({
                'timestamp': ts,
                'season': r['season'], 'week': r['week'], 'home_team': r['home_team'], 'away_team': r['away_team'],
                'market': r['market'], 'side': r['side'], 'price_american': r['price_american'], 'provider': r.get('provider'),
                'line': r.get('line', None),
                'model_prob': r['model_prob'], 'implied_prob': r['implied_prob'], 'edge': r['edge'],
                'kelly_f': r['kelly_f'], 'bankroll': bankroll, 'stake': r['stake'],
                'status': 'open', 'result': 'pending', 'pnl': 0.0
            })
        try:
            existing = pd.read_csv(RECS_PATH) if os.path.exists(RECS_PATH) else pd.DataFrame()
            new_df = pd.DataFrame(out)
            all_df = pd.concat([existing, new_df], ignore_index=True)
            all_df.to_csv(RECS_PATH, index=False)
        except Exception:
            pass

    return {'count': len(top), 'recommendations': top}, 200


@app.route('/api/recommendations/performance', methods=['GET'])
def recommendations_performance():
    # Evaluate and update tracking file using actual outcomes
    if not os.path.exists(RECS_PATH):
        return {'message': 'No recommendations logged yet.'}, 200
    recs_df = pd.read_csv(RECS_PATH)
    if recs_df.empty:
        return {'message': 'No recommendations logged yet.'}, 200

    # Merge in actuals
    merged = recs_df.merge(
        pred_df[['season','week','home_team','away_team','actual_home_points','actual_away_points']],
        on=['season','week','home_team','away_team'], how='left'
    )

    pnl_list = []
    wins = 0
    losses = 0
    pushes = 0
    for idx, r in merged.iterrows():
        status = r.get('status', 'open')
        # If game completed and still open, settle
        ah = r.get('actual_home_points')
        aa = r.get('actual_away_points')
        if pd.notna(ah) and pd.notna(aa) and status == 'open':
            market = r['market']
            outcome_win = False
            push = False
            # Determine outcome
            if market == 'ML':
                winner = 'Home' if ah > aa else ('Away' if aa > ah else 'Tie')
                outcome_win = (winner == r['side'])
                push = (winner == 'Tie')
            elif market == 'Spread':
                line = _safe_float(r.get('line', None))
                if line is None:
                    # If we can't evaluate spread, mark as push
                    push = True
                else:
                    margin = ah - aa
                    # Home bet wins if margin > line, push if == line
                    if r['side'] == 'Home':
                        outcome_win = margin > line
                        push = (abs(margin - line) < 1e-9)
                    else:  # Away
                        outcome_win = margin < line
                        push = (abs(margin - line) < 1e-9)
            elif market == 'Total':
                line = _safe_float(r.get('line', None))
                total = ah + aa
                if line is None:
                    push = True
                else:
                    if r['side'] == 'Over':
                        outcome_win = total > line
                        push = (abs(total - line) < 1e-9)
                    else:
                        outcome_win = total < line
                        push = (abs(total - line) < 1e-9)

            price = _safe_float(r.get('price_american', -110), -110)
            dec, net = american_to_decimal(price)
            stake = _safe_float(r.get('stake', 0), 0)
            pnl = 0.0
            result = 'pending'
            if push:
                result = 'push'
                pushes += 1
                pnl = 0.0
            elif outcome_win:
                result = 'win'
                wins += 1
                pnl = stake * (dec - 1)
            else:
                result = 'loss'
                losses += 1
                pnl = -stake
            pnl_list.append((idx, result, pnl))

    # Apply PnL updates
    if pnl_list:
        for idx, result, pnl in pnl_list:
            recs_df.loc[idx, 'status'] = 'closed'
            recs_df.loc[idx, 'result'] = result
            recs_df.loc[idx, 'pnl'] = round(pnl, 2)
        recs_df.to_csv(RECS_PATH, index=False)

    total_staked = float(recs_df['stake'].sum()) if 'stake' in recs_df.columns else 0.0
    total_pnl = float(recs_df['pnl'].sum()) if 'pnl' in recs_df.columns else 0.0
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

    return {
        'bets_total': int(len(recs_df)),
        'wins': int((recs_df['result'] == 'win').sum()) if 'result' in recs_df.columns else 0,
        'losses': int((recs_df['result'] == 'loss').sum()) if 'result' in recs_df.columns else 0,
        'pushes': int((recs_df['result'] == 'push').sum()) if 'result' in recs_df.columns else 0,
        'total_staked': round(total_staked, 2),
        'total_pnl': round(total_pnl, 2),
        'roi': round(roi, 4)
    }, 200


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations_page():
        # Render a client-side UI that fetches from /api/recommendations with filters/sorting
        weeks = sorted(pred_df['week'].dropna().unique())
        default_bankroll = 1000
        default_kelly = 0.5
        default_ev = 0.02
        default_limit = 100
        return render_template_string('''
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f6f8fb; }
            .container { max-width: 1080px; margin: 30px auto; background:#fff; padding:24px; border-radius:14px; box-shadow:0 8px 24px rgba(0,0,0,.08); }
            h2 { text-align:center; margin-bottom:12px; }
            .toolbar { display:grid; grid-template-columns: repeat(7, minmax(140px,1fr)); gap:12px; align-items:end; margin-bottom:10px; }
            label { font-weight: 600; color: #34495e; font-size:.95em; }
            select,input,button { padding:8px 10px; border:1px solid #d0d7de; border-radius:8px; }
            .tabs { display:flex; gap:10px; margin: 8px 0 14px; }
            .tab { padding:6px 10px; border:1px solid #d0d7de; border-radius:8px; cursor:pointer; color:#34495e; }
            .tab.active { background:#eaf1fb; border-color:#bfd3f2; color:#1d4ed8; font-weight:600; }
            .rec-card { background:#f8fafc; border:1px solid #edf2f7; border-radius:12px; padding:14px; margin:10px 0; display:flex; justify-content:space-between; gap:16px; align-items:center; }
            .lhs { display:flex; flex-direction:column; gap:6px; }
            .teams { display:flex; align-items:center; gap:10px; font-weight:600; }
            .team { display:flex; align-items:center; gap:8px; }
            .logo { width:24px; height:24px; border-radius:50%; background:#eee; display:inline-block; background-size:cover; background-position:center; border:1px solid #ddd; }
            .meta { font-size:.92em; color:#46556a; }
            .edge { font-weight:700; color:#0d3b66; font-size:1.05em; }
            .nav { text-align:right; margin-bottom:8px; }
            .pill { padding:2px 8px; border-radius:999px; font-size:.82em; }
            .pill.high { background:#eaf7ef; color:#1e8e3e; border:1px solid #bfe3c7; }
            .pill.medium { background:#fff7e6; color:#b26b00; border:1px solid #ffe0a3; }
            .pill.low { background:#fdecee; color:#b00020; border:1px solid #f4b4bd; }
            .row { display:flex; gap:10px; align-items:center; }
            .count { color:#555; margin: 6px 0 10px; }
            .controls-row { display:flex; gap:10px; align-items:center; justify-content:space-between; }
            .view-toggle { display:flex; gap:8px; align-items:center; }
            table { width:100%; border-collapse:collapse; background:#fff; }
            th,td { padding:8px 10px; border:1px solid #e0e0e0; text-align:center; }
            th { background:#f1f5f9; }
        </style>
        <div class="container">
            <div class="nav">
                <a href="/">Main</a> | <a href="/conference-records">Conference Records</a> | <a href="/recommendations/performance">Performance</a>
            </div>
            <h2>Betting Recommendations</h2>
            <div class="tabs" id="marketTabs">
                <div class="tab active" data-market="">All</div>
                <div class="tab" data-market="ML">Moneyline</div>
                <div class="tab" data-market="Spread">Spread</div>
                <div class="tab" data-market="Total">Total</div>
            </div>
            <form id="controls" class="toolbar">
                <div>
                    <label>Week</label>
                    <select name="week" id="week">
                        <option value="">All Upcoming</option>
                        {% for w in weeks %}
                            <option value="{{w}}">Week {{w}}</option>
                        {% endfor %}
                    </select>
                </div>
                <div>
                    <label>Sort</label>
                    <select name="sort" id="sort">
                        <option value="edge_desc">Edge ↓</option>
                        <option value="confidence_desc">Confidence ↓</option>
                        <option value="stake_desc">Stake ↓</option>
                        <option value="prob_desc">Model p ↓</option>
                        <option value="time">Time ↑</option>
                        <option value="market">Market A→Z</option>
                    </select>
                </div>
                <div>
                    <label>Bankroll</label>
                    <input type="number" step="1" id="bankroll" value="{{default_bankroll}}"/>
                </div>
                <div>
                    <label>Kelly</label>
                    <input type="number" step="0.05" id="kelly" value="{{default_kelly}}"/>
                </div>
                <div>
                    <label>EV ≥</label>
                    <input type="number" step="0.01" id="ev" value="{{default_ev}}"/>
                </div>
                <div>
                    <label>Limit</label>
                    <input type="number" step="1" id="limit" value="{{default_limit}}"/>
                </div>
                <div style="grid-column: 1 / -1;" class="controls-row">
                    <div class="row">
                        <label>Confidence</label>
                        <select id="confFilter">
                            <option value="">All</option>
                            <option value="High">High</option>
                            <option value="Medium">Medium</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    <div class="view-toggle">
                        <button type="button" id="refreshBtn">Refresh</button>
                        <button type="button" id="logBtn">Log Shown</button>
                        <span>|</span>
                        <label>View</label>
                        <select id="viewMode">
                            <option value="cards">Cards</option>
                            <option value="table">Table</option>
                        </select>
                    </div>
                </div>
            </form>
            <div class="count" id="count"></div>
            <div id="results"></div>
        </div>
        <script>
        const el = id => document.getElementById(id);
        function qs() {
            const p = new URLSearchParams();
            const week = el('week').value.trim();
            const market = document.querySelector('.tab.active')?.dataset.market || '';
            const sort = el('sort').value.trim();
            const bankroll = el('bankroll').value.trim();
            const kelly = el('kelly').value.trim();
            const ev = el('ev').value.trim();
            const limit = el('limit').value.trim();
            if (week) p.set('week', week);
            if (market) p.set('market', market);
            if (sort) p.set('sort', sort);
            if (bankroll) p.set('bankroll', bankroll);
            if (kelly) p.set('kelly', kelly);
            if (ev) p.set('ev', ev);
            if (limit) p.set('limit', limit);
            return p.toString();
        }
        function pillClass(t) {
            const s = (t||'').toLowerCase();
            if (s==='high') return 'pill high';
            if (s==='medium') return 'pill medium';
            return 'pill low';
        }
        function fmt(n, d=2) {
            if (n===null || n===undefined || Number.isNaN(n)) return '';
            try { return Number(n).toFixed(d); } catch { return n; }
        }
        function toLocal(iso) {
            try { return new Date(iso).toLocaleString(); } catch { return iso||''; }
        }
        function renderCards(list) {
            const frag = document.createDocumentFragment();
            list.forEach(r => {
                const card = document.createElement('div');
                card.className = 'rec-card';
                const left = document.createElement('div');
                left.className = 'lhs';
                left.innerHTML = `
                    <div class="row">
                      <span class="pill ${pillClass(r.confidence)}">${r.confidence||''}</span>
                      <span class="meta">${toLocal(r.start_iso)||r.game_time||''}</span>
                    </div>
                    <div class="teams">
                      <span class="team"><span class="logo" style="background-image:url('${r.away_logo||''}')"></span>${r.away_team}</span>
                      <span>@</span>
                      <span class="team"><span class="logo" style="background-image:url('${r.home_logo||''}')"></span>${r.home_team}</span>
                    </div>
                    <div><b>${r.market}</b> — ${r.side}${r.line!==undefined && r.line!==null ? ' ' + r.line : ''} — Price ${r.price_american} — <span class="meta">${r.provider||''}</span></div>
                    <div>Model p: ${fmt(r.model_prob,3)} | Implied: ${fmt(r.implied_prob,3)} | Kelly: ${fmt(r.kelly_f,3)} | Stake: $${fmt(r.stake,2)}</div>
                `;
                const right = document.createElement('div');
                right.innerHTML = `<div class="edge">Edge: ${fmt(r.edge,3)}</div>`;
                card.appendChild(left);
                card.appendChild(right);
                frag.appendChild(card);
            });
            el('results').appendChild(frag);
        }
        function renderTable(list) {
            const cols = ['week','start_iso','market','side','line','price_american','home_team','away_team','provider','model_prob','implied_prob','edge','kelly_f','stake','confidence'];
            const tbl = document.createElement('table');
            const thead = document.createElement('thead');
            thead.innerHTML = '<tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>';
            const tbody = document.createElement('tbody');
            list.forEach(r => {
                const tr = document.createElement('tr');
                cols.forEach(c => {
                    const td = document.createElement('td');
                    let v = r[c];
                    if (c==='start_iso') v = toLocal(r.start_iso||'');
                    if (typeof v==='number') v = fmt(v, c==='stake'?2:3);
                    td.textContent = v==null?'':v;
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            tbl.appendChild(thead); tbl.appendChild(tbody);
            el('results').appendChild(tbl);
        }
        async function fetchRecs() {
            const url = '/api/recommendations?' + qs();
            el('count').textContent = 'Loading…';
            el('results').innerHTML = '';
            try {
                const res = await fetch(url);
                const data = await res.json();
                let list = data.results || [];
                // Confidence filter
                const cf = (el('confFilter').value||'').trim();
                if (cf) list = list.filter(x => (x.confidence||'')===cf);
                el('count').textContent = `Results: ${list.length}${data.week!==null?` (Week ${data.week})`:''}`;
                const mode = el('viewMode').value;
                if (mode==='table') renderTable(list); else renderCards(list);
            } catch (e) {
                el('count').textContent = 'Error loading recommendations.';
            }
        }
        async function logShown() {
            // Use the simple API to log top N (best effort; may not match sort exactly)
            const p = new URLSearchParams();
            const week = el('week').value.trim();
            if (week) p.set('week', week);
            p.set('bankroll', el('bankroll').value.trim());
            p.set('kelly_factor', el('kelly').value.trim());
            p.set('ev_threshold', el('ev').value.trim());
            p.set('log', 'true');
            const url = '/api/recommendations/simple?' + p.toString();
            try {
                const res = await fetch(url);
                const data = await res.json();
                alert(`Logged ${data.count||0} recommendations.`);
            } catch (e) {
                alert('Log failed.');
            }
        }
        el('refreshBtn').addEventListener('click', fetchRecs);
        el('logBtn').addEventListener('click', logShown);
        // Tabs
        document.querySelectorAll('#marketTabs .tab').forEach(t => t.addEventListener('click', () => {
            document.querySelectorAll('#marketTabs .tab').forEach(x => x.classList.remove('active'));
            t.classList.add('active');
            fetchRecs();
        }));
        // View toggle
        el('viewMode').addEventListener('change', fetchRecs);
        // Auto-load on page open
        fetchRecs();
        </script>
        ''', weeks=weeks, default_bankroll=default_bankroll, default_kelly=default_kelly, default_ev=default_ev, default_limit=default_limit)
@app.route('/recommendations/performance')
def recommendations_performance_page():
    # Read performance via the same CSV and simple aggregation
    if not os.path.exists(RECS_PATH):
        recs_df = pd.DataFrame()
    else:
        recs_df = pd.read_csv(RECS_PATH)
    total = int(len(recs_df)) if not recs_df.empty else 0
    wins = int((recs_df['result'] == 'win').sum()) if total else 0
    losses = int((recs_df['result'] == 'loss').sum()) if total else 0
    pushes = int((recs_df['result'] == 'push').sum()) if total else 0
    staked = float(recs_df['stake'].sum()) if total and 'stake' in recs_df.columns else 0.0
    pnl = float(recs_df['pnl'].sum()) if total and 'pnl' in recs_df.columns else 0.0
    roi = (pnl / staked) if staked > 0 else 0.0
    last20 = recs_df.tail(20) if not recs_df.empty else pd.DataFrame()
    return render_template_string('''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; }
        .container { max-width: 900px; margin: 30px auto; background:#fff; padding:24px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); }
        h2 { text-align:center; margin-bottom:18px; }
        table { width:100%; border-collapse:collapse; background:#fff; }
        th,td { padding:8px 10px; border:1px solid #e0e0e0; text-align:center; }
        th { background:#eaf1fb; }
        .nav { text-align:right; margin-bottom:8px; }
        .kpis { display:flex; gap:18px; justify-content:center; margin: 10px 0 18px; }
        .kpi { background:#f8f8f8; padding:10px 14px; border-radius:10px; }
    </style>
    <div class="container">
        <div class="nav">
            <a href="/">Main</a> | <a href="/recommendations">Recommendations</a>
        </div>
        <h2>Betting Performance</h2>
        <div class="kpis">
            <div class="kpi">Total Bets: <b>{{total}}</b></div>
            <div class="kpi">Wins: <b>{{wins}}</b></div>
            <div class="kpi">Losses: <b>{{losses}}</b></div>
            <div class="kpi">Pushes: <b>{{pushes}}</b></div>
            <div class="kpi">Staked: <b>${{staked}}</b></div>
            <div class="kpi">PnL: <b>${{pnl}}</b></div>
            <div class="kpi">ROI: <b>{{roi}}</b></div>
        </div>
        <h3>Most Recent 20</h3>
        <table>
            <tr><th>Time</th><th>Week</th><th>Matchup</th><th>Market</th><th>Side</th><th>Price</th><th>Line</th><th>Stake</th><th>Status</th><th>Result</th><th>PnL</th></tr>
            {% for _, r in last20.iterrows() %}
            <tr>
                <td>{{r.get('timestamp','')}}</td>
                <td>{{r.get('week','')}}</td>
                <td>{{r.get('away_team','')}} @ {{r.get('home_team','')}}</td>
                <td>{{r.get('market','')}}</td>
                <td>{{r.get('side','')}}</td>
                <td>{{r.get('price_american','')}}</td>
                <td>{{r.get('line','')}}</td>
                <td>{{r.get('stake','')}}</td>
                <td>{{r.get('status','')}}</td>
                <td>{{r.get('result','')}}</td>
                <td>{{r.get('pnl','')}}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    ''', total=total, wins=wins, losses=losses, pushes=pushes, staked=round(staked,2), pnl=round(pnl,2), roi=round(roi,4), last20=last20)


def _do_refresh(quick: bool):
    """Execute the refresh pipeline and return (payload_dict, status_code)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))  # .../NCAFCompare
    # Try to ensure CFBD key is loaded in env before any network steps
    _ensure_cfbd_key()
    py = sys.executable or 'python'
    # Try to honor week from current request context if available
    week_arg = None
    try:
        from flask import request as _rq
        w = _rq.args.get('week', '').strip()
        week_arg = int(w) if w != '' else None
    except Exception:
        week_arg = None
    # Helper to resolve a script path across both top-level src/ and NCAFCompare/src/
    def _resolve_script(rel_path: str) -> str | None:
        cand = [
            os.path.join(base_dir, 'src', 'data', rel_path),
            os.path.join(base_dir, 'NCAFCompare', 'src', 'data', rel_path),
        ]
        for p in cand:
            try:
                if os.path.exists(p):
                    return p
            except Exception:
                continue
        return None

    if quick:
        cmds = []
        s1 = _resolve_script('update_scores_2025.py')
        s2 = _resolve_script('fetch_2025_lines.py')
        if s1:
            cmds.append([py, s1] + (["--week", str(week_arg)] if week_arg is not None else []))
        if s2:
            cmds.append([py, s2] + (["--week", str(week_arg)] if week_arg is not None else []))
    else:
        cmds = []
        g = _resolve_script('geocode_venues_2025.py')
        if g:
            cmds.append([py, g, '--max-new', '120'] + (["--week", str(week_arg)] if week_arg is not None else []))
        e = _resolve_script('enrich_weather_2025.py')
        if e:
            cmds.append([py, e] + (["--week", str(week_arg)] if week_arg is not None else []))
        o = _resolve_script('orchestrate_refresh.py')
        if o:
            cmds.append([py, o])
        m = _resolve_script('merge_all_features.py')
        if m:
            cmds.append([py, m])
        # generate_enhanced_predictions.py is at project root
        gen = os.path.join(base_dir, 'generate_enhanced_predictions.py')
        if os.path.exists(gen):
            cmds.append([py, gen])
        s1 = _resolve_script('update_scores_2025.py')
        s2 = _resolve_script('fetch_2025_lines.py')
        if s1:
            cmds.append([py, s1] + (["--week", str(week_arg)] if week_arg is not None else []))
        if s2:
            cmds.append([py, s2] + (["--week", str(week_arg)] if week_arg is not None else []))
        if os.path.exists(gen):
            cmds.append([py, gen])
    ran = []
    t0 = time.time()
    for cmd in cmds:
        try:
            step_start = time.time()
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
            step_dur = round(time.time() - step_start, 2)
            ran.append({'cmd': ' '.join(cmd), 'seconds': step_dur, 'returncode': out.returncode, 'stdout': out.stdout[-3000:], 'stderr': out.stderr[-1500:]})
        except Exception as e:
            ran.append({'cmd': ' '.join(cmd), 'error': str(e)})
    # Try to update actual scores via CFBD if API key is available
    try:
        week_hint = None
        try:
            from flask import request as _rq
            w = _rq.args.get('week', '').strip()
            week_hint = int(w) if w != '' else None
        except Exception:
            week_hint = None
        try:
            from flask import request as _rq
            ow = (_rq.args.get('overwrite', '0') == '1')
        except Exception:
            ow = False
        cfbd_res = _update_scores_with_cfbd(week_hint, overwrite=ow)
        if isinstance(cfbd_res, dict):
            ran.append(cfbd_res)
    except Exception as _e_cfbd:
        ran.append({'step': 'cfbd_update', 'error': str(_e_cfbd)})
    # Reload predictions
    try:
        _reload_predictions()
    except Exception as e:
        ran.append({'step': 'reload_predictions', 'error': str(e)})
        payload = {'status': 'error', 'details': ran, 'mode': ('quick' if quick else 'full')}
        return payload, 500
    # Reload lines if 2025 file exists; extend/overlay to lines_df
    try:
        _overlay_lines_2025_if_present()
    except Exception as e:
        ran.append({'step': 'reload_lines', 'error': str(e)})

    try:
        sub = pred_df[pred_df['season'] == 2025]
        uh = sub['predicted_home_points'].nunique(dropna=True) if 'predicted_home_points' in sub.columns else None
        ua = sub['predicted_away_points'].nunique(dropna=True) if 'predicted_away_points' in sub.columns else None
        ut = sub['predicted_total_points'].nunique(dropna=True) if 'predicted_total_points' in sub.columns else None
    except Exception:
        uh = ua = ut = None
    # Auto-log and settle
    try:
        from datetime import datetime as _dt
        stamp = _dt.utcnow().strftime('%Y-%m-%d')
        marker = os.path.join(base_dir, f'.autolog_{stamp}.txt')
        if not os.path.exists(marker):
            _ = compute_recommendations(week=None, bankroll=1000.0, kelly_factor=0.5, ev_threshold=0.03)
            top = _[:25]
            if top:
                _ensure_recs_file()
                ts = _dt.utcnow().isoformat()
                import pandas as _pd
                existing = _pd.read_csv(RECS_PATH) if os.path.exists(RECS_PATH) else _pd.DataFrame()
                new_df = _pd.DataFrame([
                    {'timestamp': ts, 'season': r['season'], 'week': r['week'], 'home_team': r['home_team'], 'away_team': r['away_team'], 'market': r['market'], 'side': r['side'], 'price_american': r['price_american'], 'provider': r.get('provider'), 'line': r.get('line', None), 'model_prob': r['model_prob'], 'implied_prob': r['implied_prob'], 'edge': r['edge'], 'kelly_f': r['kelly_f'], 'bankroll': 1000.0, 'stake': r['stake'], 'status': 'open', 'result': 'pending', 'pnl': 0.0}
                    for r in top
                ])
                all_df = _pd.concat([existing, new_df], ignore_index=True)
                all_df.to_csv(RECS_PATH, index=False)
                with open(marker, 'w') as f: f.write('ok')
        _ = recommendations_performance()
    except Exception as e:
        ran.append({'step': 'auto_log_or_settle', 'error': str(e)})

    total_seconds = round(time.time() - t0, 2)
    payload = {
        'status': 'ok',
        'details': ran,
        'seconds_total': total_seconds,
        'rows': int(len(pred_df)),
        'lines_rows': int(len(lines_df)) if isinstance(lines_df, pd.DataFrame) else None,
        'pred_source': PRED_SOURCE,
        'mode': ('quick' if quick else 'full'),
        'unique_home_preds': int(uh) if uh is not None else None,
        'unique_away_preds': int(ua) if ua is not None else None,
        'unique_total_preds': int(ut) if ut is not None else None,
    }
    return payload, 200

@app.route('/api/refresh-data', methods=['GET','POST'])
def refresh_data():
    # Synchronous refresh (may take 1-3 minutes)
    mode = request.args.get('mode', '').lower()
    quick = (mode == 'quick')
    payload, code = _do_refresh(quick)
    return payload, code

def _refresh_thread(quick: bool):
    global REFRESH_STATE
    try:
        # Ensure CFBD key is present before we start
        _ensure_cfbd_key()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        py = sys.executable or 'python'
        # Optional week filter from state
        with _REFRESH_LOCK:
            selected_week = REFRESH_STATE.get('week')
        # Helper to resolve a script path across both top-level src/ and NCAFCompare/src/
        def _resolve_script(rel_path: str) -> str | None:
            cand = [
                os.path.join(base_dir, 'src', 'data', rel_path),
                os.path.join(base_dir, 'NCAFCompare', 'src', 'data', rel_path),
            ]
            for p in cand:
                try:
                    if os.path.exists(p):
                        return p
                except Exception:
                    continue
            return None

        if quick:
            cmds = []
            s1 = _resolve_script('update_scores_2025.py')
            s2 = _resolve_script('fetch_2025_lines.py')
            if s1:
                cmds.append([py, s1] + (["--week", str(selected_week)] if selected_week is not None else []))
            if s2:
                cmds.append([py, s2] + (["--week", str(selected_week)] if selected_week is not None else []))
        else:
            cmds = []
            g = _resolve_script('geocode_venues_2025.py')
            if g:
                if _geocode_needed(base_dir, selected_week):
                    cmds.append([py, g, '--max-new', '120'] + (["--week", str(selected_week)] if selected_week is not None else []))
                else:
                    with _REFRESH_LOCK:
                        REFRESH_STATE['details'].append({'step': 'geocode_venues_2025.py', 'skipped': 'cache up-to-date'})
            e = _resolve_script('enrich_weather_2025.py')
            if e:
                cmds.append([py, e] + (["--week", str(selected_week)] if selected_week is not None else []))
            o = _resolve_script('orchestrate_refresh.py')
            if o:
                cmds.append([py, o])
            m = _resolve_script('merge_all_features.py')
            if m:
                cmds.append([py, m])
            gen = os.path.join(base_dir, 'generate_enhanced_predictions.py')
            if os.path.exists(gen):
                cmds.append([py, gen])
            s1 = _resolve_script('update_scores_2025.py')
            s2 = _resolve_script('fetch_2025_lines.py')
            if s1:
                cmds.append([py, s1] + (["--week", str(selected_week)] if selected_week is not None else []))
            if s2:
                cmds.append([py, s2] + (["--week", str(selected_week)] if selected_week is not None else []))
            if os.path.exists(gen):
                cmds.append([py, gen])
        t0 = time.time()
        # Run each step, updating progress between steps
        for idx, cmd in enumerate(cmds):
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append({'cmd': ' '.join(cmd), 'status': 'running', 'started': time.time()})
            step_start = time.time()
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, check=False)
                step_dur = round(time.time() - step_start, 2)
                so = (out.stdout or '')[-3000:]
                se = (out.stderr or '')[-1500:]
                if not so and not se:
                    so = '(no output)'
                entry = {'cmd': ' '.join(cmd), 'seconds': step_dur, 'returncode': out.returncode, 'stdout': so, 'stderr': se}
            except Exception as e:
                entry = {'cmd': ' '.join(cmd), 'error': str(e)}
            with _REFRESH_LOCK:
                # Replace the running marker with final entry
                REFRESH_STATE['details'][-1] = entry
                REFRESH_STATE['seconds_total'] = round(time.time() - t0, 2)
        # Update actual scores via CFBD/ESPN (best-effort)
        try:
            with _REFRESH_LOCK:
                wk = REFRESH_STATE.get('week')
                ow = bool(REFRESH_STATE.get('overwrite'))
            res = _update_scores_with_cfbd(wk, overwrite=ow)
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append(res if isinstance(res, dict) else {'step': 'cfbd_update', 'note': 'no_result'})
        except Exception as _e:
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append({'step': 'cfbd_update', 'error': str(_e)})
        # Reload predictions
        try:
            _reload_predictions()
        except Exception as e:
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append({'step': 'reload_predictions', 'error': str(e)})
                REFRESH_STATE['status'] = 'error'
                REFRESH_STATE['finished_at'] = time.time()
            return
        # Reload lines overlay
        try:
            _overlay_lines_2025_if_present()
        except Exception as e:
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append({'step': 'reload_lines', 'error': str(e)})
        # Compute summary numbers
        try:
            sub = pred_df[pred_df['season'] == 2025]
            uh = sub['predicted_home_points'].nunique(dropna=True) if 'predicted_home_points' in sub.columns else None
            ua = sub['predicted_away_points'].nunique(dropna=True) if 'predicted_away_points' in sub.columns else None
            ut = sub['predicted_total_points'].nunique(dropna=True) if 'predicted_total_points' in sub.columns else None
        except Exception:
            uh = ua = ut = None
        # Auto-log/settle
        try:
            from datetime import datetime as _dt
            stamp = _dt.utcnow().strftime('%Y-%m-%d')
            marker = os.path.join(base_dir, f'.autolog_{stamp}.txt')
            if not os.path.exists(marker):
                _ = compute_recommendations(week=None, bankroll=1000.0, kelly_factor=0.5, ev_threshold=0.03)
                top = _[:25]
                if top:
                    _ensure_recs_file()
                    ts = _dt.utcnow().isoformat()
                    import pandas as _pd
                    existing = _pd.read_csv(RECS_PATH) if os.path.exists(RECS_PATH) else _pd.DataFrame()
                    new_df = _pd.DataFrame([
                        {'timestamp': ts, 'season': r['season'], 'week': r['week'], 'home_team': r['home_team'], 'away_team': r['away_team'], 'market': r['market'], 'side': r['side'], 'price_american': r['price_american'], 'provider': r.get('provider'), 'line': r.get('line', None), 'model_prob': r['model_prob'], 'implied_prob': r['implied_prob'], 'edge': r['edge'], 'kelly_f': r['kelly_f'], 'bankroll': 1000.0, 'stake': r['stake'], 'status': 'open', 'result': 'pending', 'pnl': 0.0}
                        for r in top
                    ])
                    all_df = _pd.concat([existing, new_df], ignore_index=True)
                    all_df.to_csv(RECS_PATH, index=False)
                    with open(marker, 'w') as f: f.write('ok')
            _ = recommendations_performance()
        except Exception as e:
            with _REFRESH_LOCK:
                REFRESH_STATE['details'].append({'step': 'auto_log_or_settle', 'error': str(e)})
        # Finalize state
        with _REFRESH_LOCK:
            REFRESH_STATE.update({
                'status': 'ok',
                'seconds_total': round(time.time() - t0, 2),
                'rows': int(len(pred_df)),
                'lines_rows': int(len(lines_df)) if isinstance(lines_df, pd.DataFrame) else None,
                'pred_source': PRED_SOURCE,
                'unique_home_preds': int(uh) if uh is not None else None,
                'unique_away_preds': int(ua) if ua is not None else None,
                'unique_total_preds': int(ut) if ut is not None else None,
                'finished_at': time.time(),
            })
    except Exception as e:
        with _REFRESH_LOCK:
            REFRESH_STATE['status'] = 'error'
            REFRESH_STATE['error'] = str(e)
            REFRESH_STATE['finished_at'] = time.time()

@app.route('/api/refresh-start', methods=['POST','GET'])
def refresh_start():
    # Start refresh in the background; immediate return for reliable UX
    mode = request.args.get('mode', '').lower()
    week = request.args.get('week', '').strip()
    overwrite = request.args.get('overwrite', '0') == '1'
    try:
        week_int = int(week) if week != '' else None
    except Exception:
        week_int = None
    quick = (mode == 'quick')
    with _REFRESH_LOCK:
        if REFRESH_STATE.get('status') == 'running':
            return {**REFRESH_STATE, 'note': 'already running'}, 409
        REFRESH_STATE.update({
            'status': 'running', 'mode': ('quick' if quick else 'full'), 'started_at': time.time(), 'finished_at': None,
            'seconds_total': None, 'details': [], 'pred_source': None, 'rows': None, 'lines_rows': None,
            'unique_home_preds': None, 'unique_away_preds': None, 'unique_total_preds': None, 'error': None,
            'week': week_int,
            'overwrite': overwrite,
        })
    th = threading.Thread(target=_refresh_thread, args=(quick,), daemon=True)
    th.start()
    return {'status': 'running', 'mode': REFRESH_STATE['mode'], 'week': week_int}, 202

@app.route('/api/refresh-progress')
def refresh_progress():
    with _REFRESH_LOCK:
        state = dict(REFRESH_STATE)
    # Add elapsed if running
    if state.get('status') == 'running' and state.get('started_at'):
        state['elapsed'] = round(time.time() - state['started_at'], 1)
    return state, 200

@app.route('/refresh-status')
def refresh_status():
    """Live refresh dashboard (non-blocking). Start background job and poll progress."""
    return render_template_string('''
    <style>
        body { font-family: Segoe UI, Arial, sans-serif; background:#f6f8fb; margin:0; }
        .wrap { max-width: 1000px; margin: 24px auto; background:#fff; border-radius:12px; padding:18px 22px; box-shadow:0 2px 10px rgba(0,0,0,.06); }
        h2 { margin: 0 0 12px; color:#2c3e50; }
        .meta { color:#34495e; margin: 8px 0 12px; }
        table { width:100%; border-collapse: collapse; }
        th, td { border:1px solid #e6e9ef; padding:8px 10px; text-align:left; font-size: 14px; }
        th { background:#f0f4fa; }
        .ok { color:#1e8449; font-weight:600; }
        .err { color:#c0392b; font-weight:600; }
        details { margin-top:6px; }
        pre { max-height:160px; overflow:auto; background:#fafbfe; padding:8px; border-radius:6px; }
        .actions button { margin-right:12px; }
        .muted { color:#7f8c8d; }
    </style>
    <div class="wrap">
        <h2>Refresh Diagnostics (Live)</h2>
    <div class="actions">
            <label>Week: <input id="weekInp" type="number" min="0" max="20" style="width:80px"></label>
            <label style="margin-left:12px"><input id="owInp" type="checkbox"> Overwrite finals</label>
            <button id="runFull">Run Full</button>
            <button id="runQuick">Run Quick</button>
            <span id="note" class="muted"></span>
        </div>
        <div class="meta" id="meta">Loading…</div>
        <div id="details"></div>
    </div>
    <script>
    (function(){
        async function start(mode){
            document.getElementById('note').textContent = 'Starting ' + (mode||'full') + '…';
            const wk = document.getElementById('weekInp').value.trim();
            let url = '/api/refresh-start' + (mode==='quick'?'?mode=quick':'');
            if(wk !== '') url += (url.includes('?')?'&':'?') + 'week=' + encodeURIComponent(wk);
            if(document.getElementById('owInp').checked) url += (url.includes('?')?'&':'?') + 'overwrite=1';
            const res = await fetch(url, {method:'POST'});
            if(!res.ok){ document.getElementById('note').textContent = 'Start failed.'; return; }
            poll();
        }
        async function poll(){
            try{
                const r = await fetch('/api/refresh-progress');
                const j = await r.json();
                const meta = document.getElementById('meta');
                const det = document.getElementById('details');
                const rows = j.rows ?? '-';
                const lines = j.lines_rows ?? '-';
                const secs = j.seconds_total ? (j.seconds_total + 's') : (j.elapsed ? (j.elapsed + 's') : '-');
                meta.innerHTML = `Status: <b>${j.status}</b> • Mode: ${j.mode||'-'} • Time: ${secs} • Rows: ${rows} | Lines: ${lines} • Source: ${j.pred_source||'-'}`;
                if(Array.isArray(j.details) && j.details.length){
                    let html = '<table><thead><tr><th>#</th><th>Step</th><th>Seconds</th><th>Return</th><th>Output</th></tr></thead><tbody>';
                    j.details.forEach((d,i)=>{
                        html += `<tr><td>${i+1}</td><td style="word-break:break-all">${(d.cmd||d.step||'')}</td><td>${d.seconds??''}</td><td>${d.returncode??''}</td><td>`;
                        if(d.stdout || d.stderr){
                            html += '<details><summary>logs</summary>';
                            if(d.stdout) html += `<div><strong>stdout</strong><pre>${d.stdout}</pre></div>`;
                            if(d.stderr) html += `<div><strong>stderr</strong><pre>${d.stderr}</pre></div>`;
                            html += '</details>';
                        } else {
                            html += '<span class="muted">(no logs)</span>';
                        }
                        html += '</td></tr>';
                    });
                    html += '</tbody></table>';
                    det.innerHTML = html;
                } else if(j.status === 'running') {
                    det.innerHTML = '<div class="muted">Running… (details will appear when available)</div>';
                }
                if(j.status === 'running'){
                    setTimeout(poll, 1000);
                }
            }catch(e){
                document.getElementById('meta').textContent = 'Error loading progress: ' + e;
            }
        }
        document.getElementById('runFull').addEventListener('click', ()=>start('full'));
        document.getElementById('runQuick').addEventListener('click', ()=>start('quick'));
        // If ?autostart=1, kick off a full run immediately
        const params = new URLSearchParams(window.location.search);
        if(params.get('autostart')==='1') { start('full'); }
        // Always begin polling to show last known state
        poll();
    })();
    </script>
    ''')

@app.route('/health', methods=['GET','HEAD'])
def health():
    if request.method == 'HEAD':
        return '', 200
    try:
        sub = pred_df[pred_df['season'] == 2025] if 'season' in pred_df.columns else pred_df
        return {
            'status': 'ok',
            'pred_source': PRED_SOURCE,
            'rows': int(len(pred_df)),
            'rows_2025': int(len(sub)),
        }, 200
    except Exception as e:
        return {'status': 'error', 'error': str(e)}, 500

@app.route('/win-totals')
def win_totals_page():
    # Compute expected wins for each team using per-game win probabilities
    df = pred_df[pred_df['season'] == 2025].copy()
    if df.empty:
        return render_template_string('<div class="container"><h2>No 2025 schedule loaded.</h2></div>')
    def _prob_home_win(row):
        ph = _safe_float(row.get('predicted_home_points'))
        pa = _safe_float(row.get('predicted_away_points'))
        if ph is None or pa is None:
            return None
        pm = _safe_float(row.get('predicted_win_margin'), ph - pa)
        sig = _get_conf_std_for_game(row)
        return _phi(pm / sig)
    df['p_home'] = df.apply(_prob_home_win, axis=1)
    # Expected wins per team
    exp = {}
    for _, r in df.iterrows():
        h = r['home_team']; a = r['away_team']
        p = r.get('p_home', None)
        if h not in exp: exp[h] = 0.0
        if a not in exp: exp[a] = 0.0
        if p is not None:
            exp[h] += float(p)
            exp[a] += float(1.0 - p)
    rows = sorted(([t, round(w,2)] for t,w in exp.items()), key=lambda x: (-x[1], x[0]))
    # Join conferences for context
    conf_map_local = dict(zip(team_conf_df['school'], team_conf_df['conference']))
    data = [{ 'team': t, 'exp_wins': w, 'conference': conf_map_local.get(t, 'Unknown') } for t,w in rows]
    return render_template_string('''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6fa; }
        .container { max-width: 900px; margin: 30px auto; background:#fff; padding:24px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); }
        table { width:100%; border-collapse: collapse; }
        th,td { padding:8px 10px; border:1px solid #e0e0e0; text-align:left; }
        th { background:#eaf1fb; }
        .nav { text-align:right; margin-bottom:8px; }
    </style>
    <div class="container">
        <div class="nav">
            <a href="/">Main</a> | <a href="/analysis">Analysis</a> | <a href="/recommendations">Recommendations</a>
        </div>
        <h2>2025 Expected Wins (Model)</h2>
        <table>
            <tr><th>Team</th><th>Conference</th><th>Expected Wins</th></tr>
            {% for r in rows %}
            <tr><td>{{r.team}}</td><td>{{r.conference}}</td><td>{{r.exp_wins}}</td></tr>
            {% endfor %}
        </table>
    </div>
    ''', rows=data)

@app.route('/api/admin/reload', methods=['POST','GET'])
def admin_reload():
    auth_token = os.environ.get('ADMIN_TOKEN')
    supplied = request.args.get('token') or request.headers.get('X-Admin-Token')
    if auth_token and auth_token != supplied:
        return {'error': 'unauthorized'}, 401
    with _AUTO_REFRESH_LOCK:
        try:
            _reload_predictions()
            try:
                _overlay_lines_2025_if_present()
            except Exception:
                pass
            settle = _settle_recommendations()
            return {'reloaded': True, 'rows': int(len(pred_df)), 'settled': settle}, 200
        except Exception as e:
            return {'error': str(e)}, 500

@app.route('/api/admin/fetch-odds', methods=['POST','GET'])
def admin_fetch_odds():
    auth_token = os.environ.get('ADMIN_TOKEN')
    supplied = request.args.get('token') or request.headers.get('X-Admin-Token')
    if auth_token and auth_token != supplied:
        return {'error': 'unauthorized'}, 401
    week_arg = request.args.get('week', '').strip()
    wk = None
    try:
        if week_arg != '':
            wk = int(week_arg)
    except Exception:
        wk = None
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        return {'error': 'ODDS_API_KEY not set'}, 400
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fetch_2025_lines.py')
    if not os.path.exists(script):
        return {'error': 'fetch_2025_lines.py not found'}, 404
    cmd = [sys.executable or 'python', script]
    if wk is not None:
        cmd += ['--week', str(wk)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=150)
        try:
            _overlay_lines_2025_if_present()
        except Exception:
            pass
        return {
            'returncode': out.returncode,
            'stdout_tail': (out.stdout or '')[-4000:],
            'stderr_tail': (out.stderr or '')[-2000:],
            'lines_rows': int(len(lines_df)) if isinstance(lines_df, pd.DataFrame) else None,
        }, 200 if out.returncode == 0 else 500
    except Exception as e:
        return {'error': str(e)}, 500

# Launch auto-refresh thread if enabled and not already started
if os.environ.get('DISABLE_AUTO_REFRESH','0') != '1':
    try:
        if not any(th.name == 'auto-refresh' for th in threading.enumerate()):
            threading.Thread(target=_auto_refresh_loop, name='auto-refresh', daemon=True).start()
    except Exception as _e:
        try:
            print(f"[auto-refresh] failed to ensure thread: {_e}")
        except Exception:
            pass

@app.route('/api/game-cards')
def api_game_cards():
    """Return game card data (predictions + actuals + odds) as JSON.
    Query params:
      week: int, defaults to latest week with any finals (or earliest if none)
      filter_type: all|completed|upcoming
      conference: filter if team conf matches
      date: YYYY-MM-DD (start_date prefix) when available
      full=1 : bypass initial cap (otherwise finals + 80 upcoming)
      sort: time|winprob_desc|ou_edge_desc|ats_edge_desc
    """
    try:
        # Cache lookup key based on request args (stable ordering)
        key = (
            'v3',  # cache version bump after adding FBS vs Non-FBS inclusion & classification
            request.args.get('week',''),
            request.args.get('filter_type','all'),
            request.args.get('conference',''),
            request.args.get('date',''),
            request.args.get('full','0'),
            request.args.get('sort','time')
        )
        cached = GAME_CARDS_CACHE.get(key)
        if cached:
            return jsonify(cached), 200
        weeks = sorted(pred_df['week'].dropna().unique())
        sel_week = None
        w_q = request.args.get('week')
        if w_q and w_q.isdigit():
            wi = int(w_q)
            if wi in weeks:
                sel_week = wi
        if sel_week is None and weeks:
            try:
                finals_per = {}
                for w in weeks:
                    subw = pred_df[pred_df['week']==w]
                    finals_per[w] = int(((subw['actual_home_points'].notna()) & (subw['actual_away_points'].notna())).sum())
                with_finals = [w for w,c in finals_per.items() if c>0]
                sel_week = max(with_finals) if with_finals else min(weeks)
            except Exception:
                sel_week = min(weeks)
        dfw = pred_df[pred_df['week']==sel_week].copy() if sel_week is not None else pred_df.copy()
        # Exclude only non-FBS vs non-FBS games; keep FBS vs FBS and FBS vs Non-FBS
        if {'home_conference','away_conference'}.issubset(dfw.columns):
            fbs_confs = {
                'acc','sec','big ten','big 12','pac 12','american','mountain west','sun belt','mac','conference usa','independent','independents','fbs independents','independent (fbs)'
            }
            fbs_indies = {'notre dame','army','navy','umass','uconn','new mexico state'}
            def _is_fbs(team, conf):
                try:
                    t = str(team or '').strip().lower(); c = str(conf or '').strip().lower()
                    return c in fbs_confs or t in fbs_indies
                except Exception:
                    return False
            dfw = dfw[dfw.apply(lambda r: _is_fbs(r.get('home_team'), r.get('home_conference')) or _is_fbs(r.get('away_team'), r.get('away_conference')), axis=1)]
        dfw['date_only'] = dfw.get('start_date','').astype(str).str[:10]
        # Filters
        filt_type = request.args.get('filter_type','all')
        conf_q = request.args.get('conference','')
        date_q = request.args.get('date','')
        show_full = request.args.get('full','0') in ('1','true','yes')
        if date_q:
            dfw = dfw[dfw['date_only']==date_q]
        if conf_q:
            dfw = dfw[(dfw['home_conference']==conf_q) | (dfw['away_conference']==conf_q)]
        if filt_type == 'completed':
            dfw = dfw[(dfw['actual_home_points'].notna()) & (dfw['actual_away_points'].notna())]
        elif filt_type == 'upcoming':
            dfw = dfw[(dfw['actual_home_points'].isna()) & (dfw['actual_away_points'].isna())]
        # Build game cards
        cards = []
        for _, r in dfw.iterrows():
            try:
                cards.append(_build_game_card(r))
            except Exception:
                continue
        # Sort
        sort_by = request.args.get('sort','time')
        try:
            if sort_by == 'winprob_desc':
                cards.sort(key=lambda g: (g.get('home_win_prob') or 0.0), reverse=True)
            elif sort_by == 'ou_edge_desc':
                cards.sort(key=lambda g: abs(g.get('ou_edge_num') or 0.0), reverse=True)
            elif sort_by == 'ats_edge_desc':
                cards.sort(key=lambda g: abs(g.get('ats_edge_num') or 0.0), reverse=True)
            else:
                cards.sort(key=lambda g: (g.get('sort_ts') is None, g.get('sort_ts') or 0.0))
        except Exception:
            pass
        # Apply cap unless full
        if not show_full:
            try:
                finals = [c for c in cards if c.get('actual_home_points') is not None and c.get('actual_away_points') is not None]
                upcoming = [c for c in cards if c not in finals]
                cards = finals + upcoming[:80]
            except Exception:
                cards = cards[:80]
        payload = {'week': int(sel_week) if sel_week is not None else None, 'count': len(cards), 'results': cards}
        try:
            GAME_CARDS_CACHE[key] = payload
        except Exception:
            pass
        return jsonify(payload), 200
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5051))
    debug_flag = os.environ.get('DEBUG', '0') == '1'
    try:
        rules = list(app.url_map.iter_rules())
        print(f"[app] Starting with {len(rules)} routes registered")
        for r in sorted(rules, key=lambda x: x.rule)[:120]:
            print('[route]', r.rule, 'methods=', ','.join(sorted(m for m in r.methods if m not in ('HEAD','OPTIONS'))))
    except Exception as _e:
        print('[app] Route dump failed', _e)
    app.run(host='0.0.0.0', port=port, debug=debug_flag)
