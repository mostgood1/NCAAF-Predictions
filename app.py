import os
# noop: touch for deploy trigger (no behavior change)

# -------------------- Build / Version Introspection --------------------
import time as _time
def _get_git_commit() -> str:
    """Best-effort commit detection.
    Prefer Render's env vars when available; otherwise fall back to scanning .git.
    """
    try:
        # 1) Render environment variables (most reliable in production)
        for key in ('RENDER_GIT_COMMIT', 'RENDER_GIT_COMMIT_SHA', 'RENDER_GIT_COMMIT_ID', 'GIT_COMMIT', 'COMMIT_SHA'):
            val = os.environ.get(key)
            if val and isinstance(val, str) and len(val) >= 7:
                return val.strip()[:40]
        # 2) Local dev: scan .git directory
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
from flask import Flask, render_template_string, request, redirect, url_for, jsonify, make_response, send_from_directory
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

# Route registration filter: relaxed by default to avoid accidental 404s in prod
_orig_add_url_rule = app.add_url_rule
def _filtered_add_url_rule(rule, endpoint=None, view_func=None, provide_automatic_options=None, **options):
    try:
        # If LIMIT_ROUTES is disabled (default), pass-through and register all routes.
        limit_flag = str(os.environ.get('LIMIT_ROUTES', '0')).strip().lower()
        if limit_flag in ('0', 'false', 'no', ''):
            return _orig_add_url_rule(rule, endpoint=endpoint, view_func=view_func,
                                      provide_automatic_options=provide_automatic_options, **options)

        # Otherwise, keep a small allowlist (main pages + diagnostics + static)
        allowed = {
            '/',
            '/recommendations',
            '/recommendations/debug',
            '/favicon.ico',
            '/which-app',
            '/deploy-info',
            '/routes',
            '/healthz',
            '/recommendations/',
            '/recommendations/debug/',
        }
        if rule in allowed or (isinstance(rule, str) and rule.startswith('/static')):
            return _orig_add_url_rule(rule, endpoint=endpoint, view_func=view_func,
                                      provide_automatic_options=provide_automatic_options, **options)
        # Skip registration for all other routes when limiting is enabled
        return None
    except Exception:
        # On any error, register the route to avoid breaking the app
        return _orig_add_url_rule(rule, endpoint=endpoint, view_func=view_func,
                                  provide_automatic_options=provide_automatic_options, **options)
app.add_url_rule = _filtered_add_url_rule

# Global 500 handler to avoid raw 500s on key pages (especially /recommendations)
@app.errorhandler(500)
def _handle_500(e):
    try:
        # If the failing request targets recommendations, try a safe redirect to log source
        from flask import request as _rq, redirect as _redir, url_for as _url
        path = _rq.path or ''
        if path.rstrip('/') == '/recommendations':
            try:
                q = dict(_rq.args)
                if (q.get('source') or '').lower() != 'log':
                    q['source'] = 'log'
                    return _redir(_url('recommendations_page', **q)), 302
            except Exception:
                pass
        # Otherwise return a compact diagnostics response instead of a 500
        body = f"Internal error. Build {BUILD_TIME} • Commit {BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown'}\n"
        return body, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception:
        return "Internal error", 200, {'Content-Type': 'text/plain; charset=utf-8'}

# Removed: /api/ping

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

# Global security headers: upgrade insecure requests to avoid mixed-content issues with legacy asset URLs
@app.after_request
def _add_security_headers(resp):
    try:
        # In instructive browsers, this will automatically convert http:// to https:// for subresources
        existing_csp = resp.headers.get('Content-Security-Policy', '')
        if 'upgrade-insecure-requests' not in existing_csp:
            csp_val = (existing_csp + ('; ' if existing_csp else '') + 'upgrade-insecure-requests').strip('; ')
            resp.headers['Content-Security-Policy'] = csp_val
    except Exception:
        pass
    return resp

# Lightweight health check with build metadata
@app.route('/healthz')
def _healthz():
    try:
        from flask import jsonify as _jsonify
        return _jsonify({
            'status': 'ok',
            'build_time': BUILD_TIME,
            'commit': BUILD_COMMIT,
            'service': 'ncaaf-compare'
        }), 200
    except Exception:
        # Minimal fall-back response if jsonify import fails for any reason
        return f"ok {BUILD_TIME} {BUILD_COMMIT}", 200, {'Content-Type': 'text/plain; charset=utf-8'}

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
# Hide refresh/admin controls unconditionally now that most endpoints are removed
HIDE_REFRESH = True
_REFRESH_LOCK = threading.Lock()

# Simple in-memory cache for game cards API (invalidated on prediction reload)
GAME_CARDS_CACHE = {}
_RECOMMENDATIONS_CACHE = {
    # keys: ('compute', week) -> {'ts': float, 'enriched': list}
    #       ('log', week)     -> {'mtime': float, 'enriched': list}
}

# Dynamic total-points uncertainty (std) estimated from completed games
_TOTAL_POINTS_STD_2025 = None
def _recompute_total_points_std():
    """Compute std of (actual_total - predicted_total) for completed 2025 games.
    Prefer model_* predictions when available. Clamp to a reasonable range to avoid instability.
    """
    global _TOTAL_POINTS_STD_2025
    try:
        df = pred_df[(pred_df.get('season', 0) == 2025)].copy()
        if df.empty:
            _TOTAL_POINTS_STD_2025 = 12.0
            return _TOTAL_POINTS_STD_2025
        def _p_total(r):
            try:
                mh = r.get('model_home_points'); ma = r.get('model_away_points')
                ph = r.get('predicted_home_points'); pa = r.get('predicted_away_points')
                if pd.notna(mh) and pd.notna(ma):
                    return float(mh) + float(ma)
                if pd.notna(ph) and pd.notna(pa):
                    return float(ph) + float(pa)
            except Exception:
                return None
            return None
        df = df[(df['actual_home_points'].notna()) & (df['actual_away_points'].notna())].copy()
        if df.empty:
            _TOTAL_POINTS_STD_2025 = 12.0
            return _TOTAL_POINTS_STD_2025
        df['_pred_total'] = df.apply(_p_total, axis=1)
        df['_act_total'] = pd.to_numeric(df['actual_home_points'], errors='coerce') + pd.to_numeric(df['actual_away_points'], errors='coerce')
        errs = pd.to_numeric(df['_act_total'] - df['_pred_total'], errors='coerce')
        errs = errs.dropna()
        if len(errs) < 10:
            _TOTAL_POINTS_STD_2025 = 12.0
        else:
            try:
                val = float(errs.std())
            except Exception:
                val = 12.0
            # Clamp to a sane window
            if not math.isfinite(val) or val <= 6:
                val = 10.0
            elif val > 22:
                val = 22.0
            _TOTAL_POINTS_STD_2025 = val
        return _TOTAL_POINTS_STD_2025
    except Exception:
        _TOTAL_POINTS_STD_2025 = 12.0
        return _TOTAL_POINTS_STD_2025

def _get_total_points_std():
    global _TOTAL_POINTS_STD_2025
    try:
        if _TOTAL_POINTS_STD_2025 is None:
            return _recompute_total_points_std()
        return _TOTAL_POINTS_STD_2025
    except Exception:
        return 12.0

def _calibrate_win_prob(p):
    """Fallback calibration (identity clamp) if real calibration artifacts absent due to earlier truncation."""
    try:
        x = float(p)
        if x < 0: x = 0.0
        if x > 1: x = 1.0
        return x
    except Exception:
        return None

# --- Win probability tuning knobs (env-overridable) ---
try:
    _SIGMA_MARGIN_MIN = float(os.environ.get('SIGMA_MARGIN_MIN', 12.0))
    _SIGMA_MARGIN_MAX = float(os.environ.get('SIGMA_MARGIN_MAX', 28.0))
    _Z_TEMPERATURE_EARLY = float(os.environ.get('Z_TEMPERATURE_EARLY', 1.5))   # weeks <= 5
    _Z_TEMPERATURE_LATE = float(os.environ.get('Z_TEMPERATURE_LATE', 1.25))   # weeks > 5
    _Z_CLAMP = float(os.environ.get('Z_CLAMP', 2.75))
except Exception:
    _SIGMA_MARGIN_MIN = 12.0
    _SIGMA_MARGIN_MAX = 28.0
    _Z_TEMPERATURE_EARLY = 1.5
    _Z_TEMPERATURE_LATE = 1.25
    _Z_CLAMP = 2.75

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

def _infer_current_week(default: int | None = None) -> int | None:
    """Infer the current (upcoming) week using earliest game date per week relative to today.
    Strategy: choose the smallest week whose earliest game date is >= (today - 2 days).
    Fallback to the min available week, else provided default.
    """
    try:
        import datetime as _dt
        weeks = sorted(pred_df['week'].dropna().unique())
        if not weeks:
            return default
        today = _dt.date.today()
        week_min_dates = {}
        if 'start_date' in pred_df.columns:
            tmp = pred_df[['week','start_date']].dropna().copy()
            tmp['start_dt'] = pd.to_datetime(tmp['start_date'], errors='coerce', utc=True)
            tmp = tmp.dropna(subset=['start_dt'])
            for w, grp in tmp.groupby('week'):
                try:
                    week_min_dates[int(w)] = grp['start_dt'].min().date()
                except Exception:
                    continue
        candidate_weeks = [w for w,d in week_min_dates.items() if d >= (today - _dt.timedelta(days=2))]
        if candidate_weeks:
            return min(candidate_weeks)
        # Fallback to min week
        try:
            return int(min(weeks))
        except Exception:
            return default
    except Exception:
        return default

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
try:
    _recompute_total_points_std()
except Exception:
    pass

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
        # Ensure external asset URLs are safe for HTTPS contexts
        def _to_https(url: str | None) -> str:
            try:
                if url is None:
                    return ''
                s = str(url).strip()
                if not s:
                    return ''
                if s.startswith('//'):
                    return 'https:' + s
                if s.startswith('http://'):
                    return 'https://' + s[len('http://'):]
                return s
            except Exception:
                return ''
        def clean(v):
            try:
                import pandas as _pd
                return '' if _pd.isna(v) else v
            except Exception:
                return v if v is not None else ''
        return {
            'logo': _to_https(clean(row.iloc[0].get('logo', ''))),
            'color': clean(row.iloc[0].get('color', '')),
            'alt_color': clean(row.iloc[0].get('alt_color', ''))
        }
    return {'logo': '', 'color': '', 'alt_color': ''}

# --- Color utilities for safe contrast on team-name chips ---
def _normalize_hex_color(s: str | None) -> str | None:
    try:
        if not s:
            return None
        val = str(s).strip().lower()
        if val in ('none', 'null', 'nan'):
            return None
        # Accept forms like '#abc', 'abc', '#aabbcc', 'aabbcc'
        if val.startswith('#'):
            val = val[1:]
        if len(val) not in (3, 6) or any(ch not in '0123456789abcdef' for ch in val):
            return None
        if len(val) == 3:
            val = ''.join(ch * 2 for ch in val)
        return f"#{val}"
    except Exception:
        return None

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    try:
        h = hex_color.lstrip('#')
        if len(h) == 3:
            h = ''.join(ch * 2 for ch in h)
        if len(h) != 6:
            return None
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
        return (r, g, b)
    except Exception:
        return None

def _ideal_text_color_for_bg(bg_hex: str | None) -> str:
    """Return '#fff' or '#111' depending on background brightness using YIQ.
    Uses threshold ~186 on [0,255] scale. Falls back to dark text.
    """
    try:
        if not bg_hex:
            return '#111'
        rgb = _hex_to_rgb(bg_hex)
        if not rgb:
            return '#111'
        r, g, b = rgb
        yiq = (r * 299 + g * 587 + b * 114) / 1000.0
        # If very close to mid (e.g., golds), prefer darker text for legibility
        return '#111' if yiq >= 170 else '#fff'
    except Exception:
        return '#111'

# Load betting lines
lines_df = pd.read_csv(os.path.join(DATA_DIR, "college_football_betting_lines_last_15_years.csv")) if os.path.exists(os.path.join(DATA_DIR, "college_football_betting_lines_last_15_years.csv")) else pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
lines_index = {}
lines_index_norm = {}
def _parse_lines_field(odds_str):
    """Parse the 'lines' column which is a JSON array serialized into CSV.
    Handles typical CSV double-quote escaping (e.g., "[{""provider"":...}]") and
    gracefully falls back to ast.literal_eval. Returns a list or []."""
    try:
        # Already a list?
        if isinstance(odds_str, list):
            return odds_str
        # Must be a string to parse
        if not isinstance(odds_str, str):
            return []
        s = odds_str.strip()
        if not s:
            return []
        # Fast path: try direct JSON first
        try:
            return json.loads(s)
        except Exception:
            pass
        # CSV may have doubled quotes inside the cell; normalize them
        try:
            s2 = s.replace('""', '"')
            return json.loads(s2)
        except Exception:
            pass
        # Sometimes the whole cell is quoted, strip outer quotes then normalize
        try:
            t = s
            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1]
            t = t.replace('""', '"')
            return json.loads(t)
        except Exception:
            pass
        # Last resort: literal_eval on a cleaned string
        try:
            import ast as _ast
            return _ast.literal_eval(s)
        except Exception:
            return []
    except Exception:
        return []
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
                odds = _parse_lines_field(row.get('lines', ''))
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
    # Normalize alt colors and compute foreground text colors for better contrast
    h_bg = _normalize_hex_color(home_asset.get('alt_color')) or _normalize_hex_color(home_asset.get('color')) or '#e5e7eb'
    a_bg = _normalize_hex_color(away_asset.get('alt_color')) or _normalize_hex_color(away_asset.get('color')) or '#e5e7eb'
    h_fg = _ideal_text_color_for_bg(h_bg)
    a_fg = _ideal_text_color_for_bg(a_bg)
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
    # Parse both API and local start_date to UTC and reconcile; from week 5 onward, treat naive as UTC
    try:
        week_val_for_time = int(game_row.get('week', 0))
    except Exception:
        week_val_for_time = None
    dt_api_utc = _parse_to_utc_with_context(val_api, assume_naive='eastern', week=week_val_for_time)
    dt_sd_utc = _parse_to_utc_with_context(val_sd, assume_naive='eastern', week=week_val_for_time)
    # Prefer API kickoff time when present; fallback to start_date otherwise
    chosen_dt = dt_api_utc or dt_sd_utc

    if chosen_dt is not None:
        sort_ts = chosen_dt.timestamp()
        start_iso = chosen_dt.isoformat().replace('+00:00', 'Z')
        try:
            # Server-side initial text in UTC; client JS will replace with local time
            display_time_fallback = chosen_dt.strftime('%a, %b %d, %Y, %I:%M %p UTC')
        except Exception:
            display_time_fallback = start_iso or (val_sd or val_api or '')
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
    # Win probability: robust computation with clamp and file-prob sanity check
    p_home_win = _compute_home_win_prob(game_row)
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
    # Force displayed model total to equal the sum of the two team model values
    pred_total_adj_num = _safe_float(predicted_total_points, None)
    pred_total_pre_num = None
    if pred_total_adj_num is not None:
        try:
            # If a weather adjustment is available, expose a pre-adjusted figure for tooling/debug
            pred_total_pre_num = pred_total_adj_num - (wx_adj if wx_adj is not None else 0.0)
        except Exception:
            pred_total_pre_num = None
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
    'home_color': _normalize_hex_color(home_asset.get('color')) or '',
    'home_alt_color': h_bg,
    'home_text_color': h_fg,
    'away_logo': away_asset['logo'],
    'away_color': _normalize_hex_color(away_asset.get('color')) or '',
    'away_alt_color': a_bg,
    'away_text_color': a_fg,
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
        # Edge summary fields (duplicates numeric forms for clearer API names)
        'edge_spread': ats_edge,
        'edge_total': ou_edge,
        # edge_moneyline_ev: best expected value (per 1 unit risk) among available MLs
        'edge_moneyline_ev': None,
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

def _compute_home_win_prob(row: pd.Series) -> float | None:
    """Compute a realistic home win probability.
    Strategy:
    - Prefer margin-based probability using a per-game sigma (_get_conf_std_for_game).
    - If a file-provided model_home_win_prob exists and is close to the margin-based value (<= 0.15 abs diff),
      lightly blend it; otherwise ignore it (guards against corrupted/extreme inputs).
    - Clamp final p to [0.005, 0.995] so UI never shows 0.0%/100.0% from rounding.
    """
    try:
        # Margin: prefer point-difference first (model -> predicted), then fall back to margin fields
        def _f(x):
            return _safe_float(row.get(x))
        mh = _f('model_home_points'); ma = _f('model_away_points')
        ph = _f('predicted_home_points'); pa = _f('predicted_away_points')
        margin = None
        if mh is not None and ma is not None:
            margin = mh - ma
        elif ph is not None and pa is not None:
            margin = ph - pa
        else:
            margin = _f('model_margin')
            if margin is None:
                margin = _f('predicted_win_margin')
        if margin is None:
            return None
        sigma = _get_conf_std_for_game(row)
        # Enforce sane sigma bounds so z doesn't explode
        if sigma is None or not math.isfinite(sigma) or sigma <= 0:
            sigma = _SIGMA_MARGIN_MIN
        else:
            sigma = max(_SIGMA_MARGIN_MIN, min(_SIGMA_MARGIN_MAX, float(sigma)))
        # Apply a temperature to z to reduce early-season certainty and clamp extremes
        wk_raw = row.get('week', None)
        try:
            wk = int(wk_raw) if wk_raw is not None and str(wk_raw) != 'nan' else None
        except Exception:
            wk = None
        temp = _Z_TEMPERATURE_EARLY if (wk is None or wk <= 5) else _Z_TEMPERATURE_LATE
        z = (margin / sigma) / temp
        if z > _Z_CLAMP:
            z = _Z_CLAMP
        elif z < -_Z_CLAMP:
            z = -_Z_CLAMP
        p_margin = _phi(z)
        p_file = _safe_float(row.get('model_home_win_prob'))
        # If file prob is valid and not wildly off, blend slightly; else trust p_margin.
        if p_file is not None and 0.0 <= p_file <= 1.0:
            if abs(p_file - p_margin) <= 0.15:
                p = 0.75 * p_margin + 0.25 * p_file
            else:
                p = p_margin
        else:
            p = p_margin
        # Optional calibration hook (currently identity/clamp)
        p = _calibrate_win_prob(p) or p
        # Hard clamp to avoid 0/100% display after rounding, but leave headroom from extremes
        if p < 0.01:
            p = 0.01
        elif p > 0.99:
            p = 0.99
        return float(p)
    except Exception:
        return None

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
                    try:
                        v = float(val)
                        if not math.isfinite(v) or v <= 0:
                            raise ValueError()
                        # Clamp to global bounds
                        v = max(_SIGMA_MARGIN_MIN, min(_SIGMA_MARGIN_MAX, v))
                        return v
                    except Exception:
                        pass
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
                    # Clamp to global bounds
                    v = max(_SIGMA_MARGIN_MIN, min(_SIGMA_MARGIN_MAX, v))
                    return v
    except Exception:
        pass
    # reasonable default std for margin, clamped
    return max(_SIGMA_MARGIN_MIN, min(_SIGMA_MARGIN_MAX, 14.0))

RECS_PATH = os.path.join(DATA_DIR, "recommendations_2025.csv")
RECS_COLUMNS = [
    'timestamp','season','week','home_team','away_team','market','side','price_american','line','provider',
    'model_prob','implied_prob','edge','kelly_f','bankroll','stake','status','result','pnl'
]


def _ensure_recs_file():
    if not os.path.exists(RECS_PATH):
        pd.DataFrame(columns=RECS_COLUMNS).to_csv(RECS_PATH, index=False)

def _append_recommendations(rows: list[dict]) -> int:
    if not rows:
        return 0
    try:
        os.makedirs(os.path.dirname(RECS_PATH), exist_ok=True)
    except Exception:
        pass
    header = True
    try:
        if os.path.exists(RECS_PATH) and os.path.getsize(RECS_PATH) > 0:
            header = False
    except Exception:
        header = not os.path.exists(RECS_PATH)
    try:
        df = pd.DataFrame(rows)
        known = [c for c in RECS_COLUMNS if c in df.columns]
        extras = [c for c in df.columns if c not in known]
        ordered = known + extras
        df.to_csv(RECS_PATH, mode='a', index=False, header=header, columns=ordered)
        return len(rows)
    except Exception:
        try:
            existing = pd.read_csv(RECS_PATH) if os.path.exists(RECS_PATH) else pd.DataFrame(columns=RECS_COLUMNS)
            new_df = pd.DataFrame(rows)
            all_df = pd.concat([existing, new_df], ignore_index=True)
            all_df.to_csv(RECS_PATH, index=False)
            return len(rows)
        except Exception:
            return 0

def log_recommendations_cli(week: int | None = None, bankroll: float = 1000.0, kelly_factor: float = 0.5, ev_threshold: float = 0.02) -> dict:
    """Compute top recommendations and append them to RECS_PATH, returning a small summary.
    This mirrors /api/recommendations/simple?log=true behavior without HTTP.
    """
    try:
        recs = compute_recommendations(week=week, bankroll=bankroll, kelly_factor=kelly_factor, ev_threshold=ev_threshold)
        top = recs[:100]
        if not top:
            return {"count": 0, "logged": 0}
        _ensure_recs_file()
        ts = datetime.now(timezone.utc).isoformat()
        rows = []
        for r in top:
            rows.append({
                'timestamp': ts,
                'season': r['season'], 'week': r['week'], 'home_team': r['home_team'], 'away_team': r['away_team'],
                'market': r['market'], 'side': r['side'], 'price_american': r['price_american'], 'provider': r.get('provider'),
                'line': r.get('line', None),
                'model_prob': r['model_prob'], 'implied_prob': r['implied_prob'], 'edge': r['edge'],
                'kelly_f': r['kelly_f'], 'bankroll': bankroll, 'stake': r['stake'],
                'status': 'open', 'result': 'pending', 'pnl': 0.0
            })
        wrote = _append_recommendations(rows)
        return {"count": len(top), "logged": wrote, "path": RECS_PATH}
    except Exception as e:
        return {"error": str(e)}

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
    try:
        _recompute_total_points_std()
    except Exception:
        pass

def _refresh_schedule_kickoffs(week: int | None = None, overwrite: bool = True) -> dict:
    """Re-pull kickoff times (start_date_api) for 2025 and update the with_scores CSV.
    Primary source: CFBD /games with start_date.
    Fallback (no CFBD key): ESPN scoreboard per-date 'competitions[0].date'.
    """
    try:
        import requests
    except Exception:
        requests = None

    # Determine target CSV (prefer with_scores, else enhanced)
    global pred_path_scores, pred_path_enh
    target_path = pred_path_scores if (pred_path_scores and os.path.exists(pred_path_scores)) else pred_path_enh
    if not target_path or not os.path.exists(target_path):
        return {'step': 'schedule_refresh', 'skipped': 'no_source_csv'}

    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        return {'step': 'schedule_refresh', 'error': f'read_failed: {e}'}
    if df.empty or 'home_team' not in df.columns or 'away_team' not in df.columns:
        return {'step': 'schedule_refresh', 'skipped': 'df_invalid'}
    if 'season' in df.columns:
        try:
            df = df[df['season'] == 2025].copy()
        except Exception:
            pass
    # Ensure column exists in file (not just filtered copy)
    try:
        all_df = pd.read_csv(target_path)
        if 'start_date_api' not in all_df.columns:
            all_df['start_date_api'] = pd.NA
    except Exception:
        return {'step': 'schedule_refresh', 'error': 'cannot_prepare_output'}

    def _norm_team_base(name: str) -> str:
        try:
            s = str(name or '').strip().lower()
            s = s.replace('&', 'and').replace("ʻ", "'").replace("’", "'")
            s = s.replace("hawai'i", "hawaii")
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = re.sub(r"[^a-z0-9 '\-\(\)]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s
        except Exception:
            return str(name)

    # Build desired weeks set
    if week is None:
        try:
            weeks = sorted(int(w) for w in pd.to_numeric(df.get('week'), errors='coerce').dropna().unique())
        except Exception:
            weeks = []
    else:
        weeks = [int(week)]

    updates_map: dict[tuple[int, str, str], str] = {}

    # CFBD fetch
    api_key = os.environ.get('CFBD_API_KEY') or os.environ.get('CFBD_TOKEN') or os.environ.get('CFBD')
    if requests is not None and api_key:
        base_url = 'https://api.collegefootballdata.com/games'
        headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
        def _variants(wk: int):
            base = {'year': 2025, 'week': wk}
            combos = [
                {'seasonType': 'regular', 'division': 'fbs'},
                {'seasonType': 'regular'},
                {},
            ]
            return [dict(base, **c) for c in combos]
        try:
            target_weeks = weeks if weeks else [int(w) for w in range(0, 16)]
        except Exception:
            target_weeks = [0, 1, 2, 3, 4, 5]
        for wk in target_weeks:
            try:
                for pr in _variants(wk):
                    try:
                        resp = requests.get(base_url, headers=headers, params=pr, timeout=25)
                        if resp.status_code != 200:
                            continue
                        data = resp.json() or []
                        if not data:
                            continue
                        for g in data:
                            try:
                                ht = _norm_team_base(g.get('home_team'))
                                at = _norm_team_base(g.get('away_team'))
                                sd = g.get('start_date') or g.get('start_time') or g.get('start')
                                if not sd:
                                    continue
                                updates_map[(wk, ht, at)] = str(sd)
                            except Exception:
                                continue
                        # Stop cycling variants once we have some entries for this week
                        if any(k[0] == wk for k in updates_map.keys()):
                            break
                    except Exception:
                        continue
            except Exception:
                continue

    # ESPN fallback if no CFBD updates
    if not updates_map and requests is not None:
        try:
            unique_dates = set()
            try:
                sub = df.copy()
                if weeks:
                    sub = sub[sub['week'].isin(weeks)] if 'week' in sub.columns else sub
                if 'start_date' in sub.columns:
                    for _, r in sub.iterrows():
                        d = pd.to_datetime(r.get('start_date'), errors='coerce')
                        if pd.notna(d):
                            unique_dates.add(d.date().isoformat())
            except Exception:
                pass
            # If we still don't have dates, use an upcoming window for September
            if not unique_dates:
                unique_dates.update({'2025-09-18','2025-09-19','2025-09-20','2025-09-21'})
            for dstr in sorted(unique_dates):
                ymd = dstr.replace('-', '')
                es_urls = [
                    f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={ymd}&groups=80',
                    f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={ymd}',
                ]
                for u in es_urls:
                    try:
                        r = requests.get(u, timeout=20, headers={'Accept':'application/json','User-Agent':'Mozilla/5.0'})
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
                                    _norm_team_base(t.get('location')),
                                    _norm_team_base(t.get('shortDisplayName')),
                                    _norm_team_base(t.get('displayName')),
                                    _norm_team_base(t.get('abbreviation')),
                                ]
                            hcands = [c for c in _cands(hteam) if c]
                            acands = [c for c in _cands(ateam) if c]
                            iso = (comps.get('date') or ev.get('date') or '').strip()
                            if not iso:
                                continue
                            # ESPN iso is UTC Z; store directly
                            for ht in hcands:
                                for at in acands:
                                    updates_map[(None, ht, at)] = iso
                    except Exception:
                        continue
        except Exception:
            pass

    if not updates_map:
        return {'step': 'schedule_refresh', 'skipped': 'no_updates_from_sources'}

    # Apply to file (preserve all rows not in scope)
    try:
        out_df = pd.read_csv(target_path)
        if 'start_date_api' not in out_df.columns:
            out_df['start_date_api'] = pd.NA
        changed = 0
        for i, r in out_df.iterrows():
            try:
                if int(r.get('season', 0)) != 2025:
                    continue
            except Exception:
                continue
            if week is not None:
                try:
                    if int(r.get('week')) != int(week):
                        continue
                except Exception:
                    continue
            ht = _norm_team_base(r.get('home_team'))
            at = _norm_team_base(r.get('away_team'))
            can_keys = []
            try:
                rw = int(r.get('week')) if pd.notna(r.get('week')) else None
            except Exception:
                rw = None
            if rw is not None:
                can_keys.append((rw, ht, at))
            can_keys.append((None, ht, at))
            iso = None
            for k in can_keys:
                if k in updates_map:
                    iso = updates_map[k]
                    break
            if not iso:
                continue
            if overwrite or pd.isna(r.get('start_date_api')) or not r.get('start_date_api'):
                out_df.at[i, 'start_date_api'] = iso
                changed += 1
        if changed > 0:
            out_df.to_csv(target_path, index=False)
        return {'step': 'schedule_refresh', 'updated_rows': int(changed), 'path': target_path}
    except Exception as e:
        return {'step': 'schedule_refresh', 'error': f'apply_failed: {e}'}

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

"""
Removed legacy debug/audit API routes:
 - /api/debug-actuals
 - /api/debug-pred-source
 - /api/normalize-kickoffs
 - /api/backfill-start-date-utc
 - /logs/<filename>
 - /api/audit-kickoffs
 - /api/debug-week-counts
 - /api/debug-missing-odds
 - /api/debug-has-odds
These were used for internal diagnostics and are no longer exposed.
"""


def compute_recommendations(
    week=None,
    bankroll=1000.0,
    kelly_factor=0.5,
    ev_threshold=0.02,
    min_spread_edge_pts: float = 0.0,
    min_total_edge_pts: float = 0.0,
    min_prob: float | None = None,
    max_sigma_margin: float | None = None,
    allowed_conferences=None,
    include_completed: bool = False,
):
    """Core engine to compute EV+ recommendations, reused by API and UI.
    Optional selection filters:
      - min_spread_edge_pts: abs(model_margin - line) threshold for Spread picks
      - min_total_edge_pts: abs(model_total - OU) threshold for Total picks
      - min_prob: minimum model probability for ML side (e.g., 0.55)
      - max_sigma_margin: exclude games whose margin std (confidence) exceeds this
      - allowed_conferences: list/set or comma-separated string; keep games where either team conf is in this list
    """
    df = pred_df[(pred_df['season'] == 2025)].copy()
    # Optional upcoming-only filter
    if not include_completed:
        df = df[df['actual_home_points'].isna() & df['actual_away_points'].isna()]
    if week is not None:
        try:
            df = df[df['week'] == int(week)]
        except Exception:
            pass
    # Normalize allowed conferences to a lowercase set
    allow_set = None
    if allowed_conferences:
        if isinstance(allowed_conferences, str):
            allow_set = {s.strip().lower() for s in allowed_conferences.split(',') if s.strip()}
        else:
            try:
                allow_set = {str(s).strip().lower() for s in allowed_conferences}
            except Exception:
                allow_set = None
    recs = []
    kelly_cap = 0.10  # never stake >10% per bet
    longshot_cap_odds = 4.0  # decimal (>4.0 ~= +300)
    min_prob_for_longshot = 0.30
    for _, row in df.iterrows():
        # Conference scope filter
        if allow_set is not None:
            try:
                hc = str(row.get('home_conference','')).strip().lower(); ac = str(row.get('away_conference','')).strip().lower()
                if hc not in allow_set and ac not in allow_set:
                    continue
            except Exception:
                pass
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
        # Precompute home win prob once per row for ML usage
        p_home_model = _compute_home_win_prob(row)
        if p_home_model is None:
            try:
                p_home_model = _phi(pred_margin / sigma_m)
            except Exception:
                p_home_model = None
        # Uncertainty filter on game-level margin sigma
        try:
            if max_sigma_margin is not None and sigma_m is not None and sigma_m > float(max_sigma_margin):
                continue
        except Exception:
            pass
        sigma_t = _get_total_points_std()
        for odds in odds_list:
            provider = odds.get('provider')
            # ML
            home_ml = _safe_float(odds.get('homeMoneyline'))
            away_ml = _safe_float(odds.get('awayMoneyline'))
            if home_ml is not None:
                p_home = p_home_model
                # Only proceed if we have a valid probability
                if p_home is not None:
                    dec, _ = american_to_decimal(home_ml)
                    if dec:
                        # Side-level min prob filter
                        if not (min_prob is not None and p_home < float(min_prob)):
                            kf = None
                            # Skip extreme longshots unless model prob decent; guard against None
                            is_low_prob_longshot = (dec > longshot_cap_odds) and (p_home is not None and p_home < min_prob_for_longshot)
                            if not is_low_prob_longshot:
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
                # Derive away probability from home if available; otherwise skip ML away
                if p_home_model is not None:
                    p_away = max(0.0, 1.0 - p_home_model)
                else:
                    p_away = None
                if p_away is not None:
                    dec, _ = american_to_decimal(away_ml)
                    if dec:
                        if not (min_prob is not None and p_away < float(min_prob)):
                            kf = None
                            is_low_prob_longshot = (dec > longshot_cap_odds) and (p_away is not None and p_away < min_prob_for_longshot)
                            if not is_low_prob_longshot:
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
            if spread_val is not None and sigma_m not in (None, 0, 0.0) and not (isinstance(sigma_m, float) and math.isnan(sigma_m)):
                p_home_cover = _phi((pred_margin - spread_val) / sigma_m)
                ev_home = p_home_cover * (dec_110 - 1) - (1 - p_home_cover)
                kf_home = min(kelly_fraction(p_home_cover, dec_110), kelly_cap)
                # Edge points filter for spread
                spread_dist_ok = True
                try:
                    if min_spread_edge_pts and abs(pred_margin - spread_val) < float(min_spread_edge_pts):
                        spread_dist_ok = False
                except Exception:
                    pass
                if spread_dist_ok and ev_home > ev_threshold and kf_home > 0:
                    stake = round(bankroll * kf_home * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Spread', 'side': 'Home', 'provider': provider, 'price_american': -110, 'model_prob': round(p_home_cover,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_home,4), 'kelly_f': round(kf_home,4), 'stake': stake, 'line': spread_val})
                p_away_cover = 1 - p_home_cover
                ev_away = p_away_cover * (dec_110 - 1) - (1 - p_away_cover)
                kf_away = min(kelly_fraction(p_away_cover, dec_110), kelly_cap)
                if spread_dist_ok and ev_away > ev_threshold and kf_away > 0:
                    stake = round(bankroll * kf_away * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Spread', 'side': 'Away', 'provider': provider, 'price_american': -110, 'model_prob': round(p_away_cover,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_away,4), 'kelly_f': round(kf_away,4), 'stake': stake, 'line': spread_val})
            if ou_val is not None:
                p_over = 1 - _phi((ou_val - pred_total) / sigma_t)
                ev_over = p_over * (dec_110 - 1) - (1 - p_over)
                kf_over = min(kelly_fraction(p_over, dec_110), kelly_cap)
                # Edge points filter for total
                total_dist_ok = True
                try:
                    if min_total_edge_pts and abs(pred_total - ou_val) < float(min_total_edge_pts):
                        total_dist_ok = False
                except Exception:
                    pass
                if total_dist_ok and ev_over > ev_threshold and kf_over > 0:
                    stake = round(bankroll * kf_over * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Total', 'side': 'Over', 'provider': provider, 'price_american': -110, 'model_prob': round(p_over,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_over,4), 'kelly_f': round(kf_over,4), 'stake': stake, 'line': ou_val})
                p_under = 1 - p_over
                ev_under = p_under * (dec_110 - 1) - (1 - p_under)
                kf_under = min(kelly_fraction(p_under, dec_110), kelly_cap)
                if total_dist_ok and ev_under > ev_threshold and kf_under > 0:
                    stake = round(bankroll * kf_under * kelly_factor, 2)
                    recs.append({'season': int(row['season']), 'week': int(row['week']), 'home_team': row['home_team'], 'away_team': row['away_team'], 'market': 'Total', 'side': 'Under', 'provider': provider, 'price_american': -110, 'model_prob': round(p_under,4), 'implied_prob': round(1/(dec_110),4), 'edge': round(ev_under,4), 'kelly_f': round(kf_under,4), 'stake': stake, 'line': ou_val})
    recs.sort(key=lambda x: x['edge'], reverse=True)
    return recs

def _compute_confidence_tier(rec: dict, game_row: pd.Series | None = None) -> tuple[str, float]:
    """Richer confidence tier using EV, Kelly, prob, market context, odds, and uncertainty.
    Returns (tier, numeric_score).
    """
    # Base fields
    e = _safe_float(rec.get('edge'), 0.0) or 0.0
    k = _safe_float(rec.get('kelly_f'), 0.0) or 0.0
    p = _safe_float(rec.get('model_prob'), 0.0) or 0.0
    market = str(rec.get('market', '') or '')
    price = _safe_float(rec.get('price_american'), None)
    dec_odds, _ = american_to_decimal(price) if price is not None else (None, None)
    line = _safe_float(rec.get('line'), None)

    # Uncertainty (margin sigma) and model edges vs book lines
    sigma_m = None
    spread_edge_pts = None
    total_edge_pts = None
    try:
        if isinstance(game_row, pd.Series):
            sigma_m = _get_conf_std_for_game(game_row)
            if market == 'Spread' and line is not None:
                pred_margin = _safe_float(game_row.get('model_margin'))
                if pred_margin is None:
                    pred_margin = _safe_float(game_row.get('predicted_win_margin'))
                if pred_margin is None:
                    mh = _safe_float(game_row.get('model_home_points')) or _safe_float(game_row.get('predicted_home_points'))
                    ma = _safe_float(game_row.get('model_away_points')) or _safe_float(game_row.get('predicted_away_points'))
                    if mh is not None and ma is not None:
                        pred_margin = mh - ma
                if pred_margin is not None:
                    spread_edge_pts = abs(pred_margin - line)
            if market == 'Total' and line is not None:
                mtot = _safe_float(game_row.get('model_total_points'))
                if mtot is None:
                    mh = _safe_float(game_row.get('model_home_points')) or _safe_float(game_row.get('predicted_home_points'))
                    ma = _safe_float(game_row.get('model_away_points')) or _safe_float(game_row.get('predicted_away_points'))
                    if mh is not None and ma is not None:
                        mtot = mh + ma
                if mtot is not None:
                    total_edge_pts = abs(mtot - line)
    except Exception:
        pass

    score = 0.0
    # EV weight
    if e >= 0.050:
        score += 2.0
    elif e >= 0.035:
        score += 1.0
    elif e >= 0.025:
        score += 0.5

    # Kelly fraction weight
    if k >= 0.030:
        score += 2.0
    elif k >= 0.020:
        score += 1.0
    elif k >= 0.010:
        score += 0.5

    # Probability thresholds by market
    m = market.lower()
    if m == 'ml' or m == 'moneyline' or m == 'money line' or m == 'money_line':
        if p >= 0.62:
            score += 2.0
        elif p >= 0.58:
            score += 1.0
        elif p >= 0.55:
            score += 0.5
    else:  # spread or total style
        if p >= 0.56:
            score += 1.0
        elif p >= 0.53:
            score += 0.5

    # Uncertainty (lower sigma is better)
    if sigma_m is not None:
        try:
            sm = float(sigma_m)
            if sm <= 11.0:
                score += 1.0
            elif sm <= 14.0:
                score += 0.5
            elif sm > 18.0:
                score -= 1.0
        except Exception:
            pass

    # Edge points weight for spread and total
    if spread_edge_pts is not None:
        if spread_edge_pts >= 3.0:
            score += 1.0
        elif spread_edge_pts >= 2.0:
            score += 0.5
    if total_edge_pts is not None:
        if total_edge_pts >= 5.0:
            score += 1.0
        elif total_edge_pts >= 3.0:
            score += 0.5

    # Longshot penalty (very big prices with low model prob)
    try:
        if dec_odds is not None and dec_odds > 4.0 and p < 0.35:
            score -= 1.0
    except Exception:
        pass

    # Map to tiers
    if score >= 4.0:
        return 'High', score
    if score >= 2.5:
        return 'Medium', score
    return 'Low', score

def _confidence_tier(edge: float | None, kelly_f: float | None, model_prob: float | None) -> tuple[str, float]:
    """Backward-compatible wrapper used in a few places that only have scalars."""
    rec = {
        'edge': edge,
        'kelly_f': kelly_f,
        'model_prob': model_prob,
        # Default to ML when unknown; produces reasonable tiers
        'market': 'ML'
    }
    return _compute_confidence_tier(rec, None)

def _parse_to_utc_with_context(s_val: str | None, *, assume_naive: str = 'eastern', week: int | None = None) -> datetime | None:
    """Parse an ISO-like string to a timezone-aware UTC datetime.
    - If s_val has explicit tz (Z or offset), respect it.
    - If naive and assume_naive == 'eastern', localize to America/New_York.
    - If naive and assume_naive == 'utc', treat as UTC.
    - If week is provided and >= START_DATE_NAIVE_IS_UTC_FROM_WEEK, override naive behavior to UTC.
    """
    if not s_val:
        return None
    try:
        s = s_val.strip()
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
                dt_obj = pd.to_datetime(s, errors='coerce').to_pydatetime()
            except Exception:
                dt_obj = None
        if not dt_obj:
            return None
        # If naive, decide how to localize
        if getattr(dt_obj, 'tzinfo', None) is None:
            # From week threshold onward, treat naive as UTC to match data export behavior
            try:
                thresh = int(os.environ.get('START_DATE_NAIVE_IS_UTC_FROM_WEEK', '5'))
            except Exception:
                thresh = 5
            if week is not None and week >= thresh:
                return dt_obj.replace(tzinfo=pytz.UTC).astimezone(pytz.UTC)
            if assume_naive == 'utc':
                return dt_obj.replace(tzinfo=pytz.UTC).astimezone(pytz.UTC)
            # default: eastern
            try:
                eastern = pytz.timezone('America/New_York')
                dt_obj = eastern.localize(dt_obj)
            except Exception:
                return dt_obj.replace(tzinfo=pytz.UTC)
        return dt_obj.astimezone(pytz.UTC)
    except Exception:
        return None

def _fmt_in_zone(dt_utc: datetime | None, tz_name: str) -> str:
    try:
        if dt_utc is None:
            return ''
        tz = pytz.timezone(tz_name)
        return dt_utc.astimezone(tz).strftime('%Y-%m-%d %a %I:%M %p %Z')
    except Exception:
        try:
            return dt_utc.isoformat() if dt_utc else ''
        except Exception:
            return ''

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
    # attempt parse with context; from week threshold onward, treat naive as UTC
    # First pass: use API then start_date
    for c in candidates:
        try:
            wk = None
            try:
                wk = int(row.get('week'))
            except Exception:
                wk = None
            dt_utc = _parse_to_utc_with_context(c, assume_naive='eastern', week=wk)
            if not dt_utc:
                continue
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

# Removed route: /api/recommendations (decorator stripped)
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
        # New selection filters
        min_spread_edge_pts = float(request.args.get('min_spread_edge_pts', 0.0) or 0.0)
        min_total_edge_pts = float(request.args.get('min_total_edge_pts', 0.0) or 0.0)
        min_prob = request.args.get('min_prob')
        min_prob = float(min_prob) if (min_prob not in (None, '')) else None
        max_sigma_margin = request.args.get('max_sigma_margin')
        max_sigma_margin = float(max_sigma_margin) if (max_sigma_margin not in (None, '')) else None
        allowed_conferences = request.args.get('allowed_conferences', '')

        week_val = int(week_q) if (week_q and str(week_q).isdigit()) else None
        recs = compute_recommendations(
            week=week_val,
            bankroll=bankroll,
            kelly_factor=kelly_factor,
            ev_threshold=ev_threshold,
            min_spread_edge_pts=min_spread_edge_pts,
            min_total_edge_pts=min_total_edge_pts,
            min_prob=min_prob,
            max_sigma_margin=max_sigma_margin,
            allowed_conferences=allowed_conferences,
        )
        # Attach timing and confidence
        out = []
        # Build a quick index by (season,week,home,away) to get start time
        idx = {}
        try:
            df2025 = pred_df[(pred_df.get('season', 0) == 2025)].copy()
            if week_val is not None:
                df2025 = df2025[df2025['week']==int(week_val)]
            for _, r in df2025.iterrows():
                key = (int(r['season']), int(r['week']), str(r['home_team']), str(r['away_team']))
                idx[key] = r
        except Exception:
            idx = {}
        for rec in recs:
            key = (rec['season'], rec['week'], rec['home_team'], rec['away_team'])
            row = idx.get(key)
            tier, score = _compute_confidence_tier(rec, row)
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
        # Deduplicate: keep highest edge per (season,week,home,away,market,side)
        dedup = {}
        for r in out:
            dkey = (r.get('season'), r.get('week'), r.get('home_team'), r.get('away_team'), r.get('market'), r.get('side'))
            prev = dedup.get(dkey)
            if prev is None or (r.get('edge') or 0) > (prev.get('edge') or 0):
                dedup[dkey] = r
        out = list(dedup.values())
        # Sorting
        try:
            if sort_key == 'time':
                out.sort(key=lambda x: (x.get('sort_ts') is None, x.get('sort_ts') or 0.0))
            elif sort_key == 'confidence_desc':
                out.sort(key=lambda x: (x.get('confidence_score') or 0, x.get('edge') or 0.0), reverse=True)
            elif sort_key == 'market':
                out.sort(key=lambda x: (str(x.get('market','')), x.get('sort_ts') or 0.0))
            elif sort_key == 'bet_type':
                # Sort by bet type then side: ML/Spread/Total, then Home/Away/Over/Under
                def _bt_key(r):
                    m = str(r.get('market',''))
                    side = str(r.get('side',''))
                    m_rank = {'ML':0,'Moneyline':0,'Spread':1,'Total':2}.get(m, 3)
                    s_rank = {'Home':0,'Away':1,'Over':0,'Under':1}.get(side, 2)
                    return (m_rank, s_rank, -(r.get('edge') or 0.0))
                out.sort(key=_bt_key)
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
        return jsonify({
            'count': len(out),
            'week': week_val,
            'sort': sort_key,
            'results': out,
            'filters': {
                'min_spread_edge_pts': min_spread_edge_pts,
                'min_total_edge_pts': min_total_edge_pts,
                'min_prob': min_prob,
                'max_sigma_margin': max_sigma_margin,
                'allowed_conferences': [s.strip() for s in allowed_conferences.split(',') if s.strip()] if allowed_conferences else None,
            }
        }), 200
    except Exception as e:
        return {'error': str(e)}, 500

"""Removed: /api/admin/refresh-schedule and /api/build-calibration"""


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
        # If no explicit week chosen, pick the "current" (upcoming) week instead of earliest or last-completed.
        if week_q is None:
            try:
                # Determine current week based on earliest game date for each week.
                # Strategy: choose the smallest week whose earliest game date is >= (today - 2 days).
                today = dt.date.today()
                week_min_dates = {}
                if 'start_date' in pred_df.columns:
                    tmp = pred_df[['week','start_date']].dropna().copy()
                    # Coerce start_date to datetime safely
                    tmp['start_dt'] = pd.to_datetime(tmp['start_date'], errors='coerce', utc=True)
                    tmp = tmp.dropna(subset=['start_dt'])
                    for w, grp in tmp.groupby('week'):
                        try:
                            week_min_dates[int(w)] = grp['start_dt'].min().date()
                        except Exception:
                            continue
                candidate_weeks = [w for w,d in week_min_dates.items() if d >= (today - dt.timedelta(days=2))]
                if candidate_weeks:
                    selected_week = min(candidate_weeks)
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
        # If the chosen week has no finals yet (e.g., upcoming Week 4 on Monday),
        # fallback to the most recent week that does have finals so FINAL cards render by default.
        try:
            if selected_week is not None:
                sub_sel = pred_df[pred_df['week']==int(selected_week)]
                sel_finals = int((sub_sel['actual_home_points'].notna() & sub_sel['actual_away_points'].notna()).sum())
                if sel_finals == 0 and weeks:
                    finals_counts = {int(w): int((pred_df[pred_df['week']==w]['actual_home_points'].notna() & pred_df[pred_df['week']==w]['actual_away_points'].notna()).sum()) for w in weeks}
                    done_weeks = [w for w,c in finals_counts.items() if c>0]
                    if done_weeks:
                        selected_week = max(done_weeks)
        except Exception:
            pass
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
        # Optional matchup filter (currently only supports FBSvFBS like API endpoint)
        matchup_filter = request.args.get('matchup','').strip().lower()
        if matchup_filter == 'fbsvfbs' and {'home_conference','away_conference'}.issubset(filtered_games.columns):
            fbs_confs = {
                'acc','sec','big ten','big 12','pac 12','american','mountain west','sun belt','mac','conference usa','independent','independents','fbs independents','independent (fbs)'
            }
            fbs_indies = {'notre dame','army','navy','umass','uconn','new mexico state'}
            def _both_fbs(row):
                try:
                    hc = str(row.get('home_conference','')).strip().lower()
                    ac = str(row.get('away_conference','')).strip().lower()
                    ht = str(row.get('home_team','')).strip().lower()
                    at = str(row.get('away_team','')).strip().lower()
                    def _is_fbs(team, conf):
                        return conf in fbs_confs or team in fbs_indies
                    return _is_fbs(ht,hc) and _is_fbs(at,ac)
                except Exception:
                    return False
            filtered_games = filtered_games[filtered_games.apply(_both_fbs, axis=1)]
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
        # Odds coverage for selected week: count games with at least one real (non-synthetic) bookmaker line
        try:
            odds_with_lines_week = 0
            for _, r in week_scope_df.iterrows():
                lines = get_betting_lines(int(r.get('season', 2025)), int(r.get('week', selected_week or 0)), r.get('home_team'), r.get('away_team'))
                if lines and any(not (l.get('synthetic') or False) for l in lines):
                    odds_with_lines_week += 1
        except Exception:
            odds_with_lines_week = 0
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
    <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
    <style>
        :root { --neutral-text: #1f2937; }
        body.dark { --neutral-text: #e5e7eb; }
        html, body { height:100%; }
        body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background: #f5f8fc; margin: 0; padding: 0 0 40px; -webkit-font-smoothing: antialiased; color:#0f172a; line-height:1.5; }
        .container { max-width: 1100px; margin: 8px auto 0; background: #fff; border-radius: 14px; box-shadow: 0 6px 18px rgba(0,0,0,0.10); padding: 18px 20px 26px; }
        h2 { text-align: center; color: #1f2937; margin-bottom: 24px; font-weight:700; }
        form { display: flex; flex-direction: column; gap: 16px; margin-bottom: 32px; }
        /* Centered filter bar */
    .filterbar { position: static; z-index: 1; background: #fff; margin: 4px auto 12px; padding: 10px 12px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); display: flex; flex-direction: row; flex-wrap: wrap; gap: 10px 14px; align-items: center; justify-content: center; max-width: 1000px; }
    label { font-weight: 600; color: #1f2937; }
    select, button { padding: 9px 12px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 1em; color:#0f172a; background:#fff; }
        button { background: #2980b9; color: #fff; border: none; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #3498db; }

    .banner { background:#e8f1ff; border:1px solid #b6d3ff; padding:10px 14px; border-radius:8px; font-size:0.98em; color:#0f2a4a; margin: 8px 0 14px; }
    .card { background: #ffffff; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.10); padding: 14px 16px 14px; margin: 8px 0 0; border-left: 5px solid #9aa3ad; transition: box-shadow .15s ease, transform .15s ease; }
        .card:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.12); transform: translateY(-1px); }
    .card-header { display:flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom:8px; border-bottom: 1px dashed #e5e7eb; }
        .status { font-weight: 800; font-size: 0.85em; padding: 4px 10px; border-radius: 12px; letter-spacing:.2px; }
        .status.final { background:#e6f7ee; color:#166534; border:1px solid #b8e6cc; }
        .status.upcoming { background:#e8eef5; color:#1f2937; border:1px solid #cbd5e1; }
        .when { color:#1f2937; font-size: 0.95em; }

    .teams { display: grid; grid-template-columns: 1fr 60px 1fr; align-items: center; gap: 12px; }
        .team { text-align: center; }
    .team-logo { height: 64px; margin-bottom: 6px; }
    .team-name { font-weight: 800; font-size: 1.06em; padding: 5px 12px; border-radius: 8px; display: inline-block; margin-top: 2px; }
    .vs { font-size: 1.9em; color: #888; font-weight: 800; }

    .score-block { margin-top: 6px; }
    .score { font-size: 1.7em; font-weight: 900; color: #0f172a; letter-spacing:.2px; }
    .pred { font-size: 0.95em; color: #5b6470; }

    .rows { display:grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; margin-top: 12px; }
        .row { background:#fff; border:1px solid #d9dee7; border-radius:10px; padding:10px 12px; font-size:1.0em; color:#0f172a; }
        .row b { color:#0f172a; }
        .badges { display:flex; flex-wrap:wrap; gap:6px; }
    .badge { padding:2px 8px; border-radius:12px; font-size:0.84em; font-weight:800; letter-spacing:.2px; }
        .ok { background:#e6f7ee; color:#166534; border:1px solid #b8e6cc; }
        .err { background:#fde7e9; color:#b91c1c; border:1px solid #f5b5bb; }
        .push { background:#eef2f7; color:#374151; border:1px solid #d1d5db; }
        .muted { color:#5b6470; }

        .odds-toggle { margin-top: 8px; text-align:center; }
        .odds-toggle button { background:#6c5ce7; }
    .odds { margin-top:10px; }
    .odds-table { width: 100%; border-collapse: collapse; background: #fff; }
    .odds-table th, .odds-table td { padding: 10px 10px; border: 1px solid #cfd6df; text-align: center; }
    .odds-table th { background: #e5edf7; color: #1f2937; font-weight:700; }
    .odds-table tr:nth-child(even) { background: #f5f8fc; }
    .no-odds { color: #606b78; font-style: italic; }

        .topbar { position: sticky; top: 0; z-index: 120; display:flex; justify-content: space-between; align-items:center; margin-bottom: 10px; padding: 10px 8px; background: rgba(255,255,255,0.96); border-bottom: 1px solid #e5e7eb; backdrop-filter: saturate(180%) blur(8px); border-top-left-radius: 12px; border-top-right-radius: 12px; }
    .links a { color:#1b4d91; margin-left:12px; text-decoration: none; font-weight:600; }
    body.dark { background:#0f172a; color:#ffffff; }
    body.dark .container { background:#0b1220; box-shadow: 0 8px 20px rgba(0,0,0,0.5); }
    body.dark .card { background:#0f1a2b; box-shadow: 0 2px 10px rgba(0,0,0,0.6); }
    body.dark .row { background:#0b1220; border-color:#334155; }
    body.dark .row b { color:#ffffff; }
    body.dark .odds-table th { background:#1d2a44; color:#e5e7eb; }
    body.dark .odds-table tr:nth-child(even) { background:#0b1220; }
    body.dark a { color:#8ab4ff; }
    body.dark select, body.dark button { background:#1e293b; color:#ffffff; border:1px solid #475569; }
    body.dark .odds-toggle button { background:#4f46e5; border-color:#4f46e5; }
    body.dark .filterbar { background:#0b1220; }
    body.dark .team-name { text-shadow: 0 1px 1px rgba(0,0,0,0.6); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12); }
    /* Dark mode contrast improvements */
    body.dark .topbar { background: rgba(15,23,42,0.96); border-bottom-color:#334155; }
    /* Sticky subheader with date chips */
    .subheader { position: sticky; top: 52px; z-index: 110; background: rgba(255,255,255,0.96); border-bottom:1px solid #e5e7eb; backdrop-filter:saturate(180%) blur(8px); padding: 8px 6px; margin: 0 -6px 10px; border-radius: 10px; }
    .date-chips { display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:center; }
    .chip { padding:6px 10px; border-radius:999px; border:1px solid #cbd5e1; background:#f8fafc; color:#0f172a; font-weight:700; font-size:0.9rem; cursor:pointer; user-select:none; }
    .chip.active { background:#2d6cdf; border-color:#2d6cdf; color:#fff; }
    .chip.clear { background:#eef2f7; border-color:#cbd5e1; color:#111827; }
    .date-divider { grid-column: 1 / -1; font-weight:800; color:#334155; margin: 6px 0 -6px; padding: 6px 10px; border-left:4px solid #94a3b8; background:#f1f5f9; border-radius:8px; }
    body.dark .subheader { background: rgba(15,23,42,0.96); border-bottom-color:#334155; }
    body.dark .chip { background:#1e293b; border-color:#475569; color:#e2e8f0; }
    body.dark .chip.active { background:#4f46e5; border-color:#4f46e5; color:#fff; }
    body.dark .date-divider { background:#0b1220; color:#e2e8f0; border-left-color:#475569; }
    body.dark .banner { background:#0f1a2b; border-color:#334155; color:#e2e8f0; }
    body.dark .vs { color:#cbd5e1; }
    body.dark .score { color:#ffffff; }
    body.dark .pred { color:#a3b2c2; }
    body.dark .muted { color:#a3b2c2; }
    body.dark .odds-table td { background:#0f172a; color:#e2e8f0; border-color:#334155; }
    body.dark .status.upcoming { background:#1e293b; color:#e2e8f0; border-color:#475569; }
    body.dark .status.final { background:#0f2d1c; color:#34d399; border-color:#14532d; }
    body.dark .ok { background:#0f2d1c; color:#34d399; border-color:#14532d; }
    body.dark .err { background:#3b0f14; color:#f87171; border-color:#7f1d1d; }
    body.dark .push { background:#1f2937; color:#e2e8f0; border-color:#475569; }
    body.dark .team-logo { filter: drop-shadow(0 0 0.5px rgba(255,255,255,0.2)); }
    /* Force key text to white in dark mode for readability */
    body.dark h1, body.dark h2 { color:#ffffff; }
    body.dark .when { color:#ffffff; }
    body.dark .row { color:#ffffff; }
    body.dark .summary { color:#ffffff; }
    /* Allow computed contrast colors even in dark mode (do not force white) */
    .summary { display:flex; gap:16px; justify-content:center; color:#0f172a; font-weight:700; margin:10px 0 16px; }

        /* Responsive grid for cards */
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 660px) { .grid { grid-template-columns: 1fr 1fr; } }

    /* Mobile tweaks */
    @media (max-width: 660px) {
        .container { border-radius:0; box-shadow:none; padding:12px 10px 40px; }
        h2 { font-size:1.25rem; margin-top:4px; }
        .topbar { flex-wrap:wrap; gap:6px; padding:8px 6px; }
        .links { overflow-x:auto; white-space:nowrap; width:100%; font-size:0.85rem; }
        .links a { display:inline-block; padding:4px 6px; }
        .summary { flex-direction:column; gap:4px; font-size:0.9rem; }
    .teams { grid-template-columns: 1fr 40px 1fr; }
    .team-logo { height:48px; }
        .team-name { font-size:0.9rem; padding:3px 6px; }
        .score { font-size:1.2rem; }
    .card { padding:12px 12px 12px; }
        .rows { grid-template-columns:1fr; }
        .odds-table th, .odds-table td { padding:8px 6px; font-size:0.8rem; }
        button, select { font-size:0.9rem; }
        #backToTop { right:10px; bottom:10px; padding:6px 10px; }
        .filterbar { gap:6px 8px; padding:6px 6px; }
    }

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
                <a href="/">Cards</a>
                <a href="/recommendations">Recommendations</a>
            </div>
        {% if not HIDE_REFRESH %}
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <button type="button" id="refreshBtn" title="Click = full refresh; Shift+Click = quick (scores+odds)" onclick="if(window.refreshData){try{window.refreshData();}catch(e){alert('Refresh error: '+e);}}else{fetch('/api/refresh-data',{method:'POST'}).then(()=>location.reload()).catch(e=>alert('Refresh failed: '+e));}">Refresh Data</button>
            <span id="refreshStatus" style="font-size:0.95em; color:#555;"></span>
            <small class="muted">Refresh runs locally only</small>
            <button type="button" id="auditKickoffsBtn" title="Generate kickoff audit CSV and download">Audit Kickoffs</button>
            <button type="button" id="normalizeKickoffsBtn" title="Re-pull kickoff times from APIs (start_date_api)">Normalize Kickoffs</button>
        </div>
    {% endif %}
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-left:auto;">
            <button type="button" id="toggleAllOddsBtn" title="Toggle all odds for all games">Show All Odds</button>
            <button type="button" id="toggleThemeBtn" title="Toggle light/dark theme">Dark Theme</button>
        </div>
    </div> <!-- end topbar -->
        <h1 style="margin:8px 4px 14px; font-size:1.4rem;">NCAAF Betting – Cards</h1>
    <div class="banner">
        Week {{selected_week}}:
        <strong>{{finals_count_week}}</strong> finals / {{total_games_week}} games ({{finals_pct_week}} complete)
        • Odds coverage: <strong>{{odds_with_lines_week}}</strong> / {{total_games_week}} games with lines
    </div>
        <h2>2025 NCAA Football Predictions</h2>
        <div class="subheader">
            <div class="date-chips" id="dateChips">
                <span class="chip clear" data-date="">All Dates</span>
                {% for d in all_dates %}<span class="chip" data-date="{{d}}">{{d}}</span>{% endfor %}
            </div>
        </div>
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
            <button type="button" id="toggleFBSvFBSBtn" style="background:#34495e;">FBS vs FBS</button>
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
    <script>
        // Early, minimal local-time conversion to avoid leaving UTC text if later JS fails
        (function(){
            try{
                const opts = { weekday: 'short', month: 'short', day: '2-digit', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' };
                document.querySelectorAll('.local-time').forEach(el=>{
                    let s = (el.getAttribute('data-iso')||'').trim();
                    if(!s) return;
                    if(s.indexOf('T') === -1 && /^\d{4}-\d{2}-\d{2} /.test(s)) s = s.replace(' ', 'T');
                    if(!/[zZ]|[+-]\d{2}:?\d{2}$/.test(s)) s = s + 'Z';
                    const d = new Date(s);
                    if(!isNaN(d)) el.textContent = d.toLocaleString(undefined, opts);
                });
            }catch(e){}
        })();
    </script>
    {% for game_info in game_cards %}
    <div class="card" data-sort-ts="{{game_info['sort_ts'] or 0}}" data-home-win-prob="{{game_info['home_win_prob'] or 0}}" data-ou-edge="{{game_info['ou_edge_num'] or 0}}" data-ats-edge="{{game_info['ats_edge_num'] or 0}}" data-home-conf="{{game_info['home_conference']}}" data-away-conf="{{game_info['away_conference']}}" data-ats-actual="{{game_info['ats_actual_result'] or ''}}" data-ats-correct="{% if game_info['ats_correct'] is not none %}{{ 'true' if game_info['ats_correct'] else 'false' }}{% else %}{% endif %}" data-ou-actual="{{game_info['ou_actual_result'] or ''}}" data-ou-correct="{% if game_info['ou_correct'] is not none %}{{ 'true' if game_info['ou_correct'] else 'false' }}{% else %}{% endif %}" data-winner-correct="{% if game_info['correct_prediction'] is not none %}{{ 'true' if game_info['correct_prediction'] else 'false' }}{% else %}{% endif %}" style="border-left-color: {% if game_info['is_final'] %}{% if game_info['correct_prediction'] is not none %}{% if game_info['correct_prediction'] %}#2ecc71{% else %}#e74c3c{% endif %}{% else %}#95a5a6{% endif %}{% else %}#bdc3c7{% endif %};">
        <div class="card-header">
            <div class="when">Venue: {{game_info['venue']}} • <span class="local-time" data-iso="{{game_info['start_iso']}}">{{game_info['game_time']}}</span></div>
        <div class="status {% if game_info['is_final'] %}final{% else %}upcoming{% endif %}">{% if game_info['is_final'] %}FINAL{% else %}UPCOMING{% endif %}</div>
        </div>
        <div class="teams">
            <div class="team">
                <img src="{{game_info['away_logo']}}" alt="{{game_info['away_team']}} logo" class="team-logo" onerror="this.onerror=null;this.src='';"><br>
                <span class="team-name" style="background:{{game_info['away_alt_color']}};color:{{game_info['away_text_color']}};">{{game_info['away_team']}}</span>
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
                <span class="team-name" style="background:{{game_info['home_alt_color']}};color:{{game_info['home_text_color']}};">{{game_info['home_team']}}</span>
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
                            <br><b>Diff:</b> <span class="diff-val" style="font-weight:800; color:{% if game_info['total_points_diff']|float > 0 %}#0b84ff{% elif game_info['total_points_diff']|float < 0 %}#ff7f0e{% else %}var(--neutral-text){% endif %};">{{game_info['total_points_diff']}}</span>
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
        <div class="odds-toggle"><button type="button" class="toggleOddsBtn">Show Odds ({{ game_info['betting_lines']|length }})</button></div>
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
    <div style="margin:20px 0 10px; text-align:center; font-size:.85rem; color:#667;">
        <a href="/">Cards</a> | <a href="/recommendations">Recommendations</a>
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
                // Wire kickoff audit and normalize buttons
                const auditBtn = document.getElementById('auditKickoffsBtn');
                if(auditBtn){
                    auditBtn.addEventListener('click', async () => {
                        try{
                            auditBtn.disabled = true; auditBtn.textContent = 'Auditing…';
                            const r = await fetch('/api/audit-kickoffs');
                            const j = await r.json();
                            if(j && j.path){
                                const reCsv = new RegExp('kickoff_audit_[^/\\\\]+\\.csv$');
                                const m = (j.path.match(reCsv) || [null])[0];
                                if(m){ window.open('/logs/' + m, '_blank'); }
                            }
                            auditBtn.textContent = 'Audit Kickoffs'; auditBtn.disabled = false;
                        }catch(e){ alert('Audit failed: ' + e); auditBtn.textContent = 'Audit Kickoffs'; auditBtn.disabled = false; }
                    });
                }
                const normBtn = document.getElementById('normalizeKickoffsBtn');
                if(normBtn){
                    normBtn.addEventListener('click', async ()=>{
                        try{
                            normBtn.disabled = true; normBtn.textContent = 'Normalizing…';
                            const r = await fetch('/api/normalize-kickoffs', {method:'POST'});
                            if(r.ok){ alert('Kickoff times refreshed. UI will reload.'); location.reload(); }
                            else { const t = await r.text(); throw new Error(t || r.status); }
                        }catch(e){ alert('Normalize failed: ' + e); }
                        finally{ normBtn.textContent = 'Normalize Kickoffs'; normBtn.disabled = false; }
                    });
                }

                // Render game times in user's local timezone
                function applyLocalTimes(root){
                    try {
                        const opts = { weekday: 'short', month: 'short', day: '2-digit', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' };
                        (root || document).querySelectorAll('.local-time').forEach(el => {
                            let s = (el.getAttribute('data-iso') || '').trim();
                            if(!s){
                                // Fallback: try to parse existing UTC text content
                                s = (el.textContent || '').trim();
                            }
                            if(!s) return;
                            if(s.indexOf('T') === -1 && (new RegExp('^\\\d{4}-\\\d{2}-\\\d{2} ')).test(s)){ s = s.replace(' ', 'T'); }
                            // If no timezone provided, assume UTC (append Z)
                            if(!(new RegExp('[zZ]|[+\\-]\\\d{2}:?\\\d{2}$')).test(s)) s = s + 'Z';
                            let d = new Date(s);
                            if(isNaN(d)) return;
                            el.textContent = d.toLocaleString(undefined, opts);
                            el.setAttribute('data-iso', s);
                        });
                    } catch(e) { /* no-op */ }
                }
                applyLocalTimes(document);

                // Build Date dropdown & sticky chips from cards (local dates) and add day dividers
                try {
                    const dateSel = document.getElementById('date');
                    const chipsWrap = document.getElementById('dateChips');
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
                        c.setAttribute('data-local-date-label', label);
                        // Ensure time is rendered in local zone for this card
                        applyLocalTimes(c);
                    });
                    // Insert date dividers before the first card of each day (time-sorted in DOM already)
                    if(grid && cards.length){
                        // remove existing dividers
                        Array.from(grid.querySelectorAll('.date-divider')).forEach(el=>el.remove());
                        const seen = new Set();
                        const ordered = Array.from(cards).sort((a,b)=>{
                            const ta = parseFloat(a.getAttribute('data-sort-ts')||'0');
                            const tb = parseFloat(b.getAttribute('data-sort-ts')||'0');
                            return (isNaN(ta)?0:ta) - (isNaN(tb)?0:tb);
                        });
                        ordered.forEach(c=>{
                            const key = c.getAttribute('data-local-date');
                            if(!key || seen.has(key)) return;
                            seen.add(key);
                            const div = document.createElement('div');
                            div.className = 'date-divider';
                            div.setAttribute('data-date', key);
                            div.textContent = dOpts.get(key) || key;
                            grid.insertBefore(div, c);
                        });
                    }
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

                    function rebuildDateDividers(){
                        if(!grid) return;
                        // remove old dividers
                        Array.from(grid.querySelectorAll('.date-divider')).forEach(el=>el.remove());
                        // insert new ones based on visible cards in DOM order
                        const seen = new Set();
                        const visCards = Array.from(grid.querySelectorAll('.card')).filter(c=>c.style.display !== 'none');
                        visCards.forEach(c=>{
                            const key = c.getAttribute('data-local-date');
                            const label = c.getAttribute('data-local-date-label') || (dOpts.get(key) || key);
                            if(!key || seen.has(key)) return;
                            seen.add(key);
                            const div = document.createElement('div');
                            div.className = 'date-divider';
                            div.setAttribute('data-date', key);
                            div.textContent = label;
                            grid.insertBefore(div, c);
                        });
                        // hide any divider that is not followed by any visible card of same date
                        Array.from(grid.querySelectorAll('.date-divider')).forEach(div=>{
                            const k = div.getAttribute('data-date');
                            const nextCard = Array.from(grid.querySelectorAll('.card')).find(c=>c.style.display !== 'none' && c.getAttribute('data-local-date')===k);
                            div.style.display = nextCard ? '' : 'none';
                        });
                    }
                    try { window._rebuildDateDividers = rebuildDateDividers; } catch(_) {}

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
                        rebuildDateDividers();
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

                    // Build sticky date chips and bind interactions
                    if(chipsWrap && dOpts.size){
                        // If server pre-rendered chips, sync active state with current selection
                        const chips = Array.from(chipsWrap.querySelectorAll('.chip'));
                        if(chips.length <= 1){
                            chipsWrap.innerHTML = '';
                            const mkChip = (k, label)=>{
                                const s = document.createElement('span');
                                s.className = 'chip' + (k===''?' clear':'');
                                s.setAttribute('data-date', k);
                                s.textContent = label;
                                return s;
                            };
                            chipsWrap.appendChild(mkChip('', 'All Dates'));
                            Array.from(dOpts.keys()).sort().forEach(k=> chipsWrap.appendChild(mkChip(k, dOpts.get(k))));
                        }
                        function setActive(dateKey){
                            Array.from(chipsWrap.querySelectorAll('.chip')).forEach(c=>{
                                if((c.getAttribute('data-date')||'')=== (dateKey||'')) c.classList.add('active');
                                else c.classList.remove('active');
                            });
                        }
                        function scrollToDay(dateKey){
                            const target = grid && grid.querySelector(`.date-divider[data-date="${dateKey}"]`);
                            if(target){ target.scrollIntoView({behavior:'smooth', block:'start'}); }
                        }
                        const selected = sessionStorage.getItem('selectedLocalDate') || '';
                        setActive(selected);
                        chipsWrap.addEventListener('click', (ev)=>{
                            const el = ev.target.closest('.chip'); if(!el) return;
                            const k = el.getAttribute('data-date') || '';
                            // reflect to dropdown and session
                            if(dateSel){ dateSel.value = k; }
                            sessionStorage.setItem('selectedLocalDate', k);
                            applyCombinedFilters();
                            // only scroll when selecting a specific day
                            if(k){ scrollToDay(k); }
                            setActive(k);
                        });
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
                    rebuildDateDividers();
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

                // Global: Toggle all odds
                (function(){
                    const allBtn = document.getElementById('toggleAllOddsBtn');
                    function setAll(state){ // state: 'shown' | 'hidden'
                        const showAll = (state === 'shown');
                        document.querySelectorAll('.card').forEach(card=>{
                            const odds = card.querySelector('.odds');
                            const btn = card.querySelector('.toggleOddsBtn');
                            if(!odds || !btn) return;
                            odds.style.display = showAll ? '' : 'none';
                            btn.textContent = showAll ? 'Hide Odds' : 'Show Odds';
                        });
                        sessionStorage.setItem('allOddsState', state);
                        if(allBtn) allBtn.textContent = showAll ? 'Hide All Odds' : 'Show All Odds';
                    }
                    // init from sessionStorage
                    const saved = sessionStorage.getItem('allOddsState');
                    if(saved === 'shown' || saved === 'hidden'){
                        setAll(saved);
                    } else {
                        setAll('hidden'); // default
                    }
                    if(allBtn){
                        allBtn.addEventListener('click', ()=>{
                            const cur = sessionStorage.getItem('allOddsState') || 'hidden';
                            setAll(cur === 'shown' ? 'hidden' : 'shown');
                        });
                    }
                })();

                // Theme toggle (persist in localStorage)
                (function(){
                    const btn = document.getElementById('toggleThemeBtn');
                    function applyTheme(theme){
                        const b = document.body;
                        if(theme === 'dark'){ b.classList.add('dark'); }
                        else { b.classList.remove('dark'); }
                        if(btn) btn.textContent = (theme === 'dark') ? 'Light Theme' : 'Dark Theme';
                    }
                    const saved = (localStorage.getItem('theme') || '').toLowerCase();
                    applyTheme(saved === 'dark' ? 'dark' : 'light');
                    if(btn){
                        btn.addEventListener('click', ()=>{
                            const isDark = document.body.classList.contains('dark');
                            const next = isDark ? 'light' : 'dark';
                            localStorage.setItem('theme', next);
                            applyTheme(next);
                        });
                    }
                })();

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
                    const fbsBtn = document.getElementById('toggleFBSvFBSBtn');
                    if(fbsBtn){
                        // Initialize active state based on matchup param
                        try {
                            const currentUrl = new URL(window.location.href);
                            if(currentUrl.searchParams.get('matchup') === 'FBSvFBS'){
                                fbsBtn.classList.add('active');
                                fbsBtn.style.outline='2px solid #2ecc71';
                            }
                        } catch(e) {}
                        fbsBtn.addEventListener('click', ()=>{
                            const url = new URL(window.location.href);
                            if(url.searchParams.get('matchup') === 'FBSvFBS'){
                                url.searchParams.delete('matchup');
                            } else {
                                url.searchParams.set('matchup','FBSvFBS');
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
    </div>
    ''', weeks=weeks, selected_week=selected_week, all_dates=all_dates, selected_date=selected_date, show_all=show_all, hide_both_unknown=hide_both_unknown, all_conferences=pred_df['home_conference'].unique(), selected_conference=selected_conference, game_cards=game_cards, filter_type=filter_type, summary=summary, sort_by=sort_by, HIDE_REFRESH=HIDE_REFRESH, finals_count_week=finals_count_week, total_games_week=total_games_week, finals_pct_week=finals_pct_week, unknown_pending=unknown_pending, odds_with_lines_week=odds_with_lines_week, BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT, PRED_SOURCE=PRED_SOURCE)
    resp = make_response(page_html)
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# Restrict app to only the main cards page and recommendations
@app.before_request
def _limit_routes():
    try:
        p = request.path
        # Allow the two public pages and minimal static files under /static if any
        allowed = set([
            '/',
            '/recommendations',
            '/recommendations/debug',
            '/routes',
            '/healthz',
            '/which-app',
            '/deploy-info',
        ])
        if p in allowed:
            return None
        # Allow static and favicon
        if p.startswith('/static/') or p == '/favicon.ico':
            return None
        # Otherwise 404
        return ('Not Found', 404)
    except Exception:
        return None

# Removed: /conference-records route and implementation

"""Removed: /team-schedules page and its rendering."""

# -------------------- Lightweight Diagnostics --------------------
"""Removed: /version info endpoint"""

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

# Removed route: /api/data-health (decorator stripped)
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

# Removed route: /api/performance/ats-ou (decorator stripped)
def api_performance_ats_ou():
    """Return ATS and Totals hit rates for completed 2025 games.
        Optional params:
            - week: INT
            - min_spread_edge_pts: float; require abs(model_margin - line) >= threshold to include ATS sample
            - min_total_edge_pts: float; require abs(pred_total - OU) >= threshold to include Totals sample
            - max_sigma_margin: float; exclude games with margin sigma above this
            - allowed_conferences: comma-separated list; include games where either team conference is in list
    Uses available betting lines (averaged across providers) to approximate closing lines.
    """
    try:
        week_param = request.args.get('week')
        min_spread_edge = float(request.args.get('min_spread_edge_pts', 0.0) or 0.0)
        min_total_edge = float(request.args.get('min_total_edge_pts', 0.0) or 0.0)
        max_sigma_margin = request.args.get('max_sigma_margin')
        max_sigma_margin = float(max_sigma_margin) if (max_sigma_margin not in (None, '')) else None
        allowed_conferences = request.args.get('allowed_conferences', '')
        allow_set = {s.strip().lower() for s in allowed_conferences.split(',') if s.strip()} if allowed_conferences else None

        df = pred_df[(pred_df.get('season', 0) == 2025)].copy()
        if week_param and str(week_param).isdigit():
            df = df[df.get('week', 0) == int(week_param)]
        # Completed only
        df = df[df['actual_home_points'].notna() & df['actual_away_points'].notna()].copy()
        if df.empty:
            return {'count': 0, 'message': 'No completed games for selection'}, 200

        ats_wins = ats_losses = ats_pushes = 0
        ou_wins = ou_losses = ou_pushes = 0
        samples = []
        for _, r in df.iterrows():
            year = int(r.get('season', 2025)); wk = int(r.get('week', 0))
            ht = r.get('home_team'); at = r.get('away_team')
            # Conference and model context to filter by uncertainty and scope
            if allow_set is not None:
                hc = str(r.get('home_conference','')).strip().lower(); ac = str(r.get('away_conference','')).strip().lower()
                if hc not in allow_set and ac not in allow_set:
                    continue
            # Model predictions to measure distance to line
            ph = _safe_float(r.get('model_home_points')) or _safe_float(r.get('predicted_home_points'))
            pa = _safe_float(r.get('model_away_points')) or _safe_float(r.get('predicted_away_points'))
            pred_total = (ph + pa) if (ph is not None and pa is not None) else None
            pm = _safe_float(r.get('model_margin'))
            if pm is None:
                pm = _safe_float(r.get('predicted_win_margin'))
            if pm is None and ph is not None and pa is not None:
                pm = ph - pa
            # Uncertainty gate
            try:
                sig = _get_conf_std_for_game(r)
                if max_sigma_margin is not None and sig is not None and sig > max_sigma_margin:
                    continue
            except Exception:
                pass

            lines = get_betting_lines(year, wk, ht, at)
            # derive average home spread and total
            spreads = []
            totals = []
            for bl in lines:
                try:
                    s_fmt = bl.get('formattedSpread'); s_raw = bl.get('spread')
                    v = None
                    if isinstance(s_fmt, str) and s_fmt:
                        m = re.match(r"^(.*)\s+([+-]?[0-9]*\.?[0-9]+)$", s_fmt.strip())
                        if m:
                            team_label = m.group(1).strip().lower()
                            num = float(m.group(2))
                            if str(ht).strip().lower() in team_label and str(at).strip().lower() not in team_label:
                                v = num
                            elif str(at).strip().lower() in team_label and str(ht).strip().lower() not in team_label:
                                v = -num
                    if v is None and s_raw not in (None, ''):
                        v = float(s_raw)
                    if v is not None:
                        spreads.append(v)
                except Exception:
                    pass
                try:
                    ou = bl.get('overUnder')
                    if ou not in (None, ''):
                        totals.append(float(ou))
                except Exception:
                    pass

            ah = _safe_float(r.get('actual_home_points'))
            aa = _safe_float(r.get('actual_away_points'))
            if spreads:
                line = sum(spreads)/len(spreads)
                # Respect spread distance filter if provided
                if (min_spread_edge == 0.0) or (pm is not None and abs(pm - line) >= min_spread_edge):
                    margin = ah - aa
                    comp = margin + line
                    if abs(comp) < 1e-9:
                        ats_pushes += 1
                    elif comp > 0:
                        ats_wins += 1
                    else:
                        ats_losses += 1
            if totals:
                tline = sum(totals)/len(totals)
                # Respect total distance filter if provided
                if (min_total_edge == 0.0) or (pred_total is not None and abs(pred_total - tline) >= min_total_edge):
                    tot = ah + aa
                    diff = tot - tline
                    if abs(diff) < 1e-9:
                        ou_pushes += 1
                    elif diff > 0:
                        ou_wins += 1
                    else:
                        ou_losses += 1

            if len(samples) < 30:
                samples.append({'week': wk, 'away': at, 'home': ht, 'spread_avg': round(sum(spreads)/len(spreads),1) if spreads else None, 'total_avg': round(sum(totals)/len(totals),1) if totals else None})

        ats_games = ats_wins + ats_losses + ats_pushes
        ou_games = ou_wins + ou_losses + ou_pushes
        out = {
            'count': int(len(df)),
            'filters': {'min_spread_edge_pts': min_spread_edge, 'min_total_edge_pts': min_total_edge, 'max_sigma_margin': max_sigma_margin, 'allowed_conferences': list(allow_set) if allow_set else None},
            'ats': {'wins': ats_wins, 'losses': ats_losses, 'pushes': ats_pushes, 'hit_rate': round((ats_wins/(ats_wins+ats_losses)) if (ats_wins+ats_losses)>0 else 0.0, 4)},
            'totals': {'wins': ou_wins, 'losses': ou_losses, 'pushes': ou_pushes, 'hit_rate': round((ou_wins/(ou_wins+ou_losses)) if (ou_wins+ou_losses)>0 else 0.0, 4)},
            'sample': samples
        }
        return out, 200
    except Exception as e:
        return {'error': str(e)}, 500

# Removed route: /api/model-metrics (decorator stripped)
def api_model_metrics():
    if not _MODEL_META:
        _load_model_artifacts()
    if not _MODEL_META:
        return {'status':'no_model'}, 404
    meta = {k:v for k,v in _MODEL_META.items() if k not in ('home_pts_model','away_pts_model','margin_model')}
    try:
        meta['total_points_sigma_estimate'] = _get_total_points_std()
    except Exception:
        pass
    return meta

# Removed route: /api/week-status (decorator stripped)
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

# Removed route: /api/debug/game (decorator stripped)
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

# Removed route: /api/_routes (decorator stripped)
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

# Removed route: /api/env-info (decorator stripped)
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

# Removed route: /api/analysis-2025 (decorator stripped)
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


# Removed route: /analysis (decorator stripped)
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


# Removed route: /api/recommendations/simple (decorator stripped)
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
        _append_recommendations(out)

    return {'count': len(top), 'recommendations': top}, 200


# Removed route: /api/recommendations/performance (decorator stripped)
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
    try:
        # Server-rendered grouped layout mimicking external recommendations page
        # Build weeks list robustly even if column is missing or mixed type
        try:
            if isinstance(pred_df, pd.DataFrame) and 'week' in pred_df.columns:
                _wser = pd.to_numeric(pred_df['week'], errors='coerce').dropna().unique().tolist()
                weeks = sorted(int(w) for w in _wser if pd.notna(w))
            else:
                weeks = []
        except Exception:
            weeks = []
        # Inputs
        week_q = request.args.get('week') or request.form.get('week')
        sort_q = request.args.get('sort') or request.form.get('sort') or 'confidence_then_edge'
        # Short-circuit: if user explicitly requests compute, redirect to log to avoid any 500s
        # This preserves week/sort and can be relaxed later when compute UI is stable.
        try:
            _src_now = (request.args.get('source') or request.form.get('source') or '').strip().lower()
            if _src_now == 'compute':
                _params = {}
                if week_q:
                    _params['week'] = week_q
                if sort_q:
                    _params['sort'] = sort_q
                _params['source'] = 'log'
                return redirect(url_for('recommendations_page', **_params)), 302
        except Exception:
            pass
        bankroll = float(request.args.get('bankroll') or request.form.get('bankroll') or 1000)
        kelly_factor = float(request.args.get('kelly') or request.form.get('kelly') or 0.5)
        ev_threshold = float(request.args.get('ev') or request.form.get('ev') or 0.02)
        # New filter inputs with conservative defaults
        min_spread_edge_pts = float(request.args.get('min_spread_edge_pts') or request.form.get('min_spread_edge_pts') or 2.5)
        min_total_edge_pts = float(request.args.get('min_total_edge_pts') or request.form.get('min_total_edge_pts') or 3.0)
        min_prob = request.args.get('min_prob') or request.form.get('min_prob') or ''
        min_prob = float(min_prob) if str(min_prob).strip() not in ('', 'None') else None
        max_sigma_margin = request.args.get('max_sigma_margin') or request.form.get('max_sigma_margin') or ''
        max_sigma_margin = float(max_sigma_margin) if str(max_sigma_margin).strip() not in ('', 'None') else None
        # With filters hidden, default to all conferences
        allowed_conferences = request.args.get('allowed_conferences') or request.form.get('allowed_conferences') or ''
        # Optional: allow disabling augmentation of log results
        augment_flag = str(request.args.get('augment') or request.form.get('augment') or '0').strip()
        # Determine selected week (optional). If blank => auto upcoming
        sel_week = int(week_q) if (week_q and week_q.isdigit()) else None
        if sel_week is None:
            sel_week = _infer_current_week()
        # Decide data source: for past weeks with logged picks, prefer log; else compute fresh
        now_ts = time.time()
        try:
            current_wk = _infer_current_week()
        except Exception:
            current_wk = None
        # Optional override via query: source=log|compute
        src_override = (request.args.get('source') or request.form.get('source') or '').strip().lower()
        use_log = (src_override == 'log')
        recs = []
        recs_source = 'compute'
        # If RECS file has entries for the selected week (typically past weeks), load from there
        try:
            if sel_week is not None and os.path.exists(RECS_PATH):
                # Safely read log if non-empty; else treat as empty
                try:
                    if os.path.getsize(RECS_PATH) > 0:
                        df_log = pd.read_csv(RECS_PATH)
                    else:
                        df_log = pd.DataFrame()
                except Exception:
                    df_log = pd.DataFrame()
                if 'week' in df_log.columns and 'season' in df_log.columns:
                    rows_wk = df_log[(pd.to_numeric(df_log['season'], errors='coerce') == 2025) & (pd.to_numeric(df_log['week'], errors='coerce') == int(sel_week))]
                    # Prefer log if any rows exist for selected week unless the user explicitly forces compute
                    if not rows_wk.empty and src_override != 'compute':
                        use_log = True
                    else:
                        # fallback: for past weeks, prefer log; for current/future, compute
                        if not rows_wk.empty and (current_wk is not None and int(sel_week) < int(current_wk)):
                            use_log = True
                        else:
                            use_log = False
        except Exception:
            use_log = False

        enriched = None
        cache_ttl = float(os.environ.get('RECOMMENDATIONS_CACHE_TTL_SEC', '60'))
        if use_log:
            recs_source = 'log'
            try:
                mtime = os.path.getmtime(RECS_PATH)
            except Exception:
                mtime = None
            cache_key = ('log', int(sel_week))
            cached = _RECOMMENDATIONS_CACHE.get(cache_key)
            if cached and cached.get('mtime') == mtime and isinstance(cached.get('enriched'), list):
                enriched = cached['enriched']
            else:
                # Build recs from log for selected week
                try:
                    if os.path.exists(RECS_PATH) and os.path.getsize(RECS_PATH) > 0:
                        # Use only essential columns if present
                        cols = ['season','week','home_team','away_team','market','side','price_american','line','model_prob','implied_prob','edge','kelly_f','stake','provider','status','result']
                        try:
                            _hdr = pd.read_csv(RECS_PATH, nrows=1)
                            usecols = [c for c in cols if c in _hdr.columns]
                        except Exception:
                            usecols = []
                        df_log = pd.read_csv(RECS_PATH, usecols=usecols) if usecols else pd.read_csv(RECS_PATH)
                    else:
                        df_log = pd.DataFrame()
                except Exception:
                    df_log = pd.DataFrame()
                df_wk = df_log[(pd.to_numeric(df_log.get('season'), errors='coerce') == 2025) & (pd.to_numeric(df_log.get('week'), errors='coerce') == int(sel_week))] if not df_log.empty else pd.DataFrame()
                # Convert to list of rec dicts compatible with enrichment
                recs = []
                if not df_wk.empty:
                    for _, r in df_wk.iterrows():
                        recs.append({
                            'season': int(_safe_float(r.get('season'), 2025) or 2025),
                            'week': int(_safe_float(r.get('week'), sel_week) or sel_week or 0),
                            'home_team': str(r.get('home_team')),
                            'away_team': str(r.get('away_team')),
                            'market': r.get('market'),
                            'side': r.get('side'),
                            'price_american': _safe_float(r.get('price_american')),
                            'line': _safe_float(r.get('line')),
                            'model_prob': _safe_float(r.get('model_prob')),
                            'implied_prob': _safe_float(r.get('implied_prob')),
                            'edge': _safe_float(r.get('edge')),
                            'kelly_f': _safe_float(r.get('kelly_f')),
                            'stake': _safe_float(r.get('stake')),
                            'provider': r.get('provider'),
                            'status': r.get('status'),
                            'result': r.get('result'),
                        })
        else:
            # Compute fresh recommendations (use cache when available)
            wk_key = int(sel_week) if sel_week is not None else -1
            cache_key = ('compute', wk_key, bankroll, kelly_factor, ev_threshold, min_spread_edge_pts, min_total_edge_pts, float(min_prob) if min_prob is not None else None, float(max_sigma_margin) if max_sigma_margin is not None else None, str(allowed_conferences or ''))
            cached = _RECOMMENDATIONS_CACHE.get(cache_key)
            if cached and isinstance(cached.get('enriched'), list) and (now_ts - cached.get('ts', 0) <= cache_ttl):
                enriched = cached['enriched']
            else:
                try:
                    recs = compute_recommendations(
                        week=sel_week,
                        bankroll=bankroll,
                        kelly_factor=kelly_factor,
                        ev_threshold=ev_threshold,
                        min_spread_edge_pts=min_spread_edge_pts,
                        min_total_edge_pts=min_total_edge_pts,
                        min_prob=min_prob,
                        max_sigma_margin=max_sigma_margin,
                        allowed_conferences=allowed_conferences,
                    )
                except Exception as e:
                    app.logger.exception(f"compute_recommendations failed for week {sel_week}: {e}")
                    recs = []
        # Attach confidence + timing like API (limit to selected week for speed)
        idx = {}
        try:
            # Build a narrow base index with only required fields for enrichment
            if isinstance(pred_df, pd.DataFrame):
                base_cols = [c for c in ['season','week','home_team','away_team','start_date','start_date_api','actual_home_points','actual_away_points'] if c in pred_df.columns]
                base = pred_df[base_cols].copy() if base_cols else pred_df.copy()
                if 'season' in base.columns:
                    try:
                        base = base[pd.to_numeric(base['season'], errors='coerce') == 2025]
                    except Exception:
                        pass
                if sel_week is not None and 'week' in base.columns:
                    try:
                        base = base[pd.to_numeric(base['week'], errors='coerce') == int(sel_week)]
                    except Exception:
                        pass
                if not base.empty:
                    for _, r in base.iterrows():
                        try:
                            idx[(int(r['season']), int(r['week']), str(r['home_team']), str(r['away_team']))] = r
                        except Exception:
                            continue
        except Exception:
            pass
        # Build enrichment and compute result text when actuals exist (unless provided by cache)
        if enriched is None:
            enriched = []
            for rec in recs:
                key = (rec['season'], rec['week'], rec['home_team'], rec['away_team'])
                row = idx.get(key)
                tier, score = _compute_confidence_tier(rec, row)
                start_iso, sort_ts, display_time = _parse_start_ts(row) if row is not None else ('', None, '')
                # Normalize numeric fields for safe templating
                try:
                    line_num = _safe_float(rec.get('line'))
                except Exception:
                    line_num = None
                # Determine result for settled games
                result_txt = '—'
                try:
                    if row is not None and pd.notna(row.get('actual_home_points')) and pd.notna(row.get('actual_away_points')):
                        ah = float(row.get('actual_home_points'))
                        aa = float(row.get('actual_away_points'))
                        if rec.get('market') == 'ML':
                            if rec.get('side') == 'Home':
                                result_txt = 'Win' if ah > aa else ('Loss' if ah < aa else 'Push')
                            else:
                                result_txt = 'Win' if aa > ah else ('Loss' if aa < ah else 'Push')
                        elif rec.get('market') == 'Spread' and line_num is not None:
                            # Assume stored line is the home spread value
                            line = float(line_num)
                            margin = ah - aa
                            if rec.get('side') == 'Home':
                                diff = margin - line
                            else:
                                # Away spread is negative of home spread
                                diff = (aa - ah) - (-line)
                            result_txt = 'Win' if diff > 0 else ('Loss' if diff < 0 else 'Push')
                        elif rec.get('market') == 'Total' and line_num is not None:
                            total = ah + aa
                            line = float(line_num)
                            if rec.get('side') == 'Over':
                                result_txt = 'Win' if total > line else ('Loss' if total < line else 'Push')
                            else:
                                result_txt = 'Win' if total < line else ('Loss' if total > line else 'Push')
                except Exception:
                    result_txt = '—'
                # Derive a date-only string like the NFL page (YYYY-MM-DD)
                display_date = ''
                try:
                    if start_iso:
                        ds = pd.to_datetime(start_iso)
                        display_date = ds.strftime('%Y-%m-%d')
                    elif display_time:
                        display_date = str(display_time).split(' ')[0]
                except Exception:
                    display_date = display_time or ''
                enriched.append({**rec, 'line_num': line_num, 'confidence': tier, 'confidence_score': score, 'start_iso': start_iso, 'display_time': display_time, 'display_date': display_date, 'sort_ts': sort_ts, 'result_txt': result_txt})
        # Write to cache
        try:
            if recs_source == 'log':
                _RECOMMENDATIONS_CACHE[('log', int(sel_week))] = {'mtime': os.path.getmtime(RECS_PATH) if os.path.exists(RECS_PATH) else None, 'enriched': enriched}
            else:
                _RECOMMENDATIONS_CACHE[cache_key] = {'ts': now_ts, 'enriched': enriched}
        except Exception:
            pass
        # Deduplicate recommendations (server page) by (season,week,home,away,market,side) keeping highest edge
        dedup_page = {}
        for r in enriched:
            dkey = (r.get('season'), r.get('week'), r.get('home_team'), r.get('away_team'), r.get('market'), r.get('side'))
            prev = dedup_page.get(dkey)
            if prev is None or (r.get('edge') or 0) > (prev.get('edge') or 0):
                dedup_page[dkey] = r
        enriched = list(dedup_page.values())

        # If using log source and we have very few items, optionally augment with compute-based suggestions (not logged)
        more = []
        try:
            if recs_source == 'log' and len(enriched) < 20 and augment_flag != '0':
                try:
                    recs2 = compute_recommendations(
                        week=sel_week,
                        bankroll=bankroll,
                        kelly_factor=kelly_factor,
                        ev_threshold=ev_threshold,
                        min_spread_edge_pts=min_spread_edge_pts,
                        min_total_edge_pts=min_total_edge_pts,
                        min_prob=min_prob,
                        max_sigma_margin=max_sigma_margin,
                        allowed_conferences=allowed_conferences,
                    )
                except Exception:
                    recs2 = []
                # Build enrichment for recs2
                extra = []
                for rec in recs2:
                    key = (rec['season'], rec['week'], rec['home_team'], rec['away_team'])
                    row = idx.get(key)
                    tier, score = _compute_confidence_tier(rec, row)
                    start_iso, sort_ts, display_time = _parse_start_ts(row) if row is not None else ('', None, '')
                    try:
                        line_num = _safe_float(rec.get('line'))
                    except Exception:
                        line_num = None
                    display_date = ''
                    try:
                        if start_iso:
                            ds = pd.to_datetime(start_iso)
                            display_date = ds.strftime('%Y-%m-%d')
                        elif display_time:
                            display_date = str(display_time).split(' ')[0]
                    except Exception:
                        display_date = display_time or ''
                    extra.append({**rec, 'line_num': line_num, 'confidence': tier, 'confidence_score': score, 'start_iso': start_iso, 'display_time': display_time, 'display_date': display_date, 'sort_ts': sort_ts, 'result_txt': '—', '_augmented': True})
                # Exclude any already on the page by dkey
                have = {(r.get('season'), r.get('week'), r.get('home_team'), r.get('away_team'), r.get('market'), r.get('side')) for r in enriched}
                for r in extra:
                    dkey = (r.get('season'), r.get('week'), r.get('home_team'), r.get('away_team'), r.get('market'), r.get('side'))
                    if dkey not in have:
                        more.append(r)
                # Keep a sensible cap
                more = more[:50]
        except Exception:
            more = []
        # Sorting primary: confidence tier order (High, Medium, Low) then edge desc
        def _tier_rank(t: str):
            s = (t or '').lower()
            if s == 'high': return 0
            if s == 'medium': return 1
            if s == 'low': return 2
            return 3
        if sort_q == 'edge_desc':
            enriched.sort(key=lambda x: (-(x.get('edge') or 0.0)))
        elif sort_q == 'time':
            enriched.sort(key=lambda x: (x.get('sort_ts') is None, x.get('sort_ts') or 0.0))
        elif sort_q == 'market':
            # Group by market inside tiers using a stable order ML -> Spread -> Total, then EV desc
            def _mkey(r):
                m = str(r.get('market',''))
                m_rank = {'ML':0,'Moneyline':0,'Spread':1,'Total':2}.get(m, 3)
                return (m_rank, -(r.get('edge') or 0.0), r.get('sort_ts') is None, r.get('sort_ts') or 0.0)
            enriched.sort(key=_mkey)
        elif sort_q == 'bet_type':
            def _bt_key(r):
                m = str(r.get('market',''))
                side = str(r.get('side',''))
                m_rank = {'ML':0,'Moneyline':0,'Spread':1,'Total':2}.get(m, 3)
                s_rank = {'Home':0,'Away':1,'Over':0,'Under':1}.get(side, 2)
                return (m_rank, s_rank, -(r.get('edge') or 0.0))
            enriched.sort(key=_bt_key)
        else:  # confidence_then_edge default
            enriched.sort(key=lambda x: (_tier_rank(x.get('confidence')), -(x.get('edge') or 0.0)))
        # Group by tier (create 'Other' placeholder for future lower-confidence recs)
        high = [r for r in enriched if r.get('confidence') == 'High']
        medium = [r for r in enriched if r.get('confidence') == 'Medium']
        low = [r for r in enriched if r.get('confidence') == 'Low']
        other = []  # currently unused; kept to mirror external layout
        # Performance summary from logged CSV
        overall_stats = {}
        tier_stats = {}
        weekly_stats = []
        open_count = 0
        try:
            if os.path.exists(RECS_PATH) and os.path.getsize(RECS_PATH) > 0:
                perf_df = pd.read_csv(RECS_PATH)
                # Coerce numeric
                for col in ['stake','pnl','edge','kelly_f','model_prob','implied_prob','week','season']:
                    if col in perf_df.columns:
                        perf_df[col] = pd.to_numeric(perf_df[col], errors='coerce')
                # Recompute confidence if missing
                if 'confidence' not in perf_df.columns and {'edge','kelly_f','model_prob'}.issubset(perf_df.columns):
                    perf_df['confidence'] = perf_df.apply(lambda r: _confidence_tier(r.get('edge'), r.get('kelly_f'), r.get('model_prob'))[0], axis=1)
                def _agg(df_):
                    if df_.empty:
                        return {'count':0,'wins':0,'losses':0,'pushes':0,'acc':0.0,'stake':0.0,'pnl':0.0,'roi':0.0}
                    wins = int((df_['result']=='win').sum()) if 'result' in df_.columns else 0
                    losses = int((df_['result']=='loss').sum()) if 'result' in df_.columns else 0
                    pushes = int((df_['result']=='push').sum()) if 'result' in df_.columns else 0
                    staked = float(df_['stake'].sum()) if 'stake' in df_.columns else 0.0
                    pnl = float(df_['pnl'].sum()) if 'pnl' in df_.columns else 0.0
                    acc = (wins / (wins+losses)) if (wins+losses)>0 else 0.0
                    roi = (pnl / staked) if staked>0 else 0.0
                    return {'count':len(df_), 'wins':wins,'losses':losses,'pushes':pushes,'acc':acc,'stake':staked,'pnl':pnl,'roi':roi}
                overall_stats = _agg(perf_df)
                for t in ['High','Medium','Low']:
                    tier_stats[t] = _agg(perf_df[perf_df.get('confidence','')==t])
                # Weekly reconciliation (season 2025 only if present)
                dfw = perf_df.copy()
                if 'season' in dfw.columns:
                    try:
                        dfw_2025 = dfw[dfw['season']==2025]
                        if not dfw_2025.empty:
                            dfw = dfw_2025
                    except Exception:
                        pass
                # open vs closed
                try:
                    open_count = int((dfw.get('status','')=='open').sum()) if 'status' in dfw.columns else 0
                except Exception:
                    open_count = 0
                # Group by week
                if 'week' in dfw.columns:
                    try:
                        grp = dfw.dropna(subset=['week']).groupby('week')
                        stats = []
                        for wk, g in grp:
                            a = _agg(g)
                            try:
                                wk_int = int(wk)
                            except Exception:
                                wk_int = wk
                            a['week'] = wk_int
                            stats.append(a)
                        weekly_stats = sorted(stats, key=lambda x: x['week'])
                    except Exception:
                        weekly_stats = []
            else:
                overall_stats = {'count':0,'wins':0,'losses':0,'pushes':0,'acc':0.0,'stake':0.0,'pnl':0.0,'roi':0.0}
                tier_stats = {k: overall_stats for k in ['High','Medium','Low']}
                weekly_stats = []
                open_count = 0
        except Exception:
            overall_stats = {'count':0,'wins':0,'losses':0,'pushes':0,'acc':0.0,'stake':0.0,'pnl':0.0,'roi':0.0}
            tier_stats = {k: overall_stats for k in ['High','Medium','Low']}
            weekly_stats = []
            open_count = 0
        def fmt_pct(x):
            try:
                return f"{x*100:.1f}%"
            except Exception:
                return ""
        def fmt_money(x):
            try:
                return f"${x:.0f}"
            except Exception:
                return "$0"
        # Friendly label for the summary card
        if sort_q == 'edge_desc':
            sort_label = 'EV (desc)'
        elif sort_q == 'bet_type':
            sort_label = 'Bet Type (ML/Spread/Totals)'
        elif sort_q == 'market':
            sort_label = 'Market (ML/Spread/Totals)'
        elif sort_q == 'time':
            sort_label = 'Kickoff Time'
        else:
            sort_label = 'Confidence then EV'

        return render_template_string('''
    <style>
        body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background:#f5f8fc; margin:0; }
        h1 { font-size:1.6rem; margin:0 0 10px; }
        h2 { margin:18px 0 8px; font-size:1.1rem; }
        .wrap { max-width:1200px; margin:18px auto 60px; background:#fff; padding:26px 30px 34px; border-radius:14px; box-shadow:0 6px 18px rgba(0,0,0,.08);} 
        table { width:100%; border-collapse:collapse; margin-top:8px; }
        th,td { border:1px solid #e1e5ec; padding:8px 10px; font-size:.92rem; text-align:center; }
        th { background:#f0f4f9; }
        .nav { font-size:.85rem; margin-bottom:12px; text-align:right; }
    .nav a { color:#1b4d91; text-decoration:none; margin-left:10px; }
    .cards { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:10px 0 14px; }
    .card { background:#f8fafc; border:1px solid #e3eaf2; border-radius:12px; padding:12px 14px; }
    .card h3 { margin:0 0 6px; font-size:.9rem; color:#334155; }
    .card .big { font-size:1.35rem; font-weight:700; color:#0b1020; }
    .card .sub { font-size:.8rem; color:#0b1020; margin-top:2px; }
    .kpi-line { margin:4px 0; }
        .section-empty { font-size:.85rem; color:#777; margin:4px 0 14px; }
        .filters form { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:10px 0 8px; }
        select { padding:4px 6px; }
        button { padding:6px 12px; border:1px solid #2d6cdf; background:#2d6cdf; color:#fff; border-radius:6px; cursor:pointer; }
        button.secondary { background:#fff; color:#2d6cdf; }
    .conf-high { background:#eaf7ef; }
    .conf-medium { background:#fff7e6; }
    .conf-low { background:#fdecee; }
    body.dark .conf-high { background:#0f2d1c; }
    body.dark .conf-medium { background:#2a1f0a; }
    body.dark .conf-low { background:#3b0f14; }
        caption { text-align:left; font-weight:600; margin:12px 0 4px; }
    .meta-bar { font-size:.85rem; color:#555; margin-top:10px; }
    .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:.8rem; margin-left:6px; border:1px solid transparent; }
    .pill-high { background:#eaf7ef; color:#14532d; border-color:#bae6c4; }
    .pill-medium { background:#fff7e6; color:#7a4b00; border-color:#fde1a3; }
    .pill-low { background:#fdecee; color:#7f1d1d; border-color:#f5b5bc; }
        @media (max-width:900px){ th,td { font-size:.72rem; padding:4px; } }
        /* Dark theme variants */
        body.dark { background:#0f172a; color:#e2e8f0; }
        body.dark .wrap { background:#0b1220; box-shadow:0 8px 20px rgba(0,0,0,.5); }
    body.dark th { background:#13223a; color:#e2e8f0; }
    body.dark td { background:#0f172a; color:#e2e8f0; border-color:#223; }
    body.dark .card { background:#0f1a2b; border-color:#223; }
    body.dark .card .big, body.dark .card .sub, body.dark .card h3 { color:#ffffff; }
    body.dark .pill-high { background:#0f2d1c; color:#bbf7d0; border-color:#14532d; }
    body.dark .pill-medium { background:#2a1f0a; color:#fde68a; border-color:#7a4b00; }
    body.dark .pill-low { background:#3b0f14; color:#fecaca; border-color:#7f1d1d; }
    body.dark a, body.dark .nav a { color:#8ab4ff; }
    body.dark .nav { color:#cbd5e1; }
    </style>
    <div class="wrap">
    <div class="nav"><a href="/">Cards</a> | <a href="/recommendations">Recommendations</a> | <button type="button" id="toggleThemeBtn" style="margin-left:10px; padding:4px 8px; border-radius:6px;">Dark Theme</button></div>
        <h1>NCAAF Betting – Recommendations</h1>
        <div class="cards">
            <div class="card"><h3>Accuracy</h3><div class="big">{{fmt_pct(overall_stats.acc)}}</div><div class="sub">{{overall_stats.wins}}W-{{overall_stats.losses}}L{% if overall_stats.pushes %} / {{overall_stats.wins + overall_stats.losses + overall_stats.pushes}} settled{% endif %}</div></div>
            <div class="card"><h3>ROI</h3><div class="big">{{fmt_pct(overall_stats.roi)}}</div><div class="sub">Stake {{fmt_money(overall_stats.stake)}}</div></div>
            <div class="card"><h3>Profit/Loss</h3><div class="big">{{fmt_money(overall_stats.pnl)}}</div><div class="sub">Total picks: {{overall_stats.count}}</div></div>
            <div class="card"><h3>High/Med/Low</h3><div class="sub">High: {{tier_stats['High'].count}} • Med: {{tier_stats['Medium'].count}} • Low: {{tier_stats['Low'].count}}</div><div class="sub">Sorted by {{ sort_label }}</div></div>
        </div>
        <div class="filters">
            <form method="GET">
                <label>Week:
                    <select name="week">
                        <option value="">(auto)</option>
                        {% for w in weeks %}<option value="{{w}}" {% if sel_week==w %}selected{% endif %}>{{w}}</option>{% endfor %}
                    </select>
                </label>
                <label>Sort by:
                    <select name="sort">
                        <option value="confidence_then_edge" {% if sort_q=='confidence_then_edge' %}selected{% endif %}>Confidence (default)</option>
                        <option value="edge_desc" {% if sort_q=='edge_desc' %}selected{% endif %}>EV</option>
                        <option value="bet_type" {% if sort_q=='bet_type' %}selected{% endif %}>Bet Type</option>
                        <option value="market" {% if sort_q=='market' %}selected{% endif %}>Market</option>
                    </select>
                </label>
                <button type="submit">Apply</button>
                <a href="/recommendations" style="margin-left:6px; text-decoration:none;"><button type="button" class="secondary">Reset</button></a>
            </form>
        </div>

        <h2>High confidence</h2>
        {% if high %}
        <table class="conf-high"><tr><th>Game</th><th>Type</th><th>Selection</th><th>Odds</th><th>EV</th><th>Result</th><th>Date</th></tr>
            {% for r in high %}
            <tr>
                <td>{{r.away_team}} @ {{r.home_team}}</td>
                <td>{% if r.market=='ML' %}MONEYLINE{% elif r.market=='Spread' %}SPREAD{% elif r.market=='Total' %}TOTALS{% else %}{{ r.market|upper }}{% endif %}</td>
                <td>{% if r.market=='ML' %}{{ (r.home_team ~ ' ML') if r.side=='Home' else (r.away_team ~ ' ML') }}<span class="pill pill-high">High</span>{% elif r.market=='Spread' %}{{ (r.home_team if r.side=='Home' else r.away_team) }} {% if r.line_num is not none %}{{ '%+g' % r.line_num if r.side=='Home' else '%+g' % (-r.line_num) }}{% endif %} <span class="pill pill-high">High</span>{% else %}{{ r.side }} {% if r.line_num is not none %}{{ r.line_num }}{% endif %} <span class="pill pill-high">High</span>{% endif %}</td>
                <td>{{r.price_american}}</td>
                <td>{{'%0.1f'%(r.edge*100) if r.edge is not none else ''}}%</td>
                <td>{{r.result_txt}}</td>
                <td>{{r.display_date}}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}<div class="section-empty">No high confidence recommendations.</div>{% endif %}
        <h2>Medium confidence</h2>
        {% if medium %}
        <table class="conf-medium"><tr><th>Game</th><th>Type</th><th>Selection</th><th>Odds</th><th>EV</th><th>Result</th><th>Date</th></tr>
            {% for r in medium %}
            <tr>
                <td>{{r.away_team}} @ {{r.home_team}}</td>
                <td>{% if r.market=='ML' %}MONEYLINE{% elif r.market=='Spread' %}SPREAD{% elif r.market=='Total' %}TOTALS{% else %}{{ r.market|upper }}{% endif %}</td>
                <td>{% if r.market=='ML' %}{{ (r.home_team ~ ' ML') if r.side=='Home' else (r.away_team ~ ' ML') }}<span class="pill pill-medium">Medium</span>{% elif r.market=='Spread' %}{{ (r.home_team if r.side=='Home' else r.away_team) }} {% if r.line_num is not none %}{{ '%+g' % r.line_num if r.side=='Home' else '%+g' % (-r.line_num) }}{% endif %} <span class="pill pill-medium">Medium</span>{% else %}{{ r.side }} {% if r.line_num is not none %}{{ r.line_num }}{% endif %} <span class="pill pill-medium">Medium</span>{% endif %}</td>
                <td>{{r.price_american}}</td>
                <td>{{'%0.1f'%(r.edge*100) if r.edge is not none else ''}}%</td>
                <td>{{r.result_txt}}</td>
                <td>{{r.display_date}}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}<div class="section-empty">No medium confidence recommendations.</div>{% endif %}
        <h2>Low confidence</h2>
        {% if low %}
        <table class="conf-low"><tr><th>Game</th><th>Type</th><th>Selection</th><th>Odds</th><th>EV</th><th>Result</th><th>Date</th></tr>
            {% for r in low %}
            <tr>
                <td>{{r.away_team}} @ {{r.home_team}}</td>
                <td>{% if r.market=='ML' %}MONEYLINE{% elif r.market=='Spread' %}SPREAD{% elif r.market=='Total' %}TOTALS{% else %}{{ r.market|upper }}{% endif %}</td>
                <td>{% if r.market=='ML' %}{{ (r.home_team ~ ' ML') if r.side=='Home' else (r.away_team ~ ' ML') }}<span class="pill pill-low">Low</span>{% elif r.market=='Spread' %}{{ (r.home_team if r.side=='Home' else r.away_team) }} {% if r.line_num is not none %}{{ '%+g' % r.line_num if r.side=='Home' else '%+g' % (-r.line_num) }}{% endif %} <span class="pill pill-low">Low</span>{% else %}{{ r.side }} {% if r.line_num is not none %}{{ r.line_num }}{% endif %} <span class="pill pill-low">Low</span>{% endif %}</td>
                <td>{{r.price_american}}</td>
                <td>{{'%0.1f'%(r.edge*100) if r.edge is not none else ''}}%</td>
                <td>{{r.result_txt}}</td>
                <td>{{r.display_date}}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}<div class="section-empty">No low confidence recommendations.</div>{% endif %}
        <h2>Other</h2>
        {% if other %}
            <table><tr><th>Matchup</th><th>Market</th><th>Recommendation</th><th>Price</th><th>Edge</th><th>Stake</th><th>Model p</th><th>Date</th></tr>
            {% for r in other %}
            <tr>
                <td>{{r.away_team}} @ {{r.home_team}}</td>
                <td>{{r.market}}</td>
                <td>{{r.side}}{% if r.line_num is not none %} {{r.line_num}}{% endif %}{% if r.confidence %} {{r.confidence}}{% endif %}</td>
                <td>{{r.price_american}}</td>
                <td>{{'%0.1f'%(r.edge*100) if r.edge is not none else ''}}%</td>
                <td>${{'%0.2f'%r.stake}}</td>
                <td>{{'%0.1f'%(r.model_prob*100) if r.model_prob is not none else ''}}%</td>
                <td>{{r.display_time}}</td>
            </tr>
            {% endfor %}
            </table>
        {% else %}<div class="section-empty">No other recommendations.</div>{% endif %}

        {% if more %}
        <h2>Additional model suggestions (not yet logged)</h2>
        <table><tr><th>Game</th><th>Type</th><th>Selection</th><th>Odds</th><th>EV</th><th>Date</th></tr>
            {% for r in more %}
            <tr>
                <td>{{r.away_team}} @ {{r.home_team}}</td>
                <td>{% if r.market=='ML' %}MONEYLINE{% elif r.market=='Spread' %}SPREAD{% elif r.market=='Total' %}TOTALS{% else %}{{ r.market|upper }}{% endif %}</td>
                <td>{% if r.market=='ML' %}{{ (r.home_team ~ ' ML') if r.side=='Home' else (r.away_team ~ ' ML') }}{% elif r.market=='Spread' %}{{ (r.home_team if r.side=='Home' else r.away_team) }} {% if r.line_num is not none %}{{ '%+g' % r.line_num if r.side=='Home' else '%+g' % (-r.line_num) }}{% endif %}{% else %}{{ r.side }} {% if r.line_num is not none %}{{ r.line_num }}{% endif %}{% endif %} <span class="pill" style="border-color:#94a3b8;color:#334155;">Model</span></td>
                <td>{{r.price_american}}</td>
                <td>{{'%0.1f'%(r.edge*100) if r.edge is not none else ''}}%</td>
                <td>{{r.display_date}}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    <h2>Weekly reconciliation</h2>
        {% if weekly_stats %}
        <table>
            <tr><th>Week</th><th>Picks</th><th>Wins</th><th>Losses</th><th>Pushes</th><th>Accuracy</th><th>Stake</th><th>P/L</th><th>ROI</th></tr>
            {% for s in weekly_stats %}
            <tr>
                <td>Week {{s.week}}</td>
                <td>{{s.count}}</td>
                <td>{{s.wins}}</td>
                <td>{{s.losses}}</td>
                <td>{{s.pushes}}</td>
                <td>{{fmt_pct(s.acc)}}</td>
                <td>{{fmt_money(s.stake)}}</td>
                <td>{{fmt_money(s.pnl)}}</td>
                <td>{{fmt_pct(s.roi)}}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <div class="section-empty">No settled bets yet to reconcile.</div>
        {% endif %}
        <div style="margin-top:30px; font-size:.75rem; color:#666;">Generated at {{now}}. Edge = model EV (expected value) using American odds. Kelly stake capped & scaled. Times shown in original schedule timezone if available.</div>
        <div style="margin-top:6px; font-size:.7rem; color:#777;">Build {{ BUILD_TIME }} • Commit {{ BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown' }}</div>
    </div>
    <script>
    // Minimal theme toggle for this page
    (function(){
        const btn = document.getElementById('toggleThemeBtn');
        function applyTheme(theme){
            if(theme==='dark') document.body.classList.add('dark'); else document.body.classList.remove('dark');
            if(btn) btn.textContent = (theme==='dark') ? 'Light Theme' : 'Dark Theme';
        }
        const saved = (localStorage.getItem('theme')||'').toLowerCase();
        applyTheme(saved==='dark' ? 'dark' : 'light');
        if(btn){
            btn.addEventListener('click', ()=>{
                const next = document.body.classList.contains('dark') ? 'light' : 'dark';
                localStorage.setItem('theme', next);
                applyTheme(next);
            });
        }
    })();
    </script>
         ''', weeks=weeks, sel_week=sel_week, sort_q=sort_q, bankroll=bankroll, kelly_factor=kelly_factor, ev_threshold=ev_threshold,
    high=high, medium=medium, low=low, other=other, more=more,
         overall_stats=overall_stats, tier_stats=tier_stats, fmt_pct=fmt_pct, fmt_money=fmt_money, now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT,
                 min_spread_edge_pts=min_spread_edge_pts, min_total_edge_pts=min_total_edge_pts, min_prob=min_prob, max_sigma_margin=max_sigma_margin, allowed_conferences=allowed_conferences)
    except Exception as e:
        # Failsafe: return a friendly diagnostics page instead of a 500 so we can see the error in prod
        import traceback as _tb
        # Graceful fallback: if not already on source=log, redirect there to avoid user-visible 500s
        try:
            src_now = (request.args.get('source') or '').strip().lower()
            if src_now != 'log':
                # preserve week/sort when possible
                params = {}
                wq = request.args.get('week'); sq = request.args.get('sort')
                if wq: params['week'] = wq
                if sq: params['sort'] = sq
                params['source'] = 'log'
                return redirect(url_for('recommendations_page', **params)), 302
        except Exception:
            pass
        err = str(e)
        trace = _tb.format_exc()
        # keep it short to avoid huge responses
        trace_tail = trace[-4000:]
        try:
            return render_template_string('''
            <div style="font-family:Segoe UI,Arial,sans-serif; max-width:1000px; margin:20px auto; background:#fff; padding:18px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.08)">
                <h2>/recommendations failed</h2>
                <div style="margin:6px 0 12px; color:#666">This page caught an error and is showing diagnostics instead of a 500. Share this with the dev console.</div>
                <div><b>Error:</b> {{err}}</div>
                <div style="margin-top:8px"><b>Trace (tail):</b></div>
                <pre style="white-space:pre-wrap; background:#0f172a; color:#e2e8f0; padding:10px; border-radius:8px; max-height:420px; overflow:auto">{{trace}}</pre>
                <div style="margin-top:10px; color:#666; font-size:.85rem">Build {{BUILD_TIME}} • Commit {{BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown'}}</div>
                <div style="margin-top:8px"><a href="/">Cards</a></div>
            </div>
            ''', err=err, trace=trace_tail, BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT), 200
        except Exception:
            # As a last resort, return plain text so we never 500 invisibly
            body = f"/recommendations failed\nError: {err}\nTrace (tail):\n{trace_tail}\nBuild {BUILD_TIME} • Commit {BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown'}\n"
            return body, 200, {'Content-Type': 'text/plain; charset=utf-8'}
# Removed route: /recommendations/performance (decorator stripped)
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


@app.route('/recommendations/debug')
def recommendations_debug():
    """Lightweight diagnostics for the recommendations flow to avoid blind 500s."""
    info = {}
    try:
        week_q = request.args.get('week')
        sel_week = int(week_q) if (week_q and week_q.isdigit()) else None
    except Exception:
        sel_week = None
    try:
        cur_wk = _infer_current_week()
    except Exception:
        cur_wk = None
    info['current_week'] = cur_wk
    info['selected_week'] = sel_week
    try:
        info['recs_path'] = RECS_PATH
        info['recs_exists'] = os.path.exists(RECS_PATH)
        info['recs_size'] = os.path.getsize(RECS_PATH) if os.path.exists(RECS_PATH) else 0
    except Exception as e:
        info['recs_stat_error'] = str(e)
    # Predictions snapshot
    try:
        cnt_all = len(pred_df)
        cnt_2025 = int((pred_df.get('season',0)==2025).sum()) if 'season' in pred_df.columns else cnt_all
        info['pred_df_counts'] = {'all': cnt_all, 'y2025': cnt_2025}
        if sel_week is not None and 'week' in pred_df.columns:
            cnt_wk = int(((pred_df.get('season',0)==2025) & (pred_df['week']==int(sel_week))).sum())
            info['pred_df_counts']['week'] = cnt_wk
    except Exception as e:
        info['pred_df_error'] = str(e)
    # Attempt compute
    try:
        recs = compute_recommendations(
            week=sel_week,
            bankroll=float(request.args.get('bankroll') or 1000),
            kelly_factor=float(request.args.get('kelly') or 0.5),
            ev_threshold=float(request.args.get('ev') or 0.02),
            min_spread_edge_pts=float(request.args.get('min_spread_edge_pts') or 0.0),
            min_total_edge_pts=float(request.args.get('min_total_edge_pts') or 0.0),
            min_prob=(float(request.args.get('min_prob')) if (request.args.get('min_prob') not in (None, '')) else None),
            max_sigma_margin=(float(request.args.get('max_sigma_margin')) if (request.args.get('max_sigma_margin') not in (None, '')) else None),
            allowed_conferences=request.args.get('allowed_conferences',''),
        )
        info['compute'] = {'count': len(recs)}
        # include one sample for shape
        if recs:
            sample = recs[0].copy()
            # avoid dumping huge floats
            for k in ['edge','kelly_f','model_prob','implied_prob']:
                if k in sample and isinstance(sample[k], float):
                    sample[k] = round(sample[k], 4)
            info['compute']['sample'] = sample
    except Exception as e:
        import traceback
        info['compute_error'] = str(e)
        info['trace'] = traceback.format_exc()[-2000:]
    # Render minimal HTML
    rows = ''.join(f"<tr><td>{k}</td><td><pre style='white-space:pre-wrap'>{v}</pre></td></tr>" for k,v in info.items())
    return render_template_string("""
    <div style="font-family:Segoe UI,Arial,sans-serif; max-width:1100px; margin:20px auto; background:#fff; padding:18px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.08)">
      <h2>Recommendations Debug</h2>
      <div style="margin-bottom:8px"><a href="/">Cards</a> | <a href="/recommendations">Recommendations</a></div>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
        <tr><th style="text-align:left; width:240px">Key</th><th style="text-align:left">Value</th></tr>
        {{rows|safe}}
      </table>
      <div style="margin-top:10px; color:#666; font-size:.8rem">Built {{BUILD_TIME}} • Commit {{BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown'}}</div>
    </div>
    """, rows=rows, BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT)


@app.route('/routes')
def list_routes():
    try:
        rules = sorted([
            {'rule': str(r.rule), 'endpoint': str(r.endpoint), 'methods': sorted(list(getattr(r, 'methods', []) or []))}
            for r in app.url_map.iter_rules()
        ], key=lambda x: x['rule'])
    except Exception:
        rules = []
    rows = ''.join(
        f"<tr><td>{r.get('rule')}</td><td>{r.get('endpoint')}</td><td>{', '.join(r.get('methods', []))}</td></tr>"
        for r in rules
    )
    return render_template_string("""
    <div style="font-family:Segoe UI,Arial,sans-serif; max-width:900px; margin:20px auto; background:#fff; padding:18px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,.08)">
      <h2>Registered Routes</h2>
      <div style="margin-bottom:8px"><a href="/">Cards</a> | <a href="/recommendations">Recommendations</a> | <a href="/recommendations/debug">Recs Debug</a></div>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%">
        <tr><th style="text-align:left; width:300px">Rule</th><th style="text-align:left">Endpoint</th><th style="text-align:left">Methods</th></tr>
        {{rows|safe}}
      </table>
      <div style="margin-top:10px; color:#666; font-size:.8rem">Build {{BUILD_TIME}} • Commit {{BUILD_COMMIT[:8] if BUILD_COMMIT else 'unknown'}}</div>
    </div>
    """, rows=rows, BUILD_TIME=BUILD_TIME, BUILD_COMMIT=BUILD_COMMIT)


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

"""Removed: /api/refresh-data"""

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

"""Removed: /api/refresh-start (background refresh)"""

"""Removed: /api/refresh-progress"""

"""Removed: /refresh-status page"""

"""Removed: /health"""

"""Removed: /win-totals page"""

"""Removed: /api/admin/reload"""

"""Removed: /api/admin/fetch-odds"""

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

"""Removed: /api/game-cards"""

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
