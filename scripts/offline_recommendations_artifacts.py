#!/usr/bin/env python3
"""
Build offline JSON/CSV artifacts for recommendations and reconciliation.
- Per-week recommendations cache (CSV + JSON)
- Per-week odds coverage report (CSV)
- Season reconciliation summary from recommendations log (CSV + JSON)

This script imports the web app module for shared logic, so it uses
predictions, betting lines, and helper functions consistently with the UI.

Usage examples:
  python scripts/offline_recommendations_artifacts.py           # all upcoming weeks found in predictions
  python scripts/offline_recommendations_artifacts.py --week 6  # only week 6
  python scripts/offline_recommendations_artifacts.py --weeks 6,7,8

Outputs:
  data/recommendations_cache/week_<W>.csv
  data/recommendations_cache/week_<W>.json
  data/odds_coverage/week_<W>.csv
  data/recommendations_summary/reconciliation.csv
  data/recommendations_summary/reconciliation.json
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import Iterable, List, Dict, Any

import pandas as pd

# Import the app module to reuse data and helpers
import app as webapp

DATA_DIR = webapp.DATA_DIR
RECS_PATH = webapp.RECS_PATH

CACHE_DIR = os.path.join(DATA_DIR, "recommendations_cache")
COVERAGE_DIR = os.path.join(DATA_DIR, "odds_coverage")
SUMMARY_DIR = os.path.join(DATA_DIR, "recommendations_summary")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(COVERAGE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def list_target_weeks(spec: str | None) -> List[int]:
    df = webapp.pred_df
    try:
        df = df[df.get('season', 0) == 2025]
    except Exception:
        pass
    avail = sorted({int(w) for w in pd.to_numeric(df.get('week'), errors='coerce').dropna().unique()})
    if not spec:
        return avail
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return avail
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a_i, b_i = int(a), int(b)
                out.extend(list(range(min(a_i, b_i), max(a_i, b_i) + 1)))
            except Exception:
                continue
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    # keep only available
    out = [w for w in sorted(set(out)) if w in avail]
    return out


def compute_week_recommendations(week: int) -> List[Dict[str, Any]]:
    recs = webapp.compute_recommendations(
        week=week,
        bankroll=1000.0,
        kelly_factor=0.5,
        ev_threshold=0.02,
        min_spread_edge_pts=2.5,
        min_total_edge_pts=3.0,
        min_prob=None,
        max_sigma_margin=None,
        allowed_conferences=None,
    )
    # Build enrichment index limited to week
    idx = {}
    try:
        base = webapp.pred_df[(webapp.pred_df.get('season', 0) == 2025)].copy()
        base = base[base['week'] == int(week)]
        for _, r in base.iterrows():
            idx[(int(r['season']), int(r['week']), str(r['home_team']), str(r['away_team']))] = r
    except Exception:
        pass
    enriched = []
    for rec in recs:
        key = (rec['season'], rec['week'], rec['home_team'], rec['away_team'])
        row = idx.get(key)
        start_iso, sort_ts, display_time = webapp._parse_start_ts(row) if row is not None else ('', None, '')
        # Settle status if possible
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
                elif rec.get('market') == 'Spread' and rec.get('line') is not None:
                    line = float(rec.get('line'))
                    margin = ah - aa
                    diff = (margin - line) if rec.get('side') == 'Home' else ((aa - ah) - (-line))
                    result_txt = 'Win' if diff > 0 else ('Loss' if diff < 0 else 'Push')
                elif rec.get('market') == 'Total' and rec.get('line') is not None:
                    total = ah + aa
                    line = float(rec.get('line'))
                    if rec.get('side') == 'Over':
                        result_txt = 'Win' if total > line else ('Loss' if total < line else 'Push')
                    else:
                        result_txt = 'Win' if total < line else ('Loss' if total > line else 'Push')
        except Exception:
            result_txt = '—'
        # Date-only like UI
        display_date = ''
        try:
            if start_iso:
                ds = pd.to_datetime(start_iso)
                display_date = ds.strftime('%Y-%m-%d')
            elif display_time:
                display_date = str(display_time).split(' ')[0]
        except Exception:
            display_date = display_time or ''
        enriched.append({
            **rec,
            'start_iso': start_iso,
            'display_time': display_time,
            'display_date': display_date,
            'result_txt': result_txt,
            'sort_ts': sort_ts,
        })
    # Deduplicate keep highest edge per (season, week, home, away, market, side)
    dedup = {}
    for r in enriched:
        dkey = (r.get('season'), r.get('week'), r.get('home_team'), r.get('away_team'), r.get('market'), r.get('side'))
        prev = dedup.get(dkey)
        if prev is None or (r.get('edge') or 0) > (prev.get('edge') or 0):
            dedup[dkey] = r
    return list(dedup.values())


def write_week_cache(week: int, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        # still write an empty marker file for traceability
        path_json = os.path.join(CACHE_DIR, f"week_{week}.json")
        with open(path_json, 'w', encoding='utf-8') as f:
            json.dump({'week': week, 'generated_at': datetime.utcnow().isoformat() + 'Z', 'results': []}, f, indent=2)
        path_csv = os.path.join(CACHE_DIR, f"week_{week}.csv")
        pd.DataFrame(rows).to_csv(path_csv, index=False)
        return
    df = pd.DataFrame(rows)
    # Normalize column order for easy diffing
    cols = ['season','week','home_team','away_team','market','side','provider','price_american','line','model_prob','implied_prob','edge','kelly_f','stake','start_iso','display_date','result_txt']
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    path_csv = os.path.join(CACHE_DIR, f"week_{week}.csv")
    df.to_csv(path_csv, index=False)
    payload = {
        'week': week,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'count': len(rows),
        'results': df.to_dict(orient='records')
    }
    path_json = os.path.join(CACHE_DIR, f"week_{week}.json")
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def build_odds_coverage(week: int) -> pd.DataFrame:
    # For each game in predictions for the week, check if we have any lines
    dfw = webapp.pred_df[(webapp.pred_df.get('season', 0) == 2025) & (webapp.pred_df.get('week', -1) == int(week))].copy()
    rows = []
    for _, r in dfw.iterrows():
        home = r.get('home_team'); away = r.get('away_team')
        odds_list = webapp.get_betting_lines(int(r['season']), int(r['week']), home, away)
        providers = sorted({o.get('provider') for o in odds_list}) if odds_list else []
        rows.append({
            'season': int(r.get('season', 2025)),
            'week': int(r.get('week', week)),
            'home_team': home,
            'away_team': away,
            'has_odds': bool(odds_list),
            'provider_count': len(providers),
            'providers': ','.join([p for p in providers if p])
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(COVERAGE_DIR, f"week_{week}.csv")
    df.to_csv(out_csv, index=False)
    return df


def build_reconciliation_summary() -> None:
    if not os.path.exists(RECS_PATH):
        # Nothing logged yet
        df = pd.DataFrame(columns=['week','count','wins','losses','pushes','acc','stake','pnl','roi'])
        df.to_csv(os.path.join(SUMMARY_DIR, 'reconciliation.csv'), index=False)
        with open(os.path.join(SUMMARY_DIR, 'reconciliation.json'), 'w', encoding='utf-8') as f:
            json.dump({'generated_at': datetime.utcnow().isoformat() + 'Z', 'weeks': []}, f, indent=2)
        return
    recs_df = pd.read_csv(RECS_PATH)
    # Coerce numeric
    for col in ['stake','pnl','edge','kelly_f','model_prob','implied_prob','week','season']:
        if col in recs_df.columns:
            recs_df[col] = pd.to_numeric(recs_df[col], errors='coerce')
    # Fill confidence if missing
    if 'confidence' not in recs_df.columns and {'edge','kelly_f','model_prob'}.issubset(recs_df.columns):
        recs_df['confidence'] = recs_df.apply(lambda r: webapp._confidence_tier(r.get('edge'), r.get('kelly_f'), r.get('model_prob'))[0], axis=1)

    # Only season 2025 if present
    if 'season' in recs_df.columns:
        try:
            recs_df_2025 = recs_df[recs_df['season'] == 2025]
            if not recs_df_2025.empty:
                recs_df = recs_df_2025
        except Exception:
            pass

    def _agg(df_: pd.DataFrame) -> Dict[str, Any]:
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

    # Group by week
    weeks = sorted({int(w) for w in pd.to_numeric(recs_df.get('week'), errors='coerce').dropna().unique()})
    out_rows = []
    json_weeks = []
    for w in weeks:
        g = recs_df[recs_df['week'] == w]
        a = _agg(g)
        a['week'] = w
        out_rows.append(a)
        json_weeks.append({**a})
    out_csv = os.path.join(SUMMARY_DIR, 'reconciliation.csv')
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    with open(os.path.join(SUMMARY_DIR, 'reconciliation.json'), 'w', encoding='utf-8') as f:
        json.dump({'generated_at': datetime.utcnow().isoformat() + 'Z', 'weeks': json_weeks}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build offline recommendations artifacts")
    parser.add_argument('--week', type=int, help='Single week to build')
    parser.add_argument('--weeks', type=str, help='Comma/range list, e.g. 3,4,5 or 3-8; use all/* for every available week')
    args = parser.parse_args()

    weeks: List[int]
    if args.week is not None:
        weeks = [int(args.week)]
    else:
        weeks = list_target_weeks(args.weeks)

    if not weeks:
        print("No weeks selected or available; nothing to do.")
        return

    print(f"Building offline artifacts for weeks: {weeks}")
    for w in weeks:
        rows = compute_week_recommendations(w)
        write_week_cache(w, rows)
        build_odds_coverage(w)
        print(f" - Week {w}: {len(rows)} recs cached; odds coverage written.")

    build_reconciliation_summary()
    print("Reconciliation summary written.")


if __name__ == '__main__':
    main()
