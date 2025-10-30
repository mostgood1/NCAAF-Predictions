"""
Quick smoke test for daily odds update and front-end data wiring.

What it does:
- Detect upcoming week from latest predictions
- Count existing odds coverage for that week from lines CSV
- Optionally fetch fresh odds via The Odds API (if ODDS_API_KEY present)
- Merge/write updated lines CSV
- Reload app in-memory data and recompute odds coverage for selected week
- Save a compact JSON report to logs/smoke_daily_update.json

Exit codes:
 0 = ran successfully, see report
 1 = fatal error (see stderr)
"""
from __future__ import annotations
import os, sys, json, time, traceback
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT, 'data')
LOG_DIR = os.path.join(ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().isoformat() + 'Z'


def _detect_upcoming_week() -> int | None:
    try:
        # Use the same logic as fetch_2025_lines for simplicity
        import glob
        YEAR = 2025
        files = sorted(glob.glob(os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced*.csv')), key=os.path.getmtime, reverse=True)
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or 'week' not in df.columns or 'season' not in df.columns:
                continue
            sub = df[df['season'] == YEAR]
            if sub.empty:
                continue
            done = sub[sub['actual_home_points'].notna() & sub['actual_away_points'].notna()]
            if not done.empty:
                return int(done['week'].max()) + 1
            return int(pd.to_numeric(sub['week'], errors='coerce').dropna().min())
        return None
    except Exception:
        return None


def _count_week_rows_lines_csv(week: int) -> int:
    path = os.path.join(DATA_DIR, 'college_football_betting_lines_2025.csv')
    try:
        if not os.path.exists(path):
            return 0
        df = pd.read_csv(path)
        if df.empty:
            return 0
        return int(((pd.to_numeric(df.get('week'), errors='coerce') == week) & (pd.to_numeric(df.get('year'), errors='coerce') == 2025)).sum())
    except Exception:
        return 0


def _compute_odds_coverage_from_app(week: int) -> int:
    # Import app and use its helper get_betting_lines
    sys.path.insert(0, ROOT)
    import app
    try:
        app._overlay_lines_2025_if_present()
    except Exception:
        pass
    try:
        df = app.pred_df
        wk = df[df.get('week') == week]
        cnt = 0
        for _, r in wk.iterrows():
            odds = app.get_betting_lines(int(r.get('season', 2025)), int(r.get('week', week)), r.get('home_team'), r.get('away_team'))
            if odds and any(not (o.get('synthetic') or False) for o in odds):
                cnt += 1
        return int(cnt)
    except Exception:
        return 0


def maybe_fetch_new_lines(week: int) -> dict:
    """If ODDS_API_KEY exists, fetch odds and merge/write; else skip."""
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        # secrets fallback like app.py
        sec = os.path.join(ROOT, 'secrets', 'odds_api_key.txt')
        if os.path.exists(sec):
            try:
                api_key = open(sec, 'r', encoding='utf-8').read().strip()
            except Exception:
                api_key = None
    if not api_key:
        return {'skipped': 'no_api_key'}
    try:
        import fetch_2025_lines as f
        sport = os.environ.get('ODDS_API_SPORT', 'americanfootball_ncaaf')
        regions = os.environ.get('ODDS_API_REGIONS', 'us,us2,eu,uk')
        markets = os.environ.get('ODDS_API_MARKETS', 'h2h,spreads,totals')
        odds_format = os.environ.get('ODDS_API_ODDS_FORMAT', 'american')
        events = f.fetch_odds(api_key, sport, regions, markets, odds_format)
        rows = f.build_lines_rows(week, events, debug=False)
        # Enhance missing markets with second pass (h2h and spreads/totals)
        try:
            pred_df = f._select_predictions_frame(week)
            fbs_pairs = f._classify_fbs_pairs(pred_df)
            # If many rows lack ML/spreads or totals, do second fetches
            # h2h pass
            try:
                rows_idx = { (r['homeTeam'], r['awayTeam']): r for r in rows }
                need_ml = []
                for (ht, at) in fbs_pairs:
                    rr = rows_idx.get((ht, at))
                    if not rr:
                        need_ml.append((ht, at))
                        continue
                    try:
                        lst = json.loads(rr['lines']) if isinstance(rr['lines'], str) else rr['lines']
                    except Exception:
                        lst = []
                    has_ml = any((p.get('homeMoneyline') is not None or p.get('awayMoneyline') is not None) for p in lst)
                    if not has_ml:
                        need_ml.append((ht, at))
                if need_ml:
                    h2h_events = f.fetch_odds(api_key, sport, os.environ.get('ODDS_API_H2H_EXTRA_REGIONS', 'us,us2,eu,uk'), 'h2h', odds_format)
                    f._merge_provider_entries(rows, f.build_lines_rows(week, h2h_events, debug=False))
            except Exception:
                pass
            # spreads/totals pass
            try:
                need_st = []
                for r in rows:
                    try:
                        lst = json.loads(r['lines']) if isinstance(r['lines'], str) else r['lines']
                    except Exception:
                        lst = []
                    has_spread = any(p.get('spread') is not None for p in lst)
                    has_total = any(p.get('overUnder') is not None for p in lst)
                    if not (has_spread and has_total):
                        need_st.append(True)
                if need_st:
                    st_events = f.fetch_odds(api_key, sport, os.environ.get('ODDS_API_SPREADS_TOTALS_EXTRA_REGIONS', 'us,us2,eu,uk'), 'spreads,totals', odds_format)
                    f._merge_provider_entries(rows, f.build_lines_rows(week, st_events, debug=False))
            except Exception:
                pass
        except Exception:
            pass
        res = f.merge_and_write(rows, week)
        return {'status': 'ok', **res}
    except Exception as e:
        return {'error': str(e)}


def main() -> int:
    report = {
        'ts_utc': _now_iso(),
        'status': 'unknown',
    }
    try:
        week = _detect_upcoming_week()
        report['upcoming_week'] = week
        before_rows = _count_week_rows_lines_csv(week) if week is not None else None
        report['lines_rows_before'] = before_rows
        # Compute pre coverage via app
        if week is not None:
            report['coverage_before'] = _compute_odds_coverage_from_app(week)
        # Fetch new lines & write
        if week is not None:
            fetch_res = maybe_fetch_new_lines(week)
        else:
            fetch_res = {'skipped': 'no_week'}
        report['fetch'] = fetch_res
        after_rows = _count_week_rows_lines_csv(week) if week is not None else None
        report['lines_rows_after'] = after_rows
        # Reload app & recompute coverage
        if week is not None:
            report['coverage_after'] = _compute_odds_coverage_from_app(week)
        # Simple pass heuristic: after_rows >= before_rows and coverage_after >= coverage_before
        ok = True
        try:
            if before_rows is not None and after_rows is not None and after_rows < before_rows:
                ok = False
            cb = report.get('coverage_before'); ca = report.get('coverage_after')
            if isinstance(cb, int) and isinstance(ca, int) and ca < cb:
                ok = False
        except Exception:
            pass
        report['status'] = 'PASS' if ok else 'WARN'
    except Exception as e:
        report['status'] = 'ERROR'
        report['error'] = str(e)
        report['trace'] = traceback.format_exc()[-2000:]
    out_path = os.path.join(LOG_DIR, 'smoke_daily_update.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(out_path)
    except Exception:
        print(json.dumps(report, indent=2))
    return 0 if report.get('status') in ('PASS','WARN') else 1


if __name__ == '__main__':
    raise SystemExit(main())
