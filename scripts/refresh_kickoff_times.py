"""Refresh kickoff times (start_date_api) for a range of weeks and reload app data.

Usage examples:
  python scripts/refresh_kickoff_times.py --from-week 7 --to-week 20
  python scripts/refresh_kickoff_times.py --weeks 7 8 9 10
  python scripts/refresh_kickoff_times.py --from-week 7   # defaults to 7-20
"""
from __future__ import annotations
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weeks', type=int, nargs='*', help='Explicit list of weeks to refresh (overrides range).')
    ap.add_argument('--from-week', type=int, default=7, help='Start week (inclusive). Default 7.')
    ap.add_argument('--to-week', type=int, default=20, help='End week (inclusive). Default 20.')
    ap.add_argument('--no-overwrite', action='store_true', help='Do not overwrite existing start_date_api values.')
    ap.add_argument('--print-json', action='store_true', help='Print JSON summary instead of text.')
    args = ap.parse_args()

    # Ensure repo root is on sys.path so we can import app.py
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import app

    # Build target weeks
    if args.weeks:
        weeks = sorted(set(int(w) for w in args.weeks))
    else:
        w0 = int(args.from_week)
        w1 = int(args.to_week)
        if w1 < w0:
            w0, w1 = w1, w0
        weeks = list(range(w0, w1 + 1))

    results = {'weeks': weeks, 'steps': []}
    total_updates = 0
    for wk in weeks:
        try:
            res = app._refresh_schedule_kickoffs(week=wk, overwrite=(not args.no_overwrite))
        except Exception as e:
            res = {'step': 'schedule_refresh', 'error': str(e), 'week': wk}
        else:
            res['week'] = wk
        results['steps'].append(res)
        try:
            total_updates += int(res.get('updated_rows', 0))
        except Exception:
            pass

    # Reload in-memory data and overlay lines for immediate use
    try:
        app._reload_predictions()
        try:
            app._overlay_lines_2025_if_present()
        except Exception:
            pass
        results['reload'] = {'status': 'ok', 'rows': int(len(app.pred_df))}
    except Exception as e:
        results['reload'] = {'error': str(e)}

    results['total_updated_rows'] = total_updates

    if args.print_json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Kickoff refresh weeks={weeks} total_updated_rows={total_updates}")
        for s in results['steps']:
            if 'error' in s:
                print(f"- week {s.get('week')}: ERROR {s['error']}")
            elif s.get('skipped'):
                print(f"- week {s.get('week')}: skipped ({s['skipped']})")
            else:
                print(f"- week {s.get('week')}: updated_rows={s.get('updated_rows',0)}")


if __name__ == '__main__':
    main()
