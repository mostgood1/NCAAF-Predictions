"""Daily scores finalization check.

Purpose:
  Run lightweight score updates for both the prior completed week and the current (upcoming) week in progress so that
  newly finalized mid-week games (e.g., Thursday/Friday) get captured in the with_scores file without waiting for the
  next full weekly pipeline.

Behavior:
  - Detect prior and upcoming week using the same helper logic from weekly_update.
  - Call app._update_scores_with_cfbd(prior_week) and app._update_scores_with_cfbd(upcoming_week) when present.
  - If upcoming_week is None, no second update is attempted.
  - Prints a concise human summary or JSON (via --print-json).

Usage:
  python daily_scores_check.py
  python daily_scores_check.py --print-json
"""
from __future__ import annotations
import argparse
import json


def detect_weeks():
    try:
        import weekly_update
        return weekly_update._detect_weeks_from_pred_df()  # (prior, upcoming, current_min)
    except Exception:
        return None, None, None


def _weeks_for_dates(dates):
    """Return set of week numbers in pred_df whose start_date date component is in dates."""
    weeks = set()
    try:
        import app
        df = app.pred_df
        if 'start_date' not in df.columns or 'week' not in df.columns:
            return weeks
        sub = df[df['start_date'].notna()]
        for _, r in sub.iterrows():
            try:
                d = r['start_date']
                import pandas as pd
                dts = pd.to_datetime(d, errors='coerce')
                if pd.isna(dts):
                    continue
                if dts.date() in dates:
                    try:
                        wk = int(r['week'])
                        weeks.add(wk)
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        return weeks
    return weeks


def run_updates(prior_week, upcoming_week, extra_weeks=None):
    import app  # noqa: F401
    results = {}
    try:
        if prior_week is not None:
            results['prior_week'] = app._update_scores_with_cfbd(prior_week)
        else:
            results['prior_week'] = {'skipped': 'no_prior_week'}
    except Exception as e:
        results['prior_week'] = {'error': str(e)}

    try:
        if upcoming_week is not None:
            results['upcoming_week'] = app._update_scores_with_cfbd(upcoming_week)
        else:
            results['upcoming_week'] = {'skipped': 'no_upcoming_week'}
    except Exception as e:
        results['upcoming_week'] = {'error': str(e)}

    if extra_weeks:
        ew_res = {}
        for wk in sorted(extra_weeks):
            # Avoid re-running duplicate weeks already covered
            if wk in (prior_week, upcoming_week):
                continue
            try:
                import app as _app
                ew_res[wk] = _app._update_scores_with_cfbd(wk)
            except Exception as e:
                ew_res[wk] = {'error': str(e)}
        results['extra_weeks'] = ew_res
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print-json', action='store_true')
    ap.add_argument('--scan-today-yesterday', action='store_true', help='Also update weeks containing games starting today or yesterday (by date)')
    ap.add_argument('--scan-last-days', type=int, default=0, help='Also update weeks that contain games in the last N days (UTC)')
    ap.add_argument('--weeks', type=str, default='', help='Comma-separated explicit week numbers to also update (e.g., 9,10)')
    args = ap.parse_args()

    prior, upcoming, _ = detect_weeks()
    extra = set()
    # If scanning last days, hint app's ESPN fetcher to include those dates when start_date might be missing
    if args.scan_last_days and args.scan_last_days > 0:
        try:
            import os
            if not os.environ.get('ESPN_BACKFILL_DAYS'):
                os.environ['ESPN_BACKFILL_DAYS'] = str(int(args.scan_last_days))
        except Exception:
            pass
    if args.scan_today_yesterday:
        from datetime import datetime, timedelta
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        extra.update(_weeks_for_dates({today, yesterday}))
    if args.scan_last_days and args.scan_last_days > 0:
        from datetime import datetime, timedelta
        today = datetime.utcnow().date()
        days = { today - timedelta(days=k) for k in range(0, int(args.scan_last_days)+1) }
        extra.update(_weeks_for_dates(days))
    if args.weeks:
        for w in str(args.weeks).split(','):
            try:
                extra.add(int(w.strip()))
            except Exception:
                continue
    res = run_updates(prior, upcoming, extra_weeks=(sorted(extra) if extra else None))
    res['detected'] = {'prior': prior, 'upcoming': upcoming, 'extra_weeks': sorted(extra) if extra else []}

    if args.print_json:
        print(json.dumps(res, indent=2))
        return

    def line(label, d):
        if isinstance(d, dict):
            if 'error' in d:
                return f"- {label}: ERROR {d['error']}"
            if 'skipped' in d:
                return f"- {label}: skipped ({d['skipped']})"
            if d.get('updated'):
                return f"- {label}: updated={d.get('updated')}"  # fallback
        return f"- {label}: ok"

    print(f"Daily scores check (prior={prior}, upcoming={upcoming}, extra={res['detected'].get('extra_weeks')})")
    print(line('prior_week', res.get('prior_week', {})))
    print(line('upcoming_week', res.get('upcoming_week', {})))
    if 'extra_weeks' in res:
        for wk, det in res['extra_weeks'].items():
            print(line(f'extra_week_{wk}', det))


if __name__ == '__main__':
    main()
