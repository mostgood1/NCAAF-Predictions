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


def run_updates(prior_week, upcoming_week):
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
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print-json', action='store_true')
    args = ap.parse_args()

    prior, upcoming, _ = detect_weeks()
    res = run_updates(prior, upcoming)
    res['detected'] = {'prior': prior, 'upcoming': upcoming}

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

    print(f"Daily scores check (prior={prior}, upcoming={upcoming})")
    print(line('prior_week', res.get('prior_week', {})))
    print(line('upcoming_week', res.get('upcoming_week', {})))


if __name__ == '__main__':
    main()
