"""
Weekly automation script.

Steps:
- Determine prior completed week and upcoming week (or accept CLI overrides)
- Archive current predictions to data/archive with timestamp/week
- Update final scores for prior week (CFBD/ESPN via app helper)
- Fetch betting lines for upcoming week (calls existing script if present)
- Update weather for upcoming week (calls existing script if present)
- Update/merge team stats & features (calls merge scripts if present)
- Re-tune model (if script exists)
- Re-run predictions for rest of season (if generator exists)
- Reload in-memory predictions and overlay lines

Usage:
  python weekly_update.py --prior-week 1 --upcoming-week 2
  python weekly_update.py            # auto-detect from current predictions
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import datetime as dt
import subprocess


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_script(rel_path: str) -> str | None:
    base = _here()
    cands = [
        os.path.join(base, 'src', 'data', rel_path),
        os.path.join(base, 'src', 'modeling', rel_path),
        os.path.join(base, rel_path),
    ]
    for p in cands:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def _run_script_if_exists(rel_path: str, args: list[str] | None = None) -> dict:
    script = _resolve_script(rel_path)
    if not script:
        return { 'script': rel_path, 'skipped': 'not_found' }
    cmd = [sys.executable, script] + (args or [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            'script': os.path.relpath(script, _here()),
            'returncode': out.returncode,
            'stdout': out.stdout[-4000:],
            'stderr': out.stderr[-4000:],
        }
    except Exception as e:
        return { 'script': rel_path, 'error': str(e) }


def _archive_predictions(data_dir: str, week: int | None) -> list[dict]:
    os.makedirs(os.path.join(data_dir, 'archive'), exist_ok=True)
    ts = dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    suffix = f"wk{week}" if week is not None else 'wkNA'
    files = [
        'college_football_schedule_2025_predicted_totals_enhanced.csv',
        'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv',
        'win_margin_predictions_with_confidence.csv',
        'college_football_betting_lines_2025.csv',
    ]
    ops = []
    for name in files:
        src = os.path.join(data_dir, name)
        if os.path.exists(src):
            dst = os.path.join(data_dir, 'archive', f"{name}.{suffix}.{ts}.bak")
            try:
                shutil.copy2(src, dst)
                ops.append({'file': name, 'archived_to': os.path.relpath(dst, data_dir)})
            except Exception as e:
                ops.append({'file': name, 'error': str(e)})
        else:
            ops.append({'file': name, 'skipped': 'missing'})
    return ops


def _detect_weeks_from_pred_df() -> tuple[int | None, int | None, int | None]:
    # Import app lazily to reuse its loaders without starting the server
    import app as webapp
    try:
        df = webapp.pred_df.copy()
    except Exception:
        return None, None, None
    if df.empty or 'week' not in df.columns:
        return None, None, None
    # Prior completed week: max week with any actuals present
    prior = None
    try:
        sub = df[(df.get('season', 0) == 2025)]
        done = sub[sub['actual_home_points'].notna() & sub['actual_away_points'].notna()]
        if not done.empty:
            prior = int(done['week'].max())
    except Exception:
        prior = None
    upcoming = (prior + 1) if (prior is not None) else None
    current = None
    try:
        current = int(df['week'].min()) if not df.empty else None
    except Exception:
        current = None
    return prior, upcoming, current


def weekly_update(prior_week: int | None, upcoming_week: int | None) -> dict:
    import app as webapp
    base = _here()
    results = { 'steps': [] }

    # Detect weeks if not provided
    if prior_week is None or upcoming_week is None:
        det_prior, det_upcoming, _ = _detect_weeks_from_pred_df()
        prior_week = prior_week if prior_week is not None else det_prior
        upcoming_week = upcoming_week if upcoming_week is not None else det_upcoming
    results['prior_week'] = prior_week
    results['upcoming_week'] = upcoming_week

    # Archive current predictions
    data_dir = webapp.DATA_DIR
    results['archive'] = _archive_predictions(data_dir, prior_week)

    # Update final scores for prior week using in-app helper
    try:
        upd = webapp._update_scores_with_cfbd(prior_week)
    except Exception as e:
        upd = {'error': f'update_scores_failed: {e}'}
    results['update_scores'] = upd

    # Fetch betting lines for upcoming week (if script exists)
    if upcoming_week is not None:
        results['fetch_lines'] = _run_script_if_exists('fetch_2025_lines.py', ['--week', str(upcoming_week)])
    else:
        results['fetch_lines'] = {'skipped': 'no_upcoming_week'}

    # Update weather for upcoming week (if script exists)
    if upcoming_week is not None:
        results['weather'] = _run_script_if_exists('enrich_weather_2025.py', ['--week', str(upcoming_week)])
    else:
        results['weather'] = {'skipped': 'no_upcoming_week'}

    # Merge/feature updates
    results['merge_features'] = _run_script_if_exists('merge_all_features.py')

    # Retune models (if script exists)
    results['retune_models'] = _run_script_if_exists('retune_models.py')

    # Re-generate enhanced predictions (if generator exists)
    results['generate_predictions'] = _run_script_if_exists('generate_enhanced_predictions.py')

    # Reload predictions and overlay lines in memory for immediate app usage
    try:
        webapp._reload_predictions()
        try:
            webapp._overlay_lines_2025_if_present()
        except Exception:
            pass
    except Exception as e:
        results['reload'] = {'error': str(e)}
    else:
        results['reload'] = {'status': 'ok', 'rows': int(len(webapp.pred_df))}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-week', type=int, default=None, help='Completed prior week (e.g., 1)')
    ap.add_argument('--upcoming-week', type=int, default=None, help='Upcoming week to prep (e.g., 2)')
    ap.add_argument('--print-json', action='store_true', help='Print JSON summary to stdout')
    args = ap.parse_args()

    res = weekly_update(args.prior_week, args.upcoming_week)
    if args.print_json:
        import json
        print(json.dumps(res, indent=2))
    else:
        # Compact human summary
        def line(name, d):
            if isinstance(d, dict) and 'error' in d:
                return f"- {name}: ERROR: {d['error']}"
            if isinstance(d, dict) and 'skipped' in d:
                return f"- {name}: skipped ({d['skipped']})"
            if isinstance(d, dict) and 'returncode' in d:
                return f"- {name}: rc={d['returncode']} ({d.get('script','')})"
            return f"- {name}: ok"
        print(f"Weekly update (prior={res.get('prior_week')}, upcoming={res.get('upcoming_week')})")
        print(line('archive', {'ok': True}))
        print(line('update_scores', res.get('update_scores', {})))
        print(line('fetch_lines', res.get('fetch_lines', {})))
        print(line('weather', res.get('weather', {})))
        print(line('merge_features', res.get('merge_features', {})))
        print(line('retune_models', res.get('retune_models', {})))
        print(line('generate_predictions', res.get('generate_predictions', {})))
        print(line('reload', res.get('reload', {})))


if __name__ == '__main__':
    main()
