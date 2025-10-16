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
        # Prefer the root script first (Odds API version)
        os.path.join(base, rel_path),
        # Then project subfolders
        os.path.join(base, 'src', 'data', rel_path),
        os.path.join(base, 'src', 'modeling', rel_path),
        os.path.join(base, 'NCAFCompare', 'src', 'data', rel_path),
        os.path.join(base, 'NCAFCompare', 'src', 'modeling', rel_path),
        os.path.join(base, 'NCAFCompare', rel_path),
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

    # Refresh kickoff times for upcoming week to avoid UTC drift before lines/weather
    if upcoming_week is not None:
        rk = _run_script_if_exists('scripts/refresh_kickoff_times.py', ['--weeks', str(upcoming_week)])
        if rk.get('skipped') == 'not_found':
            # fallback to in-app helper
            try:
                res = webapp._refresh_schedule_kickoffs(week=upcoming_week, overwrite=True)
                rk = res if isinstance(res, dict) else {'status': 'ok', 'details': res}
            except Exception as e:
                rk = {'error': f'kickoff_refresh_failed: {e}'}
        results['refresh_kickoffs'] = rk
    else:
        results['refresh_kickoffs'] = {'skipped': 'no_upcoming_week'}

    # Fetch betting lines for upcoming week (if script exists)
    if upcoming_week is not None:
        results['fetch_lines'] = _run_script_if_exists('fetch_2025_lines.py', ['--week', str(upcoming_week), '--debug'])
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

    # Train ATS/Totals classifiers (if script exists)
    results['train_ats_totals'] = _run_script_if_exists('train_ats_totals.py')

    # Re-generate enhanced predictions (if generator exists)
    results['generate_predictions'] = _run_script_if_exists('generate_enhanced_predictions.py')

    # Ensure finalized games use pregame predictions by restoring from nearest pre-kickoff snapshots
    try:
        restore = _run_script_if_exists('scripts/restore_pregame_predictions.py', ['--write'])
    except Exception as e:
        restore = {'error': f'restore_failed: {e}'}
    results['restore_pregame'] = restore

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

    # Post-reload sanity: ensure upcoming week has FBS-involved games available
    try:
        fbs_confs = {
            'acc','sec','big ten','big 12','pac 12','american','mountain west','sun belt','mac','conference usa','independent','independents','fbs independents','independent (fbs)'
        }
        fbs_indies = {'notre dame','army','navy','umass','uconn','new mexico state'}
        def _is_fbs(team, conf):
            try:
                t = str(team or '').strip().lower()
                c = str(conf or '').strip().lower()
                return c in fbs_confs or t in fbs_indies
            except Exception:
                return False
        df = webapp.pred_df
        wk = upcoming_week
        wk_df = df[df.get('week', -1) == wk] if wk is not None else df.iloc[0:0]
        total = int(len(wk_df))
        fbs_involved = int(wk_df.apply(lambda r: (_is_fbs(r.get('home_team'), r.get('home_conference')) or _is_fbs(r.get('away_team'), r.get('away_conference'))), axis=1).sum()) if total > 0 else 0
        fbsvfbs = int(wk_df.apply(lambda r: (_is_fbs(r.get('home_team'), r.get('home_conference')) and _is_fbs(r.get('away_team'), r.get('away_conference'))), axis=1).sum()) if total > 0 else 0
        results['sanity'] = {
            'upcoming_week': wk,
            'total_rows_week': total,
            'fbs_involved_count': fbs_involved,
            'fbsvfbs_count': fbsvfbs,
        }
    except Exception as e:
        results['sanity'] = {'error': f'sanity_check_failed: {e}'}

    # Produce a recommendations snapshot (non-destructive) for visibility
    try:
        recs = webapp.compute_recommendations(week=upcoming_week, bankroll=1000.0, kelly_factor=0.5, ev_threshold=0.02)
        snap_limit = 200
        snap = recs[:snap_limit]
        import json as _json
        snap_path = os.path.join(webapp.DATA_DIR, 'recommendations_latest.json')
        with open(snap_path, 'w', encoding='utf-8') as f:
            _json.dump({
                'generated_utc': dt.datetime.utcnow().isoformat() + 'Z',
                'week': upcoming_week,
                'count': len(snap),
                'bankroll': 1000.0,
                'kelly_factor': 0.5,
                'ev_threshold': 0.02,
                'results': snap,
            }, f)
        results['recommendations_snapshot'] = {'path': os.path.relpath(snap_path, base), 'count': len(snap)}
    except Exception as e:
        results['recommendations_snapshot'] = {'error': str(e)}

    # Build offline artifacts for easy reconciliation (CSV/JSON caches)
    try:
        builder = os.path.join(base, 'scripts', 'offline_recommendations_artifacts.py')
        if os.path.exists(builder) and upcoming_week is not None:
            out = subprocess.run([sys.executable, builder, '--week', str(upcoming_week)], capture_output=True, text=True, check=False)
            results['offline_artifacts'] = {
                'returncode': out.returncode,
                'stdout': out.stdout[-2000:],
                'stderr': out.stderr[-2000:],
            }
        else:
            results['offline_artifacts'] = {'skipped': 'builder_missing_or_no_week'}
    except Exception as e:
        results['offline_artifacts'] = {'error': str(e)}

    # Evaluate ATS/Totals accuracy for visibility
    try:
        eval_script = os.path.join(base, 'scripts', 'evaluate_ats_totals.py')
        if os.path.exists(eval_script):
            out = subprocess.run([sys.executable, eval_script], capture_output=True, text=True, check=False)
            results['evaluation'] = {
                'returncode': out.returncode,
                'stdout': out.stdout[-2000:],
                'stderr': out.stderr[-2000:],
            }
        else:
            results['evaluation'] = {'skipped': 'script_missing'}
    except Exception as e:
        results['evaluation'] = {'error': str(e)}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-week', type=int, default=None, help='Completed prior week (e.g., 1)')
    ap.add_argument('--upcoming-week', type=int, default=None, help='Upcoming week to prep (e.g., 2)')
    ap.add_argument('--print-json', action='store_true', help='Print JSON summary to stdout')
    # Future: could expose rec parameters via CLI; for now snapshot uses defaults
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
        print(line('refresh_kickoffs', res.get('refresh_kickoffs', {})))
        print(line('fetch_lines', res.get('fetch_lines', {})))
        print(line('weather', res.get('weather', {})))
        print(line('merge_features', res.get('merge_features', {})))
        print(line('retune_models', res.get('retune_models', {})))
        print(line('generate_predictions', res.get('generate_predictions', {})))
        print(line('train_ats_totals', res.get('train_ats_totals', {})))
        print(line('reload', res.get('reload', {})))
        if 'sanity' in res:
            s = res.get('sanity', {})
            if isinstance(s, dict) and 'error' in s:
                print(f"- sanity: ERROR: {s['error']}")
            elif isinstance(s, dict):
                print(f"- sanity: wk={s.get('upcoming_week')} total={s.get('total_rows_week')} fbs-involved={s.get('fbs_involved_count')} fbsvfbs={s.get('fbsvfbs_count')}")
        if 'recommendations_snapshot' in res:
            print(line('recommendations_snapshot', res.get('recommendations_snapshot', {})))
        if 'offline_artifacts' in res:
            print(line('offline_artifacts', res.get('offline_artifacts', {})))
        if 'evaluation' in res:
            print(line('evaluation', res.get('evaluation', {})))


if __name__ == '__main__':
    main()
