"""
Evaluate ATS and Totals predictions on completed 2025 games.

Computes accuracy (hit rate) for:
- Classifier-based probabilities (if classifiers are present)
- Gaussian baseline probabilities (existing approach)

Data sources:
- app.pred_df (with_scores merged)
- app.get_betting_lines for market spread/total (median across providers)

Outputs:
- Prints JSON summary to stdout
- Writes data/metrics/ats_totals_eval.json (with timestamp)
"""
from __future__ import annotations
import os
import json
import math
from datetime import datetime, timezone

import numpy as np

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
import app as webapp


def _median_from_lines(lines, key):
    vals = []
    for e in (lines or []):
        v = e.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            vals.append(float(v))
    if not vals:
        return None
    return float(np.median(vals))


def evaluate():
    df = webapp.pred_df.copy()
    df = df[(df.get('season', 0) == 2025)]
    # Completed only
    df = df[df['actual_home_points'].notna() & df['actual_away_points'].notna()].copy()
    if df.empty:
        return {'error': 'no_completed_games'}

    # Ensure models loaded
    webapp._load_model_artifacts()

    res = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'samples': {'ats': 0, 'totals': 0, 'ats_pushes': 0, 'totals_pushes': 0},
        'accuracy': {
            'ats': {'clf': None, 'gaussian': None, 'clf_correct': 0, 'gaussian_correct': 0},
            'totals': {'clf': None, 'gaussian': None, 'clf_correct': 0, 'gaussian_correct': 0},
        },
        'coverage': {'with_any_lines': 0, 'with_spread': 0, 'with_total': 0}
    }

    # Iterate games
    for _, r in df.iterrows():
        lines = webapp.get_betting_lines(int(r['season']), int(r['week']), r['home_team'], r['away_team'])
        if lines:
            res['coverage']['with_any_lines'] += 1
        spread = _median_from_lines(lines, 'spread')
        ou = _median_from_lines(lines, 'overUnder')

        actual_home = webapp._safe_float(r.get('actual_home_points'))
        actual_away = webapp._safe_float(r.get('actual_away_points'))
        if actual_home is None or actual_away is None:
            continue
        actual_margin = actual_home - actual_away
        actual_total = actual_home + actual_away

        # Totals evaluation
        if ou is not None:
            res['coverage']['with_total'] += 1
            # Determine result/push
            if abs(actual_total - ou) < 1e-9:
                res['samples']['totals_pushes'] += 1
            else:
                res['samples']['totals'] += 1
                sigma_t = webapp._get_total_points_std()
                pred_total = None
                mh = webapp._safe_float(r.get('model_home_points')) or webapp._safe_float(r.get('predicted_home_points'))
                ma = webapp._safe_float(r.get('model_away_points')) or webapp._safe_float(r.get('predicted_away_points'))
                if mh is not None and ma is not None:
                    pred_total = mh + ma
                # Gaussian baseline
                p_over_g = None
                try:
                    if pred_total is not None and sigma_t not in (None, 0, 0.0) and not (isinstance(sigma_t, float) and math.isnan(sigma_t)):
                        p_over_g = 1 - webapp._phi((ou - pred_total) / sigma_t)
                except Exception:
                    pass
                if p_over_g is not None:
                    pred_side = 'Over' if p_over_g >= 0.5 else 'Under'
                    actual_side = 'Over' if actual_total > ou else 'Under'
                    if pred_side == actual_side:
                        res['accuracy']['totals']['gaussian_correct'] += 1
                # Classifier
                p_over_c = webapp._predict_p_over(r, ou, pred_total, sigma_t)
                if p_over_c is not None:
                    pred_side_c = 'Over' if p_over_c >= 0.5 else 'Under'
                    actual_side = 'Over' if actual_total > ou else 'Under'
                    if pred_side_c == actual_side:
                        res['accuracy']['totals']['clf_correct'] += 1

        # ATS evaluation
        if spread is not None:
            res['coverage']['with_spread'] += 1
            comp = actual_margin + spread
            if abs(comp) < 1e-9:
                res['samples']['ats_pushes'] += 1
            else:
                res['samples']['ats'] += 1
                sigma_m = webapp._get_conf_std_for_game(r)
                # Gaussian baseline
                p_home_cover_g = None
                try:
                    pred_margin = webapp._safe_float(r.get('model_margin'))
                    if pred_margin is None:
                        pred_margin = webapp._safe_float(r.get('predicted_win_margin'))
                    if pred_margin is None and mh is not None and ma is not None:
                        pred_margin = mh - ma
                    if pred_margin is not None and sigma_m not in (None, 0, 0.0) and not (isinstance(sigma_m, float) and math.isnan(sigma_m)):
                        p_home_cover_g = webapp._phi((pred_margin - spread) / sigma_m)
                except Exception:
                    pass
                if p_home_cover_g is not None:
                    pred_side = 'Home' if p_home_cover_g >= 0.5 else 'Away'
                    actual_side = 'Home' if comp > 0 else 'Away'
                    if pred_side == actual_side:
                        res['accuracy']['ats']['gaussian_correct'] += 1
                # Classifier
                p_home_cover_c = webapp._predict_p_home_cover(r, spread, sigma_m)
                if p_home_cover_c is not None:
                    pred_side_c = 'Home' if p_home_cover_c >= 0.5 else 'Away'
                    actual_side = 'Home' if comp > 0 else 'Away'
                    if pred_side_c == actual_side:
                        res['accuracy']['ats']['clf_correct'] += 1

    # Finalize accuracy rates
    for mkt in ('ats', 'totals'):
        tot = res['samples'][mkt]
        if tot > 0:
            res['accuracy'][mkt]['gaussian'] = round(res['accuracy'][mkt]['gaussian_correct'] / tot, 4)
            res['accuracy'][mkt]['clf'] = round(res['accuracy'][mkt]['clf_correct'] / tot, 4)

    # Persist
    try:
        out_dir = os.path.join(webapp.DATA_DIR, 'metrics')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'ats_totals_eval.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        res['output'] = {'path': os.path.relpath(out_path, os.path.dirname(out_dir))}
    except Exception:
        pass

    return res


def main():
    res = evaluate()
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
