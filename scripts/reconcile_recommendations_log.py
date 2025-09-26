#!/usr/bin/env python3
"""
Settle and export the recommendations log into easy-to-diff CSV/JSON artifacts.

Outputs:
  data/recommendations_summary/bets_settled.csv
  data/recommendations_summary/bets_settled.json
  data/recommendations_summary/open_bets.csv
  data/recommendations_summary/open_bets.json

Usage:
  python scripts/reconcile_recommendations_log.py            # all weeks
  python scripts/reconcile_recommendations_log.py --week 6   # specific week
  python scripts/reconcile_recommendations_log.py --weeks 4-6,8
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import List

import pandas as pd
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import app as webapp

DATA_DIR = webapp.DATA_DIR
RECS_PATH = webapp.RECS_PATH
SUMMARY_DIR = os.path.join(DATA_DIR, "recommendations_summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)


def parse_weeks(spec: str | None, available: List[int]) -> List[int]:
    if not spec:
        return available
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
    out = sorted({w for w in out if w in set(available)})
    return out


def settle_and_export(weeks: List[int] | None) -> None:
    if not os.path.exists(RECS_PATH):
        print("No recommendations log found; nothing to do.")
        return
    recs_df = pd.read_csv(RECS_PATH)
    for col in ['stake','pnl','edge','kelly_f','model_prob','implied_prob','week','season']:
        if col in recs_df.columns:
            recs_df[col] = pd.to_numeric(recs_df[col], errors='coerce')
    # Filter to season 2025 if present
    if 'season' in recs_df.columns:
        try:
            sub = recs_df[recs_df['season'] == 2025]
            if not sub.empty:
                recs_df = sub
        except Exception:
            pass
    # Restrict to selected weeks
    if weeks:
        recs_df = recs_df[recs_df['week'].isin(weeks)]

    # Join with predictions for actuals and times
    pred = webapp.pred_df.copy()
    try:
        pred = pred[pred.get('season', 0) == 2025]
    except Exception:
        pass
    key_cols = ['season','week','home_team','away_team']
    left = recs_df.copy()
    right = pred[key_cols + ['actual_home_points','actual_away_points','start_date','start_date_api']].copy()
    merged = left.merge(right, on=key_cols, how='left')

    # Compute result and pnl for each bet
    res = []
    for _, r in merged.iterrows():
        market = r.get('market')
        side = r.get('side')
        price = r.get('price_american', -110)
        stake = float(r.get('stake', 0) or 0)
        ah = r.get('actual_home_points')
        aa = r.get('actual_away_points')
        result = r.get('result', 'open')
        pnl = r.get('pnl', 0)
        if pd.notna(ah) and pd.notna(aa):
            ah = float(ah); aa = float(aa)
            push = False
            win = False
            if market == 'ML':
                if side == 'Home':
                    win = ah > aa; push = (ah == aa)
                else:
                    win = aa > ah; push = (ah == aa)
            elif market == 'Spread':
                line = r.get('line')
                if pd.isna(line):
                    push = True
                else:
                    line = float(line)
                    margin = ah - aa
                    if side == 'Home':
                        win = margin > line; push = abs(margin - line) < 1e-9
                    else:
                        win = margin < line; push = abs(margin - line) < 1e-9
            elif market == 'Total':
                line = r.get('line')
                if pd.isna(line):
                    push = True
                else:
                    line = float(line)
                    total = ah + aa
                    if side == 'Over':
                        win = total > line; push = abs(total - line) < 1e-9
                    else:
                        win = total < line; push = abs(total - line) < 1e-9
            dec, _net = webapp.american_to_decimal(price)
            if push:
                result = 'push'; pnl = 0.0
            elif win:
                result = 'win'; pnl = stake * (dec - 1)
            else:
                result = 'loss'; pnl = -stake
        res.append({**r.to_dict(), 'result': result, 'pnl': round(float(pnl or 0), 2)})
    out_df = pd.DataFrame(res)

    # Export settled bets
    out_path_csv = os.path.join(SUMMARY_DIR, 'bets_settled.csv')
    out_df.to_csv(out_path_csv, index=False)
    with open(os.path.join(SUMMARY_DIR, 'bets_settled.json'), 'w', encoding='utf-8') as f:
        json.dump(out_df.to_dict(orient='records'), f, indent=2)

    # Open bets subset
    open_df = out_df[(out_df.get('result', 'open') == 'open') | (out_df.get('result', 'open') == 'pending')]
    open_path_csv = os.path.join(SUMMARY_DIR, 'open_bets.csv')
    open_df.to_csv(open_path_csv, index=False)
    with open(os.path.join(SUMMARY_DIR, 'open_bets.json'), 'w', encoding='utf-8') as f:
        json.dump(open_df.to_dict(orient='records'), f, indent=2)

    # Print a quick summary
    wins = int((out_df['result'] == 'win').sum()) if 'result' in out_df.columns else 0
    losses = int((out_df['result'] == 'loss').sum()) if 'result' in out_df.columns else 0
    pushes = int((out_df['result'] == 'push').sum()) if 'result' in out_df.columns else 0
    pnl_total = float(out_df['pnl'].sum()) if 'pnl' in out_df.columns else 0.0
    print(f"Settled: {wins}W-{losses}L-{pushes}P, PnL={pnl_total:+.2f}")


def main():
    ap = argparse.ArgumentParser(description='Settle and export recommendations log')
    ap.add_argument('--week', type=int, help='Single week to reconcile')
    ap.add_argument('--weeks', type=str, help='Comma/range list, e.g. 3,4,5 or 3-8; default all in log')
    args = ap.parse_args()

    if not os.path.exists(RECS_PATH):
        print("No recommendations log found.")
        return
    recs_df = pd.read_csv(RECS_PATH)
    weeks_avail = sorted({int(w) for w in pd.to_numeric(recs_df.get('week'), errors='coerce').dropna().unique()})

    if args.week is not None:
        weeks = [int(args.week)]
    else:
        weeks = parse_weeks(args.weeks, weeks_avail) if args.weeks else weeks_avail

    settle_and_export(weeks)


if __name__ == '__main__':
    main()
