#!/usr/bin/env python3
"""
Restore pregame predictions for finalized games into the with_scores CSV.

Strategy:
- Load with_scores (college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv).
- Identify finalized games (actual_home_points & actual_away_points present).
- Determine kickoff timestamp from start_date_api (preferred) or start_date.
- Scan data/ for enhanced snapshot files (timestamped variant preferred) excluding with_scores.
- For each finalized game, find the snapshot with timestamp <= kickoff and closest to it.
- Pull predicted_home_points/predicted_away_points from that snapshot row and set them in with_scores.
- Optional --write flag will persist updates; otherwise prints a dry-run report.

Usage:
  python scripts/restore_pregame_predictions.py          # dry run
  python scripts/restore_pregame_predictions.py --write  # write updates
"""
import os
import re
import sys
import math
import glob
import argparse
from datetime import datetime, timezone
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
WITH_SCORES = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv')

SNAPSHOT_PATTERN = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced*.csv')

TS_RE = re.compile(r'_([0-9]{8}T[0-9]{6})Z')  # e.g., _20251014T134052Z


def _parse_snapshot_ts(path: str) -> int:
    base = os.path.basename(path)
    m = TS_RE.search(base)
    if not m:
        # non-timestamped base file -> treat as very early (priority low, but usable)
        return -1
    try:
        # convert to int YYYYMMDDHHMMSS for ordering
        s = m.group(1)
        return int(s.replace('T', ''))
    except Exception:
        return -1


def _parse_kickoff_iso(iso: str | None) -> float | None:
    if not iso or (isinstance(iso, float) and math.isnan(iso)):
        return None
    try:
        s = str(iso).strip()
        # Expect Z or offset; pandas can handle
        dt = pd.to_datetime(s, errors='coerce', utc=True)
        if pd.isna(dt):
            return None
        return float(dt.timestamp())
    except Exception:
        return None


def _load_with_scores() -> pd.DataFrame:
    if not os.path.exists(WITH_SCORES):
        raise FileNotFoundError(f"with_scores not found: {WITH_SCORES}")
    df = pd.read_csv(WITH_SCORES)
    return df


def _list_snapshots() -> list[tuple[str, int]]:
    files = [p for p in glob.glob(SNAPSHOT_PATTERN) if not p.endswith('with_scores.csv')]
    # filter to preferred filenames: base or timestamped only
    def _is_preferred(fname: str) -> bool:
        b = os.path.basename(fname)
        return bool(re.match(r'^college_football_schedule_2025_predicted_totals_enhanced(?:_[0-9]{8}T[0-9]{6}Z)?\.csv$', b))
    files = [p for p in files if _is_preferred(p)]
    snaps = [(p, _parse_snapshot_ts(p)) for p in files]
    # newest last for searching <= kickoff (we'll binary search-ish by simple scan)
    snaps.sort(key=lambda t: t[1])
    return snaps


def main(write: bool = False) -> int:
    ws = _load_with_scores()
    # Finals only
    finals = ws.dropna(subset=['actual_home_points', 'actual_away_points']).copy()
    if finals.empty:
        print('[restore] No finalized games found in with_scores; nothing to do.')
        return 0
    # Build kickoff timestamps
    finals['kick_ts'] = None
    if 'start_date_api' in finals.columns:
        finals['kick_ts'] = finals['start_date_api'].apply(_parse_kickoff_iso)
    # fallback to start_date (naive assumed UTC)
    if 'start_date' in finals.columns:
        finals['kick_ts'] = finals['kick_ts'].fillna(finals['start_date'].apply(_parse_kickoff_iso))
    snaps = _list_snapshots()
    if not snaps:
        print('[restore] No enhanced snapshots found; cannot restore.')
        return 2
    # For each final, find best snapshot path
    finals['snap_idx'] = None
    def _choose_snap(ts: float | None) -> int | None:
        if ts is None:
            # no kickoff -> choose latest pre-season baseline if available (ts == -1), else earliest timestamped
            # prefer last entry with ts == -1 if present
            idx_base = None
            for i, (_, t) in enumerate(snaps):
                if t == -1:
                    idx_base = i
            if idx_base is not None:
                return idx_base
            return 0  # earliest snapshot
        # find the right-most snapshot with t <= ts
        best = None
        for i, (_, t) in enumerate(snaps):
            if t <= ts:
                best = i
            else:
                break
        return best
    finals['snap_idx'] = finals['kick_ts'].apply(_choose_snap)
    # Group by snapshot index to limit file reads
    file_to_keys: dict[int, list[tuple[int,int,str,str]]] = {}
    for i, r in finals.iterrows():
        si = r['snap_idx']
        if si is None or (isinstance(si, float) and math.isnan(si)):
            continue
        y = int(r.get('season', 2025))
        w = int(r.get('week')) if pd.notna(r.get('week')) else None
        if w is None:
            continue
        key = (y, w, r['home_team'], r['away_team'])
        file_to_keys.setdefault(int(si), []).append(key)
    # Build mapping from (season,week,home,away) -> (ph, pa)
    restored: dict[tuple[int,int,str,str], tuple[float|None,float|None]] = {}
    for si, keys in file_to_keys.items():
        path, _ = snaps[int(si)]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[restore] failed reading {os.path.basename(path)}: {e}")
            continue
        # Minimal columns
        need = {'season','week','home_team','away_team','predicted_home_points','predicted_away_points'}
        missing = [c for c in need if c not in df.columns]
        if missing:
            # Try model_* fallback to construct predicted_* if necessary
            if {'model_home_points','model_away_points'}.issubset(df.columns):
                df['predicted_home_points'] = df.get('predicted_home_points', df['model_home_points'])
                df['predicted_away_points'] = df.get('predicted_away_points', df['model_away_points'])
            else:
                # cannot use this file
                print(f"[restore] snapshot {os.path.basename(path)} missing required columns: {missing}")
                continue
        # Build index
        df['_key'] = list(zip(df['season'].astype(int), df['week'].astype(int), df['home_team'], df['away_team']))
        sub = df[df['_key'].isin(keys)][['_key','predicted_home_points','predicted_away_points']]
        for _, row in sub.iterrows():
            k = row['_key']
            try:
                ph = float(row['predicted_home_points']) if pd.notna(row['predicted_home_points']) else None
            except Exception:
                ph = None
            try:
                pa = float(row['predicted_away_points']) if pd.notna(row['predicted_away_points']) else None
            except Exception:
                pa = None
            restored[k] = (ph, pa)
    # Compare and optionally write
    mismatches = 0
    updates = 0
    applied_rows = 0
    for i, r in ws.iterrows():
        try:
            if pd.isna(r.get('actual_home_points')) or pd.isna(r.get('actual_away_points')):
                continue
            y = int(r.get('season', 2025))
            w = int(r.get('week')) if pd.notna(r.get('week')) else None
            if w is None:
                continue
            k = (y, w, r['home_team'], r['away_team'])
            if k not in restored:
                continue
            ph_s, pa_s = restored[k]
            if ph_s is None or pa_s is None:
                continue
            # current values
            try:
                ph_c = float(r.get('predicted_home_points')) if pd.notna(r.get('predicted_home_points')) else None
            except Exception:
                ph_c = None
            try:
                pa_c = float(r.get('predicted_away_points')) if pd.notna(r.get('predicted_away_points')) else None
            except Exception:
                pa_c = None
            if ph_c != ph_s or pa_c != pa_s:
                mismatches += 1
                if write:
                    ws.at[i, 'predicted_home_points'] = ph_s
                    ws.at[i, 'predicted_away_points'] = pa_s
                    updates += 1
            applied_rows += 1
        except Exception:
            continue
    print(f"[restore] finals considered: {len(finals)}; matched rows: {applied_rows}; mismatches: {mismatches}")
    if write and updates > 0:
        ws.to_csv(WITH_SCORES, index=False)
        print(f"[restore] wrote updates: {updates} -> {os.path.basename(WITH_SCORES)}")
    elif not write:
        print("[restore] dry-run only; re-run with --write to persist.")
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='Persist updates to with_scores file')
    args = ap.parse_args()
    sys.exit(main(write=args.write))
