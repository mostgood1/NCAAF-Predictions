"""Fetch 2025 NCAAF betting lines via The Odds API and write/merge into data/college_football_betting_lines_2025.csv.

Usage:
  python fetch_2025_lines.py --week 3
  python fetch_2025_lines.py                 # auto-detect upcoming week from predictions

Environment:
  ODDS_API_KEY (required if --api-key not provided)
  ODDS_API_SPORT (default americanfootball_ncaaf)
  ODDS_API_REGIONS (default us)
  ODDS_API_MARKETS (default h2h,spreads,totals)
  ODDS_API_ODDS_FORMAT (default american)

Notes:
  - We DO NOT persist the API key; use env or CLI arg.
  - Provider list includes any bookmakers returned (e.g., DraftKings, Bovada, Caesars etc.).
  - Each row in the CSV stores a JSON list of provider dicts consistent with existing parsing logic in app.py.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime as dt, time, math
import pandas as pd
import requests
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRED_FILES_GLOB = os.path.join(DATA_DIR, 'college_football_schedule_2025_predicted_totals_enhanced*.csv')
LINES_PATH = os.path.join(DATA_DIR, 'college_football_betting_lines_2025.csv')
YEAR = 2025

# Minimal team normalization mirroring logic in app._norm_team_for_odds (simplified)
import re
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^a-z0-9 ]+")

ALIASES = {
    'hawai\'i': 'hawaii', 'hawaii': 'hawaii',
    'utsa': 'texas san antonio', 'ut san antonio': 'texas san antonio',
    'app st': 'appalachian state', 'app state': 'appalachian state', 'appalachian st': 'appalachian state',
    'ole miss': 'mississippi', 'miss st': 'mississippi state', 'la lafayette': 'louisiana', 'louisiana lafayette': 'louisiana',
    'la monroe': 'louisiana monroe', 'umass': 'massachusetts', 'uconn': 'connecticut',
    'byu cougars': 'byu', 'utsa roadrunners': 'texas san antonio',
    'san jose state': 'san josé state', 'san josé state': 'san josé state',
    'sjsu': 'san josé state',
    'texas a&m': 'texas am', 'texas a and m': 'texas am',
    'penn st': 'penn state', 'mich st': 'michigan state', 'florida st': 'florida state', 'boise st': 'boise state',
}

def norm_team(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    s = s.replace('&', ' and ')
    s = _punct_re.sub('', s)
    s = _ws_re.sub(' ', s).strip()
    return ALIASES.get(s, s)

# ---------------------------------------------------------------------------
# Week detection
# ---------------------------------------------------------------------------

def _detect_upcoming_week() -> int | None:
    import glob
    files = sorted(glob.glob(PRED_FILES_GLOB), key=os.path.getmtime, reverse=True)
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
        # Completed week = max where both actual scores present
        done = sub[sub['actual_home_points'].notna() & sub['actual_away_points'].notna()]
        if not done.empty:
            prior = int(done['week'].max())
            return prior + 1
        # If nothing completed yet, start at first min week in file
        return int(sub['week'].min())
    return None

# ---------------------------------------------------------------------------
# Odds API fetch
# ---------------------------------------------------------------------------

def fetch_odds(api_key: str, sport: str, regions: str, markets: str, odds_format: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        'regions': regions,
        'markets': markets,
        'oddsFormat': odds_format,
        'apiKey': api_key,
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:400]}")
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse odds JSON: {e}")

# ---------------------------------------------------------------------------
# Build mapping from events to provider lines per game
# ---------------------------------------------------------------------------

def _select_predictions_frame(week: int) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(PRED_FILES_GLOB), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or 'season' not in df.columns:
            continue
        sub = df[(df['season'] == YEAR) & (df.get('week') == week)]
        if not sub.empty:
            return sub
    raise FileNotFoundError(f"No predictions file with week {week} found")


def _extract_markets(bookmaker: Dict[str, Any], home_team: str, away_team: str) -> Dict[str, Any]:
    spread_val = None
    total_val = None
    home_ml = None
    away_ml = None
    markets = bookmaker.get('markets', [])
    for m in markets:
        key = m.get('key')
        outcomes = m.get('outcomes', [])
        if key == 'spreads':
            # outcomes contain team, point, price
            for o in outcomes:
                t = norm_team(o.get('name',''))
                if t == norm_team(home_team):
                    try: spread_val = float(o.get('point'))
                    except Exception: pass
                elif t == norm_team(away_team):
                    # If we only saw away spread (positive), derive home as negative
                    if spread_val is None:
                        try:
                            p = float(o.get('point'))
                            spread_val = -p
                        except Exception:
                            pass
        elif key == 'totals':
            # two outcomes: Over / Under with point
            for o in outcomes:
                try:
                    pt = float(o.get('point'))
                except Exception:
                    continue
                # prefer the first valid point; assume both have same total
                total_val = pt
                break
        elif key == 'h2h':
            for o in outcomes:
                t = norm_team(o.get('name',''))
                price = o.get('price')
                if t == norm_team(home_team):
                    home_ml = price
                elif t == norm_team(away_team):
                    away_ml = price
    provider_entry = {
        'provider': bookmaker.get('title') or bookmaker.get('key'),
        'spread': spread_val,
        'formattedSpread': (f"{home_team} {spread_val:+.1f}" if spread_val is not None else ''),
        'overUnder': total_val,
        'homeMoneyline': home_ml,
        'awayMoneyline': away_ml,
    }
    return provider_entry


def build_lines_rows(week: int, odds_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pred_df = _select_predictions_frame(week)
    # Map normalized pair -> schedule row
    schedule_index = {}
    for _, r in pred_df.iterrows():
        ht = str(r['home_team']); at = str(r['away_team'])
        schedule_index[(norm_team(ht), norm_team(at))] = (ht, at)
    rows = []
    unmatched = []
    for ev in odds_events:
        home = ev.get('home_team'); away = ev.get('away_team')
        if not home or not away:
            continue
        key = (norm_team(home), norm_team(away))
        if key not in schedule_index:
            # Try reversed orientation (Odds API may list differently)
            key_rev = (norm_team(away), norm_team(home))
            if key_rev in schedule_index:
                home, away = away, home
                key = key_rev
            else:
                unmatched.append({'home': home, 'away': away})
                continue
        (sched_home, sched_away) = schedule_index[key]
        provs = []
        for bookmaker in ev.get('bookmakers', []):
            try:
                provs.append(_extract_markets(bookmaker, sched_home, sched_away))
            except Exception:
                continue
        if not provs:
            continue
        rows.append({
            'year': YEAR,
            'week': week,
            'homeTeam': sched_home,
            'awayTeam': sched_away,
            'lines': json.dumps(provs, separators=(',',':')),
        })
    if unmatched:
        print(f"[warn] Unmatched odds events: {len(unmatched)}", file=sys.stderr)
    return rows

# ---------------------------------------------------------------------------
# Merge & write
# ---------------------------------------------------------------------------

def merge_and_write(rows: List[Dict[str, Any]], week: int):
    new_df = pd.DataFrame(rows)
    if os.path.exists(LINES_PATH):
        try:
            old = pd.read_csv(LINES_PATH)
        except Exception:
            old = pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
    else:
        old = pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
    # Drop existing rows for this (year, week)
    mask = ~((old.get('year',0)==YEAR) & (old.get('week',0)==week))
    merged = pd.concat([old[mask], new_df], ignore_index=True)
    # Sort for stability
    try:
        merged['week_int'] = pd.to_numeric(merged['week'], errors='coerce')
        merged.sort_values(['year','week_int','homeTeam','awayTeam'], inplace=True)
        merged.drop(columns=['week_int'], inplace=True)
    except Exception:
        pass
    tmp = LINES_PATH + '.tmp'
    merged.to_csv(tmp, index=False)
    os.replace(tmp, LINES_PATH)
    return {'written_rows': len(new_df), 'total_rows': len(merged), 'path': LINES_PATH}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week', type=int, default=None, help='Target week (default: auto-detect upcoming)')
    ap.add_argument('--api-key', type=str, default=None, help='Odds API Key (fallback ODDS_API_KEY env)')
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get('ODDS_API_KEY')
    if not api_key:
        print('[error] Missing Odds API key (provide --api-key or set ODDS_API_KEY).', file=sys.stderr)
        return 2
    week = args.week if args.week is not None else _detect_upcoming_week()
    if week is None:
        print('[error] Could not determine upcoming week.', file=sys.stderr)
        return 3

    sport = os.environ.get('ODDS_API_SPORT', 'americanfootball_ncaaf')
    regions = os.environ.get('ODDS_API_REGIONS', 'us')
    markets = os.environ.get('ODDS_API_MARKETS', 'h2h,spreads,totals')
    odds_format = os.environ.get('ODDS_API_ODDS_FORMAT', 'american')

    print(f"[info] Fetching OddsAPI sport={sport} week={week} regions={regions} markets={markets}")
    try:
        events = fetch_odds(api_key, sport, regions, markets, odds_format)
    except Exception as e:
        print(f"[error] fetch failed: {e}", file=sys.stderr)
        return 4
    rows = build_lines_rows(week, events)
    if not rows:
        print('[warn] No rows matched schedule / produced provider lines; nothing written.')
        return 0
    res = merge_and_write(rows, week)
    print(json.dumps({'status':'ok','week':week, **res}, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
