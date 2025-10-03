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

"""Team name normalization & aliasing.

This now closely mirrors the logic in app._norm_team_for_odds plus expanded
alias coverage for Odds API naming conventions (mascot suffixes, full school
names vs abbreviations, etc.). We also implement a fallback that progressively
strips trailing tokens (e.g., 'north carolina state wolfpack' ->
'north carolina state') when attempting to match schedule pairs.
"""
import re
_space_re = re.compile(r"\s+")

def _base_norm(name: str) -> str:
    try:
        s = str(name or '')
    except Exception:
        return ''
    s = s.strip().lower()
    s = s.replace('&', ' and ')
    # unify different apostrophes
    s = s.replace("ʻ", "'").replace("’", "'")
    # keep letters/numbers/space/apostrophe/hyphen
    s = re.sub(r"[^a-z0-9 '\-]", " ", s)
    s = s.replace("hawai'i", "hawaii")
    s = _space_re.sub(' ', s).strip()
    return s

# Alias map (keys & values are post _base_norm). Includes canonical collapse of mascots
ALIASES = {
    # Existing / prior
    'miami oh': 'miami ohio', 'miami ohio': 'miami ohio',
    'ole miss': 'mississippi', 'miss st': 'mississippi state',
    'utsa': 'texas san antonio', 'ut san antonio': 'texas san antonio',
    'utsa roadrunners': 'texas san antonio',
    'app state': 'appalachian state', 'appalachian st': 'appalachian state', 'app st': 'appalachian state',
    'southern miss': 'southern mississippi',
    'hawaii': 'hawaii',
    'la lafayette': 'louisiana', 'louisiana lafayette': 'louisiana',
    'la monroe': 'louisiana monroe',
    'umass': 'massachusetts', 'uconn': 'connecticut',
    'byu cougars': 'byu', 'byu': 'byu',
    'ucf': 'ucf', 'central florida': 'ucf', 'central florida knights': 'ucf',
    'usf': 'usf', 'south florida': 'usf', 'south florida bulls': 'usf',
    # Common abbreviation expansions
    'lsu': 'lsu', 'louisiana state': 'lsu', 'louisiana state tigers': 'lsu',
    'tcu': 'tcu', 'texas christian': 'tcu', 'texas christian horned frogs': 'tcu',
    'usc': 'usc', 'southern california': 'usc', 'southern california trojans': 'usc',
    'smu': 'smu', 'southern methodist': 'smu', 'southern methodist mustangs': 'smu',
    'uab': 'uab', 'alabama birmingham': 'uab', 'alabama birmingham blazers': 'uab',
    'utep': 'utep', 'texas el paso': 'utep', 'texas el paso miners': 'utep',
    'utsa roadrunners': 'texas san antonio',
    'texas a&m': 'texas am', 'texas a and m': 'texas am', 'texas am': 'texas am',
    'penn st': 'penn state', 'penn state nittany lions': 'penn state',
    'mich st': 'michigan state', 'michigan st': 'michigan state',
    'florida st': 'florida state', 'florida state seminoles': 'florida state',
    'boise st': 'boise state', 'boise state broncos': 'boise state',
    'san jose state': 'san jose state', 'san josé state': 'san jose state', 'sjsu': 'san jose state',
    'texas san antonio': 'texas san antonio',
    # Parenthetical state schools
    'miami fl': 'miami', 'miami florida': 'miami',
    # Mascot forms -> school canonical
    'ole miss rebels': 'mississippi',
    'arkansas razorbacks': 'arkansas',
    'umass minutemen': 'massachusetts',
    'notre dame fighting irish': 'notre dame',
    'texas a m aggies': 'texas am', 'texas am aggies': 'texas am', 'texas a and m aggies': 'texas am',
    'miami hurricanes': 'miami',
    'florida gators': 'florida', 'florida state seminoles': 'florida state',
    'georgia bulldogs': 'georgia', 'alabama crimson tide': 'alabama',
    'penn state nittany lions': 'penn state', 'oregon ducks': 'oregon',
    'nebraska cornhuskers': 'nebraska', 'michigan wolverines': 'michigan',
    'lsu tigers': 'lsu', 'uconn huskies': 'connecticut', 'delaware blue hens': 'delaware',
}

def norm_team(name: str) -> str:
    b = _base_norm(name)
    return ALIASES.get(b, b)

def best_schedule_norm(raw: str, schedule_norm_set: set[str]) -> str:
    """Return the normalized form most likely to match schedule names.

    Strategy: exact alias/normalized match, else iteratively trim trailing tokens
    (to discard mascot words) until a match arises, else original normalized.
    """
    n = norm_team(raw)
    if n in schedule_norm_set:
        return n
    tokens = n.split()
    while len(tokens) > 1:
        tokens = tokens[:-1]
        cand = ' '.join(tokens)
        if cand in schedule_norm_set:
            return cand
    return n

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
    home_norm = norm_team(home_team)
    away_norm = norm_team(away_team)
    def _belongs(out_name_norm: str, base_norm: str) -> bool:
        if out_name_norm == base_norm:
            return True
        # Allow mascot extension (e.g., 'arizona wildcats' startswith 'arizona ')
        if out_name_norm.startswith(base_norm + ' '):
            return True
        return False
    for m in markets:
        key = m.get('key')
        outcomes = m.get('outcomes', [])
        if key == 'spreads':
            # outcomes contain team, point, price
            for o in outcomes:
                t = norm_team(o.get('name',''))
                if _belongs(t, home_norm):
                    try: spread_val = float(o.get('point'))
                    except Exception: pass
                elif _belongs(t, away_norm):
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
                if _belongs(t, home_norm):
                    home_ml = price
                elif _belongs(t, away_norm):
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


def build_lines_rows(week: int, odds_events: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    pred_df = _select_predictions_frame(week)
    # Map normalized pair -> schedule row
    schedule_index: dict[tuple[str,str], tuple[str,str]] = {}
    schedule_team_norms: set[str] = set()
    # Build schedule index for normalization lookups

    for _, r in pred_df.iterrows():
        ht = str(r['home_team']); at = str(r['away_team'])
        n_ht = norm_team(ht); n_at = norm_team(at)
        schedule_index[(n_ht, n_at)] = (ht, at)
        schedule_team_norms.add(n_ht); schedule_team_norms.add(n_at)
    # Determine approximate temporal bounds for the week (if start_date present)
    week_start = None; week_end = None
    if 'start_date' in pred_df.columns:
        try:
            sd = pd.to_datetime(pred_df['start_date'], errors='coerce')
            if not sd.isna().all():
                week_start = sd.min() - pd.Timedelta(hours=6)
                week_end = sd.max() + pd.Timedelta(hours=6)
        except Exception:
            pass
    rows: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    skipped_time = 0
    for ev in odds_events:
        raw_home = ev.get('home_team'); raw_away = ev.get('away_team')
        if not raw_home or not raw_away:
            continue
        if week_start is not None and week_end is not None:
            ct = ev.get('commence_time') or ev.get('commenceTime')
            if ct:
                try:
                    ctd = pd.to_datetime(ct, utc=True)
                    if ctd < week_start or ctd > week_end:
                        skipped_time += 1
                        continue
                except Exception:
                    pass
        # Best-match normalization (account for mascots / suffixes)
        n_home = best_schedule_norm(raw_home, schedule_team_norms)
        n_away = best_schedule_norm(raw_away, schedule_team_norms)
        key = (n_home, n_away)
        if key not in schedule_index:
            key_rev = (n_away, n_home)
            if key_rev in schedule_index:
                key = key_rev
            else:
                unmatched.append({'home': raw_home, 'away': raw_away, 'norm_home': n_home, 'norm_away': n_away})
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
    if skipped_time and debug:
        print(f"[info] Skipped events outside inferred week window: {skipped_time}", file=sys.stderr)
    if unmatched and debug:
        sample = unmatched[:10]
        print('[debug] sample unmatched events:', json.dumps(sample, indent=2), file=sys.stderr)
    return rows

# ---------------------------------------------------------------------------
# Helpers for ML enhancement / merging second-pass h2h fetch
# ---------------------------------------------------------------------------

def _classify_fbs_pairs(pred_df: pd.DataFrame) -> set[tuple[str,str]]:
    fbs_confs = {
        'acc','sec','big ten','big 12','pac 12','american','mountain west','sun belt','mac','conference usa','independent','independents','fbs independents','independent (fbs)'
    }
    fbs_indies = {'notre dame','army','navy','umass','uconn','new mexico state'}
    pairs: set[tuple[str,str]] = set()
    for _, r in pred_df.iterrows():
        try:
            ht = str(r['home_team']); at = str(r['away_team'])
            hc = str(r.get('home_conference','') or '').lower().strip()
            ac = str(r.get('away_conference','') or '').lower().strip()
            def _is_fbs(team, conf):
                t = str(team or '').lower().strip()
                return conf in fbs_confs or t in fbs_indies
            if _is_fbs(ht, hc) and _is_fbs(at, ac):
                pairs.add((ht, at))
        except Exception:
            continue
    return pairs

def _rows_to_index(rows: List[Dict[str, Any]]) -> dict[tuple[str,str], Dict[str, Any]]:
    idx: dict[tuple[str,str], Dict[str, Any]] = {}
    for r in rows:
        try:
            idx[(r['homeTeam'], r['awayTeam'])] = r
        except Exception:
            continue
    return idx

def _merge_provider_entries(base_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]):
    """Merge provider lines, filling in missing markets (spread/total/ML) per provider.

    - Adds entirely new game rows when absent in base_rows.
    - When provider exists for a game, only fills missing fields from new_rows (does not overwrite existing values).
    """
    base_idx = _rows_to_index(base_rows)
    for r in new_rows:
        k = (r['homeTeam'], r['awayTeam'])
        if k not in base_idx:
            base_rows.append(r)
            base_idx[k] = r
            continue
        try:
            existing_lines = json.loads(base_idx[k]['lines']) if isinstance(base_idx[k]['lines'], str) else base_idx[k]['lines']
            new_lines = json.loads(r['lines']) if isinstance(r['lines'], str) else r['lines']
            prov_map = { (pl.get('provider') or '').lower(): pl for pl in existing_lines }
            for nl in new_lines:
                pname = (nl.get('provider') or '').lower()
                if not pname:
                    continue
                if pname in prov_map:
                    pl = prov_map[pname]
                    # Fill missing spread/total/ML values
                    if (pl.get('spread') is None or pl.get('spread')=='') and nl.get('spread') is not None:
                        pl['spread'] = nl.get('spread')
                        pl['formattedSpread'] = nl.get('formattedSpread', pl.get('formattedSpread',''))
                    if (pl.get('overUnder') is None or pl.get('overUnder')=='') and nl.get('overUnder') is not None:
                        pl['overUnder'] = nl.get('overUnder')
                    if (pl.get('homeMoneyline') is None or pl.get('homeMoneyline')=='') and nl.get('homeMoneyline') is not None:
                        pl['homeMoneyline'] = nl.get('homeMoneyline')
                    if (pl.get('awayMoneyline') is None or pl.get('awayMoneyline')=='') and nl.get('awayMoneyline') is not None:
                        pl['awayMoneyline'] = nl.get('awayMoneyline')
                else:
                    existing_lines.append(nl)
            base_idx[k]['lines'] = json.dumps(existing_lines, separators=(',',':'))
        except Exception:
            continue

# ---------------------------------------------------------------------------
# Merge & write
# ---------------------------------------------------------------------------

def merge_and_write(rows: List[Dict[str, Any]], week: int):
    """Merge new odds rows into the CSV without dropping unmatched games for the week.

    Prior behavior removed all rows for the target week, which caused odds to disappear
    for games once The Odds API stopped returning them (e.g., after kickoff). We now
    only replace keys that appear in the new fetch and preserve other existing rows.
    """
    new_df = pd.DataFrame(rows)
    if os.path.exists(LINES_PATH):
        try:
            old = pd.read_csv(LINES_PATH)
        except Exception:
            old = pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])
    else:
        old = pd.DataFrame(columns=['year','week','homeTeam','awayTeam','lines'])

    # Build key sets for efficient merge: (year, week, homeTeam, awayTeam)
    def _key_iter(df: pd.DataFrame):
        for _, r in df.iterrows():
            try:
                yield (int(r.get('year', YEAR)), int(r.get('week', week)), str(r.get('homeTeam','')), str(r.get('awayTeam','')))
            except Exception:
                continue
    try:
        new_keys = set(_key_iter(new_df))
    except Exception:
        new_keys = set()

    # Keep everything outside the target YEAR/WEEK as-is.
    base_mask = ~((old.get('year', 0) == YEAR) & (old.get('week', 0) == week))
    keep_rows = old[base_mask].copy()

    # Within the target YEAR/WEEK, preserve rows not in new_keys; replace rows that are in new_keys.
    wk_mask = ((old.get('year', 0) == YEAR) & (old.get('week', 0) == week))
    if 'homeTeam' in old.columns and 'awayTeam' in old.columns:
        def _row_key(r):
            try:
                return (int(r['year']), int(r['week']), str(r['homeTeam']), str(r['awayTeam']))
            except Exception:
                return None
        preserved_same_week = [r for _, r in old[wk_mask].iterrows() if _row_key(r) not in new_keys]
        if preserved_same_week:
            keep_rows = pd.concat([keep_rows, pd.DataFrame(preserved_same_week)], ignore_index=True)

    # Append new rows (they will replace any existing keys that we purposely did not carry over above)
    merged = pd.concat([keep_rows, new_df], ignore_index=True)

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
    return {'written_rows': len(new_df), 'preserved_rows': int(len(merged) - len(new_df) - len(old[~base_mask])), 'total_rows': len(merged), 'path': LINES_PATH}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week', type=int, default=None, help='Target week (default: auto-detect upcoming)')
    ap.add_argument('--api-key', type=str, default=None, help='Odds API Key (fallback ODDS_API_KEY env or secrets file)')
    ap.add_argument('--debug', action='store_true', help='Enable debug logging (sample unmatched)')
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get('ODDS_API_KEY')
    if not api_key:
        # fallback to secrets/odds_api_key.txt
        try:
            sec_path = os.path.join(BASE_DIR, 'secrets', 'odds_api_key.txt')
            if os.path.exists(sec_path):
                with open(sec_path, 'r', encoding='utf-8') as fh:
                    api_key = fh.read().strip()
        except Exception:
            pass
    if not api_key:
        print('[error] Missing Odds API key (provide --api-key or set ODDS_API_KEY).', file=sys.stderr)
        return 2
    week = args.week if args.week is not None else _detect_upcoming_week()
    if week is None:
        print('[error] Could not determine upcoming week.', file=sys.stderr)
        return 3

    sport = os.environ.get('ODDS_API_SPORT', 'americanfootball_ncaaf')
    # Broaden default regions to include additional books which may post earlier (can override via env)
    regions = os.environ.get('ODDS_API_REGIONS', 'us,us2,eu,uk')
    markets = os.environ.get('ODDS_API_MARKETS', 'h2h,spreads,totals')
    odds_format = os.environ.get('ODDS_API_ODDS_FORMAT', 'american')

    print(f"[info] Fetching OddsAPI sport={sport} week={week} regions={regions} markets={markets}")
    try:
        events = fetch_odds(api_key, sport, regions, markets, odds_format)
    except Exception as e:
        print(f"[error] fetch failed: {e}", file=sys.stderr)
        return 4
    if args.debug:
        try:
            print(f"[info] fetched events count: {len(events)}", file=sys.stderr)
        except Exception:
            pass
    rows = build_lines_rows(week, events, debug=args.debug)
    # Attempt second-pass h2h fetch if FBS vs FBS moneylines missing
    try:
        pred_df = _select_predictions_frame(week)
        fbs_pairs = _classify_fbs_pairs(pred_df)
        # Determine which FBS vs FBS pairs lack any moneyline
        rows_idx = _rows_to_index(rows)
        missing_ml = []
        for (ht, at) in fbs_pairs:
            r = rows_idx.get((ht, at))
            if not r:
                missing_ml.append((ht, at))
                continue
            try:
                line_list = json.loads(r['lines']) if isinstance(r['lines'], str) else r['lines']
            except Exception:
                line_list = []
            has_ml = any( (pl.get('homeMoneyline') is not None or pl.get('awayMoneyline') is not None) for pl in line_list )
            if not has_ml:
                missing_ml.append((ht, at))
        if missing_ml:
            extra_regions = os.environ.get('ODDS_API_H2H_EXTRA_REGIONS', 'us,us2,eu,uk')
            if args.debug:
                print(f"[info] Second-pass h2h fetch for missing ML games: {len(missing_ml)} regions={extra_regions}", file=sys.stderr)
            try:
                h2h_events = fetch_odds(api_key, sport, extra_regions, 'h2h', odds_format)
                h2h_rows = build_lines_rows(week, h2h_events, debug=args.debug)
                _merge_provider_entries(rows, h2h_rows)
            except Exception as e:
                if args.debug:
                    print(f"[warn] second-pass h2h fetch failed: {e}", file=sys.stderr)
    except Exception as e:
        if args.debug:
            print(f"[warn] ML enhancement logic failed: {e}", file=sys.stderr)
    # Optional second pass for spreads/totals if many games lack those markets
    try:
        need_market_fill = []
        rows_idx2 = _rows_to_index(rows)
        for (ht, at), r in rows_idx2.items():
            try:
                line_list = json.loads(r['lines']) if isinstance(r['lines'], str) else r['lines']
            except Exception:
                line_list = []
            has_spread = any(pl.get('spread') is not None for pl in line_list)
            has_total = any(pl.get('overUnder') is not None for pl in line_list)
            if not (has_spread and has_total):
                need_market_fill.append((ht, at))
        if need_market_fill:
            extra_regions_st = os.environ.get('ODDS_API_SPREADS_TOTALS_EXTRA_REGIONS', 'us,us2,eu,uk')
            if args.debug:
                print(f"[info] Second-pass spreads/totals fetch for games missing markets: {len(need_market_fill)} regions={extra_regions_st}", file=sys.stderr)
            try:
                st_events = fetch_odds(api_key, sport, extra_regions_st, 'spreads,totals', odds_format)
                st_rows = build_lines_rows(week, st_events, debug=args.debug)
                _merge_provider_entries(rows, st_rows)
            except Exception as e:
                if args.debug:
                    print(f"[warn] second-pass spreads/totals fetch failed: {e}", file=sys.stderr)
    except Exception as e:
        if args.debug:
            print(f"[warn] spreads/totals enhancement logic failed: {e}", file=sys.stderr)
    if not rows:
        print('[warn] No rows matched schedule / produced provider lines; nothing written.')
        return 0
    res = merge_and_write(rows, week)
    # Print compact stats to aid coverage diagnostics
    try:
        total_games = len(_select_predictions_frame(week))
    except Exception:
        total_games = None
    stats = {
        'status': 'ok',
        'week': week,
        **res,
        'matched_games': len(rows),
        'pred_games_in_week': total_games,
    }
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
