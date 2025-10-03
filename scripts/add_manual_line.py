import argparse, os, json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LINES_PATH = os.path.join(DATA_DIR, 'college_football_betting_lines_2025.csv')
YEAR = 2025

parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, required=True)
parser.add_argument('--home', type=str, required=True)
parser.add_argument('--away', type=str, required=True)
parser.add_argument('--provider', type=str, default='Manual')
parser.add_argument('--spread', type=float, default=None, help='Home-team spread (e.g., -10.5 means home favored by 10.5)')
parser.add_argument('--total', type=float, default=None)
parser.add_argument('--home-ml', type=int, default=None)
parser.add_argument('--away-ml', type=int, default=None)
args = parser.parse_args()

if not os.path.exists(LINES_PATH):
    raise SystemExit(f"Lines CSV not found at {LINES_PATH}")

df = pd.read_csv(LINES_PATH)

# Find an existing row for this game/week or create one
mask = (df.get('year', 0) == YEAR) & (df.get('week', 0) == args.week) & (df.get('homeTeam','') == args.home) & (df.get('awayTeam','') == args.away)
if not mask.any():
    # Append new row
    providers = [{
        'provider': args.provider,
        'spread': args.spread,
        'formattedSpread': f"{args.home} {args.spread:+.1f}" if args.spread is not None else '',
        'overUnder': args.total,
        'homeMoneyline': args.home_ml,
        'awayMoneyline': args.away_ml,
    }]
    new_row = {
        'year': YEAR,
        'week': args.week,
        'homeTeam': args.home,
        'awayTeam': args.away,
        'lines': json.dumps(providers, separators=(',',':')),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    # Merge provider into existing row (replace same provider or append)
    idx = df.index[mask][0]
    try:
        curr = df.at[idx, 'lines']
        pl = json.loads(curr) if isinstance(curr, str) else (curr or [])
    except Exception:
        pl = []
    # Replace if provider already exists
    lc_name = (args.provider or '').lower()
    replaced = False
    for p in pl:
        if (p.get('provider') or '').lower() == lc_name:
            p['spread'] = args.spread
            p['formattedSpread'] = f"{args.home} {args.spread:+.1f}" if args.spread is not None else p.get('formattedSpread','')
            p['overUnder'] = args.total
            p['homeMoneyline'] = args.home_ml
            p['awayMoneyline'] = args.away_ml
            replaced = True
            break
    if not replaced:
        pl.append({
            'provider': args.provider,
            'spread': args.spread,
            'formattedSpread': f"{args.home} {args.spread:+.1f}" if args.spread is not None else '',
            'overUnder': args.total,
            'homeMoneyline': args.home_ml,
            'awayMoneyline': args.away_ml,
        })
    df.at[idx, 'lines'] = json.dumps(pl, separators=(',',':'))

# Sort for stability
try:
    df['week_int'] = pd.to_numeric(df['week'], errors='coerce')
    df.sort_values(['year','week_int','homeTeam','awayTeam'], inplace=True)
    df.drop(columns=['week_int'], inplace=True)
except Exception:
    pass

tmp = LINES_PATH + '.tmp'
df.to_csv(tmp, index=False)
os.replace(tmp, LINES_PATH)
print(f"Wrote manual line for week {args.week}: {args.home} vs {args.away} at {LINES_PATH}")
