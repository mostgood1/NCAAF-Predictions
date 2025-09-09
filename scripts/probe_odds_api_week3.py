import os, json, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from fetch_2025_lines import fetch_odds
api_key=os.environ.get('ODDS_API_KEY')
if not api_key: raise SystemExit('no api key')
SPORT=os.environ.get('ODDS_API_SPORT','americanfootball_ncaaf')
regions=os.environ.get('ODDS_API_REGIONS','us,us2')
markets='h2h,spreads,totals'
print('Fetching raw odds...')
raw=fetch_odds(api_key, SPORT, regions, markets, os.environ.get('ODDS_API_ODDS_FORMAT','american'))
print('events',len(raw))
# find event with Arizona vs Kansas State
for ev in raw:
    ht=ev.get('home_team'); at=ev.get('away_team')
    if 'Arizona' in (ht or '') or 'Arizona' in (at or ''):
        if 'Kansas State' in (ht or '') or 'Kansas State' in (at or ''):
            print('Found candidate event:')
            print(json.dumps(ev, indent=2)[:4000])
            break
