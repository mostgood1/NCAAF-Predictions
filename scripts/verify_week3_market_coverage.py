import os, sys, json
BASE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(BASE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
c=app.app.test_client()
resp=c.get('/api/game-cards?week=3&full=1')
js=resp.get_json()
rows=js.get('results', [])
spread_ok=0
ml_ok=0
for g in rows:
    lines=g.get('betting_lines') or []
    # consider a spread present if any provider has non-null spread
    if any(l.get('spread') is not None for l in lines):
        spread_ok+=1
    # moneyline if home OR away ML present
    if any(l.get('homeMoneyline') is not None or l.get('awayMoneyline') is not None for l in lines):
        ml_ok+=1
print(json.dumps({
    'week': js.get('week'),
    'game_count': len(rows),
    'games_with_spread': spread_ok,
    'games_with_moneyline': ml_ok,
    'sample_missing_spread': [ (g['home_team'], g['away_team']) for g in rows if not any(l.get('spread') is not None for l in (g.get('betting_lines') or [])) ][:5],
    'sample_missing_ml': [ (g['home_team'], g['away_team']) for g in rows if not any(l.get('homeMoneyline') is not None or l.get('awayMoneyline') is not None for l in (g.get('betting_lines') or [])) ][:5]
}, indent=2))
