import os, sys, json
BASE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(BASE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
client=app.app.test_client()
js=client.get('/api/game-cards?week=3&full=1').get_json()
res=js.get('results', [])
from collections import defaultdict
agg=defaultdict(lambda: {'games':0,'totals':0})
for g in res:
    lvl=g.get('matchup_level') or 'Unknown'
    agg[lvl]['games']+=1
    lines=g.get('betting_lines') or []
    if any(l.get('overUnder') is not None for l in lines):
        agg[lvl]['totals']+=1
print(json.dumps({'week': js.get('week'), 'totals_breakdown': agg}, indent=2))
