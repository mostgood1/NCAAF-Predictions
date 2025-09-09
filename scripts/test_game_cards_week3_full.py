import os, sys, json
BASE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(BASE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
c=app.app.test_client()
resp=c.get('/api/game-cards?week=3&full=1')
print('status', resp.status_code)
js=resp.get_json()
print('count', js.get('count'))
levels=[g.get('matchup_level') for g in js.get('results', [])[:10]]
print('sample_levels', levels)
# distribution
from collections import Counter
cnt=Counter(g.get('matchup_level') for g in js.get('results', []))
print('distribution', dict(cnt))
