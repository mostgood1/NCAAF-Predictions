import json, os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app
c=app.app.test_client()
js=c.get('/api/game-cards?week=3&full=1').get_json()
first=js['results'][0]
print('edge keys:', [k for k in first if 'edge' in k])
print('edge values subset:', {k:first[k] for k in first if k.startswith('edge_')})
print('games', len(js['results']))
