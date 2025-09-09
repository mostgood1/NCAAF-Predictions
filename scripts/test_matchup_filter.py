import sys, os, json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app
c=app.app.test_client()
js_all=c.get('/api/game-cards?week=3&full=1').get_json()
js_fbs=c.get('/api/game-cards?week=3&full=1&matchup=FBSvFBS').get_json()
print(json.dumps({'all_count': js_all['count'], 'fbsvfbs_count': js_fbs['count']}, indent=2))
