import os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
client = app.app.test_client()
resp = client.get('/api/game-cards?week=3')
print('status', resp.status_code)
js = resp.get_json()
print('week', js.get('week'), 'count', js.get('count'))
results = js.get('results', [])
synthetic_cards = sum(1 for g in results if any(l.get('synthetic') for l in (g.get('betting_lines') or [])))
no_lines = [ (g['home_team'], g['away_team']) for g in results if not g.get('betting_lines') ]
print('synthetic_cards', synthetic_cards)
print('no_line_games', len(no_lines))
print('sample_no_lines', no_lines[:10])
