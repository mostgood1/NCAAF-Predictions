import os, sys, json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
import pandas as pd
week = 3
pred = app.pred_df[app.pred_df.week==week]
# Consider FBS as both conferences non-null and non-empty
fbs = pred[pred.home_conference.notna() & pred.away_conference.notna()]
with_lines = 0
missing = []
real = 0
for _, r in fbs.iterrows():
    card = app._build_game_card(r)
    bl = card.get('betting_lines') or []
    if bl:
        with_lines += 1
        if not any(not x.get('synthetic') for x in bl):
            # only synthetic present
            missing.append((r.home_team, r.away_team, 'synthetic-only'))
        else:
            real += 1
    else:
        missing.append((r.home_team, r.away_team, 'no-lines'))
print(json.dumps({'total_fbs_games': len(fbs), 'cards_with_any_lines': with_lines, 'cards_with_real_lines': real, 'missing_samples': missing[:15]}, indent=2))
