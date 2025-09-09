import os, sys, json
import pandas as pd
from datetime import datetime
BASE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(BASE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import app  # noqa
from app import app as flask_app

# Usage: python scripts/enrich_edges_week.py 3
week=int(sys.argv[1]) if len(sys.argv)>1 else 3
client=flask_app.test_client()
resp=client.get(f'/api/game-cards?week={week}&full=1')
js=resp.get_json()
rows=[]
for g in js.get('results', []):
    # Use existing ats_edge_num, ou_edge_num and moneyline providers
    lines=g.get('betting_lines') or []
    best_spread=None
    best_total=None
    # For moneyline, compute expected value vs model prob for each provider if both MLs exist
    p_home=g.get('home_win_prob')
    ml_entries=[]
    if p_home is not None:
        for bl in lines:
            hml=bl.get('homeMoneyline'); aml=bl.get('awayMoneyline')
            if hml is None or aml is None: continue
            # EV for betting home or away (per unit risked)
            try:
                # American -> risk/reward
                def ev(odds, prob_win):
                    o=float(odds)
                    if o>0:
                        return prob_win*(o/100.0) - (1-prob_win)
                    else:
                        return prob_win*(100.0/abs(o)) - (1-prob_win)
                ev_home=ev(hml, p_home)
                ev_away=ev(aml, 1-p_home)
            except Exception:
                continue
            ml_entries.append({'provider': bl.get('provider'),'homeMoneyline':hml,'awayMoneyline':aml,'ev_home':ev_home,'ev_away':ev_away})
    # Choose best ML edge (absolute) for summary
    best_ev=None; best_ev_side=None; best_ev_provider=None
    for e in ml_entries:
        for side, val in (('home', e['ev_home']), ('away', e['ev_away'])):
            if val is None: continue
            if best_ev is None or val>best_ev:
                best_ev=val; best_ev_side=side; best_ev_provider=e['provider']
    rows.append({
        'season': 2025,
        'week': week,
        'home_team': g['home_team'],
        'away_team': g['away_team'],
        'ats_edge': g.get('ats_edge_num'),
        'ou_edge': g.get('ou_edge_num'),
        'best_ml_ev': best_ev,
        'best_ml_side': best_ev_side,
        'best_ml_provider': best_ev_provider,
        'timestamp': datetime.utcnow().isoformat()+'Z'
    })

out_path=f"data/college_football_schedule_2025_with_edges_week{week}.csv"
pd.DataFrame(rows).to_csv(out_path, index=False)
print(json.dumps({'status':'ok','week':week,'rows':len(rows),'path':out_path}, indent=2))
