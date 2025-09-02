import importlib.util, os, json
base=os.path.dirname(os.path.dirname(__file__))
app_p=os.path.join(base,'app.py')
spec=importlib.util.spec_from_file_location('appmod', app_p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import pandas as pd
pred_p=os.path.join(base,'data','college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv')
df=pd.read_csv(pred_p)
row=df[(df['season']==2025)&(df['week']==2)&(df['home_team']=='Louisville')&(df['away_team']=='James Madison')].iloc[0]
card=m._build_game_card(row)
print(json.dumps({k:card[k] for k in ['ats_line','ats_model_lean','ats_edge','ou_line','ou_model_lean','ou_edge']}, indent=2))
