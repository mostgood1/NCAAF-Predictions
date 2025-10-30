import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import app
from weekly_update import _detect_weeks_from_pred_df

prior, upcoming, _ = _detect_weeks_from_pred_df()
week = prior or upcoming

df = app.pred_df.copy()
if week is not None:
    df = df[df['week']==int(week)]

print({'week': week, 'rows': int(len(df))})
cols = ['home_team','away_team','home_conference','away_conference']
print(df[cols].head(20).to_dict(orient='records'))

unknown_rows = df[(df['home_conference']=='Unknown') | (df['away_conference']=='Unknown')]
print({'unknown_rows': int(len(unknown_rows))})
print('Sample unknown rows:')
print(unknown_rows[cols].head(30).to_dict(orient='records'))
