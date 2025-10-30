import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app
from weekly_update import _detect_weeks_from_pred_df

fbs_conf_aliases = app._fbs_conference_names()
indies = {'notre dame','army','navy','umass','uconn','new mexico state'}
FBS_NAMES = set(app.team_conf_df['school_norm'].astype(str).tolist()) if not app.team_conf_df.empty else set()

def _norm(s: str) -> str:
    return app._norm_team_for_conf(s)

def is_fbs(team, conf):
    t_norm = _norm(team)
    c = str(conf or '').strip().lower()
    return (t_norm in FBS_NAMES) or (c in fbs_conf_aliases) or (t_norm in indies)

prior, upcoming, _ = _detect_weeks_from_pred_df()
week = prior or upcoming

df = app.pred_df.copy()
if week is not None:
    df = df[df['week']==int(week)]

both_unknown = (df['home_conference']=='Unknown') & (df['away_conference']=='Unknown')
fbs_involved_by_name = df.apply(lambda r: is_fbs(r.get('home_team'), r.get('home_conference')) or is_fbs(r.get('away_team'), r.get('away_conference')), axis=1)

print({
    'week': week,
    'rows': int(len(df)),
    'unknown_both': int(both_unknown.sum()),
    'unknown_both_and_fbs_involved': int((both_unknown & fbs_involved_by_name).sum()),
})
