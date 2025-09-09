import pandas as pd, json

df=pd.read_csv('data/college_football_betting_lines_2025.csv')
w3=df[(df.year==2025)&(df.week==3)]
print('week3 total rows',len(w3))
ml_missing=[]; spread_missing=[]
for _,r in w3.iterrows():
    lines=json.loads(r.lines)
    has_ml=any(l.get('homeMoneyline') is not None or l.get('awayMoneyline') is not None for l in lines)
    has_spread=any(l.get('spread') is not None for l in lines)
    if not has_ml: ml_missing.append((r.homeTeam,r.awayTeam))
    if not has_spread: spread_missing.append((r.homeTeam,r.awayTeam))
print('missing ml',len(ml_missing),'missing spread',len(spread_missing))
print('sample ml missing',ml_missing[:10])
print('sample spread missing',spread_missing[:10])
