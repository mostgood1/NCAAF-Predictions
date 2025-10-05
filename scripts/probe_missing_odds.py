import os, json
import pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data')
W_SCORES = os.path.join(DATA, 'college_football_schedule_2025_predicted_totals_enhanced_with_scores.csv')
ENH = os.path.join(DATA, 'college_football_schedule_2025_predicted_totals_enhanced.csv')
LINES = os.path.join(DATA, 'college_football_betting_lines_2025.csv')

def load_csv(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[err] failed to read {p}: {e}")
        return pd.DataFrame()

def find_games(df, substrs):
    if df.empty:
        return df
    m = False
    cols = []
    for col in ['home_team','away_team','venue']:
        if col in df.columns:
            cols.append(col)
    if not cols:
        return df.iloc[:0]
    mask = pd.Series(False, index=df.index)
    for s in substrs:
        for c in cols:
            mask = mask | df[c].astype(str).str.contains(s, case=False, na=False)
    return df.loc[mask]

def norm(name: str) -> str:
    import re
    try:
        s = str(name or '')
    except Exception:
        return ''
    s = s.strip().lower().replace('&',' and ').replace("ʻ","'").replace("’","'")
    s = re.sub(r"[^a-z0-9 '\-]"," ", s)
    s = s.replace("hawai'i","hawaii")
    s = re.sub(r"\s+"," ", s).strip()
    return s

if __name__ == '__main__':
    targets = ['Sam Houston', 'New Mexico State', 'Aggie Memorial']
    dfw = load_csv(W_SCORES)
    dfe = load_csv(ENH)
    dfl = load_csv(LINES)
    print('[with_scores] subset:')
    print(find_games(dfw, targets)[['season','week','start_date','home_team','away_team','venue']].head(20).to_string(index=False))
    print('\n[enhanced] subset:')
    print(find_games(dfe, targets)[['season','week','start_date','home_team','away_team','venue']].head(20).to_string(index=False))
    if not dfl.empty:
        print('\n[lines candidates in week 6 with either team]')
        try:
            sub = dfl[(dfl.get('year',0)==2025) & (dfl.get('week',0)==6)].copy()
        except Exception:
            sub = dfl.iloc[:0]
        mask = (sub['homeTeam'].astype(str).str.contains('Sam Houston|New Mexico State', case=False, na=False)) | (sub['awayTeam'].astype(str).str.contains('Sam Houston|New Mexico State', case=False, na=False))
        print(sub.loc[mask].head(50).to_string(index=False))
    # Print norms to inspect matching
    print('\n[norm examples]')
    for t in ['Sam Houston State Bearkats', 'Sam Houston', 'New Mexico State Aggies', 'New Mexico State']:
        print(t, '->', norm(t))
