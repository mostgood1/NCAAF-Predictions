import requests
import pandas as pd
import os

# Fetch FBS team data from College Football Data API
api_url = 'https://api.collegefootballdata.com/teams/fbs'
api_key = os.getenv('CFBD_API_KEY')
if not api_key:
    raise RuntimeError('CFBD_API_KEY environment variable not set. Get your key at CollegeFootballData.com.')
headers = {'Authorization': f'Bearer {api_key}'}
resp = requests.get(api_url, headers=headers)
try:
    teams = resp.json()
    if isinstance(teams, dict) and 'teams' in teams:
        teams = teams['teams']
    elif not isinstance(teams, list):
        raise ValueError(f"Unexpected API response format: {teams}")
    # Build DataFrame
    df = pd.DataFrame([
        {'school': t.get('school', ''), 'conference': t.get('conference', 'Unknown')}
        for t in teams if isinstance(t, dict)
    ])
    # Save to CSV
    out_path = 'src/data/team_conferences.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} teams to {out_path}")
except Exception as e:
    print(f"Error parsing API response: {e}")
