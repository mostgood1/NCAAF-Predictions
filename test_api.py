from app import app

with app.test_client() as c:
    resp = c.get('/api/game-cards?week=2&full=1')
    print('status', resp.status_code)
    if resp.status_code==200:
        data=resp.get_json()
        rows = data.get('results', [])
        finals = [r for r in rows if r.get('actual_home_points') is not None and r.get('actual_away_points') is not None]
        print('week', data.get('week'), 'total', len(rows), 'finals', len(finals))
        # Show first 3 rows overall
        for g in rows[:3]:
            print('SAMPLE', g.get('home_team'), 'vs', g.get('away_team'), 'PH', g.get('predicted_home_points'), 'PA', g.get('predicted_away_points'), 'AH', g.get('actual_home_points'), 'AA', g.get('actual_away_points'))
        # Show one final if exists
        if finals:
            f = finals[0]
            print('FINAL_SAMPLE', f.get('home_team'), f.get('actual_home_points'), '-', f.get('away_team'), f.get('actual_away_points'))
