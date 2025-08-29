from flask import Flask, render_template_string, request
import pandas as pd
import ast
import pytz
from datetime import datetime

app = Flask(__name__)

# Lazy-load predictions so the app can boot even if CSVs are missing
pred_df = None
PRED_PATHS = [
    "data/college_football_schedule_2025_predicted_totals_enhanced.csv",
    "data/college_football_schedule_2025_predicted_totals.csv",
    "src/data/college_football_schedule_2025_predicted_totals_enhanced.csv",
    "src/data/college_football_schedule_2025_predicted_totals.csv",
]
def _try_load_predictions():
    global pred_df
    if pred_df is not None:
        return pred_df
    for path in PRED_PATHS:
        try:
            df = pd.read_csv(path)
            pred_df = df
            return pred_df
        except Exception:
            continue
    return None

# Load team conferences (prefer root data/), coerce to strings for safe .str usage
team_conf_df = None
for path in [
    "data/team_conferences.csv",
    "src/data/team_conferences.csv",
]:
    try:
        team_conf_df = pd.read_csv(path)
        break
    except Exception:
        pass
if team_conf_df is None or not isinstance(team_conf_df, pd.DataFrame):
    # Create a minimal frame so the app still runs
    team_conf_df = pd.DataFrame({"school": [], "conference": []})
# Ensure required columns exist
for _col in ("school", "conference"):
    if _col not in team_conf_df.columns:
        team_conf_df[_col] = ""
# Coerce to string before using .str; normalize '&' and whitespace
def _normalize_series(s: pd.Series) -> pd.Series:
    try:
        return (
            s.astype(str)
             .fillna("")
             .str.strip()
             .str.lower()
             .str.replace('&', 'and', regex=False)
             .str.replace(r"\s+", " ", regex=True)
        )
    except Exception:
        return pd.Series([], dtype=str)
team_conf_df['school_norm'] = _normalize_series(team_conf_df['school'])

def norm(name):
    return str(name).strip().lower().replace('&', 'and').replace('  ', ' ')
conf_map = {}
try:
    if {'school_norm','conference'}.issubset(set(team_conf_df.columns)):
        # Coerce conference to string to avoid .str errors later
        conf_map = dict(zip(team_conf_df['school_norm'], team_conf_df['conference'].astype(str)))
except Exception:
    conf_map = {}

def _ensure_conf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df is pd.NA:
        return df
    try:
        if 'home_conference' not in df.columns:
            df['home_conference'] = df['home_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
        if 'away_conference' not in df.columns:
            df['away_conference'] = df['away_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
    except Exception:
        pass
    return df

# Load win margin confidence intervals (optional)
win_margin_conf_df = None
for path in [
    "data/win_margin_predictions_with_confidence.csv",
    "src/data/win_margin_predictions_with_confidence.csv",
]:
    try:
        win_margin_conf_df = pd.read_csv(path)
        win_margin_conf_df.columns = win_margin_conf_df.columns.str.strip()
        break
    except Exception:
        pass

# Load team assets (optional)
assets_df = None
for path in [
    "data/team_assets.csv",
    "src/data/team_assets.csv",
]:
    try:
        assets_df = pd.read_csv(path)
        break
    except Exception:
        pass
def get_team_asset(team_name):
    if assets_df is not None:
        row = assets_df[assets_df['school'] == team_name]
        if not row.empty:
            return {
                'logo': row.iloc[0].get('logo', ''),
                'color': row.iloc[0].get('color', ''),
                'alt_color': row.iloc[0].get('alt_color', '')
            }
    return {'logo': '', 'color': '', 'alt_color': ''}

# Load betting lines (optional)
lines_df = None
for path in [
    "data/college_football_betting_lines_last_15_years.csv",
    "src/data/college_football_betting_lines_last_15_years.csv",
]:
    try:
        lines_df = pd.read_csv(path)
        break
    except Exception:
        pass
def get_betting_lines(year, week, home_team, away_team):
    if lines_df is None:
        return []
    # Filter for matching year, week, home, and away teams (use correct column names)
    lines = lines_df[(lines_df['year'] == year) & (lines_df['week'] == week) &
                    (lines_df['homeTeam'] == home_team) & (lines_df['awayTeam'] == away_team)]
    if not lines.empty:
        odds_str = lines.iloc[0].get('lines', '')
        try:
            odds = ast.literal_eval(odds_str)
            if isinstance(odds, list):
                # Return list of dicts for template rendering
                return odds
        except Exception:
            pass
    return []

@app.route('/', methods=['GET', 'POST'])
def index():
    df = _try_load_predictions()
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        # Onboarding page when predictions are missing
        return render_template_string('''
        <style> body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6fa;padding:40px} .card{max-width:760px;margin:0 auto;background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);padding:28px} h2{margin-top:0;color:#2c3e50} code{background:#f3f4f7;padding:2px 6px;border-radius:4px} li{margin:6px 0} </style>
        <div class="card">
          <h2>Predictions CSV not found</h2>
          <p>Place one of the following files in your repository and redeploy:</p>
          <ul>
            <li><code>data/college_football_schedule_2025_predicted_totals_enhanced.csv</code></li>
            <li><code>data/college_football_schedule_2025_predicted_totals.csv</code></li>
          </ul>
          <p>Alternatively, update <code>app.py</code> to point at your actual predictions path.</p>
        </div>
        ''')
    # Ensure conference columns and proceed
    df = _ensure_conf_columns(df)
    weeks = sorted(df['week'].unique())
    selected_week = request.form.get('week', weeks[0])
    # Get all unique dates for the selected week (date only, no time)
    week_games = df[df['week'] == int(selected_week)].copy()
    week_games['date_only'] = week_games['start_date'].str[:10]
    all_dates = sorted(week_games['date_only'].dropna().unique())
    selected_date = request.form.get('date', '')
    show_all = request.form.get('show_all', '') == 'on'
    # Filter games by date if not showing all
    if selected_date and not show_all:
        filtered_games = week_games[week_games['date_only'] == selected_date]
    else:
        filtered_games = week_games
    games = filtered_games.apply(lambda row: f"{row['away_team']} at {row['home_team']}", axis=1).tolist()
    selected_game = request.form.get('game')
    game_info = None
    # If the selected game is not in the refreshed games list, reset selection
    if selected_game not in games:
        selected_game = None
    if selected_game:
        game_row = filtered_games.iloc[games.index(selected_game)]
        home_asset = get_team_asset(game_row['home_team'])
        away_asset = get_team_asset(game_row['away_team'])
        betting_lines = get_betting_lines(
            year=int(game_row['season']),
            week=int(game_row['week']),
            home_team=game_row['home_team'],
            away_team=game_row['away_team']
        )
        def r2(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return val
        # Parse start_date and convert to Central Time
        start_date_str = game_row.get('start_date', '')
        central_time_str = ''
        if start_date_str:
            try:
                dt_utc = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                dt_central = dt_utc.astimezone(pytz.timezone('US/Central'))
                central_time_str = dt_central.strftime('%A, %B %d, %Y %I:%M %p CT')
            except Exception:
                central_time_str = start_date_str

        # Get win margin confidence interval for this game
        conf_lower = conf_upper = conf_std = None
        def normalize_team_name(name):
            return str(name).strip().lower().replace('&', 'and').replace('  ', ' ')
        if win_margin_conf_df is not None:
            week_val = int(game_row.get('week', 0))
            season_val = int(game_row.get('season', 0))
            home_team = normalize_team_name(game_row.get('home_team', ''))
            away_team = normalize_team_name(game_row.get('away_team', ''))
            conf_row = win_margin_conf_df[(win_margin_conf_df['week'] == week_val) &
                                         (win_margin_conf_df['season'] == season_val) &
                                         (win_margin_conf_df['home_team'].apply(normalize_team_name) == home_team) &
                                         (win_margin_conf_df['away_team'].apply(normalize_team_name) == away_team)]
            if not conf_row.empty:
                conf_lower = r2(conf_row.iloc[0].get('conf_interval_lower', None))
                conf_upper = r2(conf_row.iloc[0].get('conf_interval_upper', None))
                conf_std = r2(conf_row.iloc[0].get('conf_std', None))

        game_info = {
            'home_team': game_row['home_team'],
            'away_team': game_row['away_team'],
            'venue': game_row.get('venue', ''),
            'game_time': central_time_str,
            'predicted_total_points': r2(game_row.get('predicted_total_points', '')),
            'predicted_home_points': r2(game_row.get('predicted_home_points', '')),
            'predicted_away_points': r2(game_row.get('predicted_away_points', '')),
            'predicted_win_margin': r2(game_row.get('predicted_win_margin', '')),
            'win_margin_conf_lower': conf_lower,
            'win_margin_conf_upper': conf_upper,
            'win_margin_conf_std': conf_std,
            'home_logo': home_asset['logo'],
            'home_color': home_asset['color'],
            'home_alt_color': home_asset['alt_color'],
            'away_logo': away_asset['logo'],
            'away_color': away_asset['color'],
            'away_alt_color': away_asset['alt_color'],
            'betting_lines': betting_lines,
        }
    return render_template_string('''
    <div style="text-align:right; margin: 12px 0 0 0;">
        <a href="/conference-records" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">View Projected Conference Records</a>
    </div>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
        .container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        form { display: flex; flex-direction: column; gap: 16px; margin-bottom: 32px; }
        label { font-weight: 500; color: #34495e; }
        select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 1em; }
        button { background: #2980b9; color: #fff; border: none; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #3498db; }
        .card { background: #f8f8f8; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.07); padding: 24px; margin-top: 16px; }
        .teams { display: flex; align-items: center; justify-content: center; gap: 32px; margin-bottom: 18px; }
        .team { text-align: center; }
        .team-logo { height: 60px; margin-bottom: 6px; }
        .team-name { font-weight: bold; font-size: 1.1em; padding: 4px 10px; border-radius: 6px; display: inline-block; margin-top: 2px; }
        .vs { font-size: 2em; color: #888; }
        ul.prediction { list-style: none; padding: 0; margin: 0 0 18px 0; }
        ul.prediction li { margin-bottom: 6px; font-size: 1.05em; }
        .odds-table { width: 100%; border-collapse: collapse; margin-top: 10px; background: #fff; }
        .odds-table th, .odds-table td { padding: 8px 10px; border: 1px solid #e0e0e0; text-align: center; }
        .odds-table th { background: #eaf1fb; color: #2c3e50; }
        .odds-table tr:nth-child(even) { background: #f4f6fa; }
        .no-odds { color: #888; font-style: italic; }
    </style>
    <div class="container">
        <h2>2025 NCAA Football Predictions</h2>
        <form method="post" id="mainForm">
            <label for="week">Select Week:</label>
            <select name="week" id="week" onchange="document.getElementById('mainForm').submit();">
                {% for w in weeks %}
                <option value="{{w}}" {% if w == selected_week|int %}selected{% endif %}>Week {{w}}</option>
                {% endfor %}
            </select>
            <label for="date">Select Date:</label>
            <select name="date" id="date" onchange="document.getElementById('mainForm').submit();">
                <option value="">All Dates</option>
                {% for d in all_dates %}
                <option value="{{d}}" {% if d == selected_date %}selected{% endif %}>{{d}}</option>
                {% endfor %}
            </select>
            <label><input type="checkbox" name="show_all" {% if show_all %}checked{% endif %} onchange="document.getElementById('mainForm').submit();"> Show all games for week</label>
            <label for="game">Select Game:</label>
            <select name="game" id="game">
                {% for g in games %}
                <option value="{{g}}" {% if g == selected_game %}selected{% endif %}>{{g}}</option>
                {% endfor %}
            </select>
            <button type="submit">Submit</button>
        </form>
        {% if game_info %}
        <div class="card">
            <div class="teams">
                <div class="team">
                    <img src="{{game_info['away_logo']}}" alt="{{game_info['away_team']}} logo" class="team-logo"><br>
                    <span class="team-name" style="color:{{game_info['away_color']}};background:{{game_info['away_alt_color']}};padding:4px 10px;border-radius:6px;display:inline-block;">{{game_info['away_team']}}</span>
                    <ul class="prediction">
                        <li><strong>Predicted Away Team Points:</strong> {{game_info['predicted_away_points']}}</li>
                    </ul>
                </div>
                <span class="vs">@</span>
                <div class="team">
                    <img src="{{game_info['home_logo']}}" alt="{{game_info['home_team']}} logo" class="team-logo"><br>
                    <span class="team-name" style="color:{{game_info['home_color']}};background:{{game_info['home_alt_color']}};padding:4px 10px;border-radius:6px;display:inline-block;">{{game_info['home_team']}}</span>
                    <ul class="prediction">
                        <li><strong>Predicted Home Team Points:</strong> {{game_info['predicted_home_points']}}</li>
                    </ul>
                </div>
            </div>
            <ul class="prediction" style="text-align:center;">
                <li><strong>Venue:</strong> {{game_info['venue']}}</li>
                <li><strong>Day & Time (Central):</strong> {{game_info['game_time']}}</li>
                <li><strong>Predicted Total Points:</strong> {{game_info['predicted_total_points']}}</li>
                <li><strong>Predicted Win Margin:</strong> {{game_info['predicted_win_margin']}}
                    {% if game_info['win_margin_conf_lower'] and game_info['win_margin_conf_upper'] %}
                        <br><span style="font-size:0.95em;color:#888;">95% CI: [{{game_info['win_margin_conf_lower']}}, {{game_info['win_margin_conf_upper']}}]</span>
                        <br><span style="font-size:0.95em;color:#888;">Std Dev: {{game_info['win_margin_conf_std']}}</span>
                    {% endif %}
                </li>
            </ul>
            {% if game_info['betting_lines'] and game_info['betting_lines']|length > 0 %}
                <h4>Betting Odds</h4>
                <table class="odds-table">
                    <tr><th>Provider</th><th>Spread</th><th>Over/Under</th><th>Home ML</th><th>Away ML</th></tr>
                    {% for odds in game_info['betting_lines'] %}
                    <tr>
                        <td>{{ odds['provider'] }}</td>
                        <td>{{ odds['formattedSpread'] or odds['spread'] }}</td>
                        <td>{{ odds['overUnder'] }}</td>
                        <td>{{ odds['homeMoneyline'] }}</td>
                        <td>{{ odds['awayMoneyline'] }}</td>
                    </tr>
                    {% endfor %}
                </table>
            {% else %}
                <div class="no-odds">No betting odds available for this game.</div>
            {% endif %}
        </div>
        {% endif %}
    </div>
    ''', weeks=weeks, selected_week=selected_week, games=games, selected_game=selected_game, game_info=game_info, all_dates=all_dates, selected_date=selected_date, show_all=show_all)


# New route: Projected Conference Records for 2025
@app.route('/conference-records')
def conference_records():
    # Only use 2025 season games
    df = _try_load_predictions()
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return render_template_string('<div style="font-family:Segoe UI,Arial,sans-serif;padding:24px">Predictions CSV not found. See instructions on the home page.</div>')
    df = _ensure_conf_columns(df)
    conf_games = df[df['season'] == 2025].copy()
    # Use merged conference info, skip non-conference games
    conf_games = conf_games[(conf_games['home_conference'] == conf_games['away_conference'])]
    conf_col = 'home_conference'
    # Determine winner for each game
    def get_winner(row):
        try:
            home_pts = float(row.get('predicted_home_points', 0))
            away_pts = float(row.get('predicted_away_points', 0))
            if home_pts > away_pts:
                return row['home_team']
            elif away_pts > home_pts:
                return row['away_team']
            else:
                return 'TIE'
        except Exception:
            return None
    conf_games['winner'] = conf_games.apply(get_winner, axis=1)
    # Aggregate records
    records = {}
    for _, row in conf_games.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']
        conference = row[conf_col] if conf_col else 'Unknown'
        for team in [home, away]:
            if team not in records:
                records[team] = {'conference': conference, 'W': 0, 'L': 0, 'T': 0}
        if winner == 'TIE':
            records[home]['T'] += 1
            records[away]['T'] += 1
        elif winner == home:
            records[home]['W'] += 1
            records[away]['L'] += 1
        elif winner == away:
            records[away]['W'] += 1
            records[home]['L'] += 1
    # Convert to DataFrame for conference records
    conf_rec_df = pd.DataFrame([
        {'Team': team, 'Conference': rec['conference'], 'Conf_Wins': rec['W'], 'Conf_Losses': rec['L'], 'Conf_Ties': rec['T']}
        for team, rec in records.items()
    ])
    conf_rec_df = conf_rec_df.sort_values(['Conference', 'Conf_Wins', 'Conf_Losses'], ascending=[True, False, True])

    # Calculate overall records for all teams (all games, not just conference)
    overall_records = {}
    all_games = pred_df[pred_df['season'] == 2025]
    for _, row in all_games.iterrows():
        home = row['home_team']
        away = row['away_team']
        try:
            home_pts = float(row.get('predicted_home_points', 0))
            away_pts = float(row.get('predicted_away_points', 0))
        except Exception:
            home_pts = away_pts = 0
        for team in [home, away]:
            if team not in overall_records:
                overall_records[team] = {'W': 0, 'L': 0, 'T': 0}
        if home_pts == away_pts:
            overall_records[home]['T'] += 1
            overall_records[away]['T'] += 1
        elif home_pts > away_pts:
            overall_records[home]['W'] += 1
            overall_records[away]['L'] += 1
        elif away_pts > home_pts:
            overall_records[away]['W'] += 1
            overall_records[home]['L'] += 1

    # Merge overall records into conference records
    conf_rec_df['Overall_Wins'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('W', 0))
    conf_rec_df['Overall_Losses'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('L', 0))
    conf_rec_df['Overall_Ties'] = conf_rec_df['Team'].map(lambda t: overall_records.get(t, {}).get('T', 0))

    # Load conference logos (prefer root data/; fallback to src/data/)
    conf_logo_df = None
    for _p in [
        "data/conference_logos.csv",
        "src/data/conference_logos.csv",
    ]:
        try:
            conf_logo_df = pd.read_csv(_p)
            break
        except Exception:
            pass
    conf_logo_map = {}
    if conf_logo_df is not None and {'conference','logo_url'}.issubset(conf_logo_df.columns):
        conf_logo_map = dict(zip(conf_logo_df['conference'], conf_logo_df['logo_url']))
    # Group by conference for cards
    conferences = conf_rec_df['Conference'].unique()
    conf_groups = {conf: conf_rec_df[conf_rec_df['Conference'] == conf] for conf in conferences}
    conf_logos = {conf: conf_logo_map.get(conf, '') for conf in conferences}
    # Load team assets for colors (prefer root data/; fallback to src/data/)
    _assets_df = None
    for _p in [
        "data/team_assets.csv",
        "src/data/team_assets.csv",
    ]:
        try:
            _assets_df = pd.read_csv(_p)
            break
        except Exception:
            pass
    team_color_map = {}
    team_alt_color_map = {}
    if _assets_df is not None and 'school' in _assets_df.columns:
        team_color_map = dict(zip(_assets_df['school'], _assets_df.get('color', '').fillna('')))
        if 'alt_color' in _assets_df.columns:
            team_alt_color_map = dict(zip(_assets_df['school'], _assets_df['alt_color'].fillna('')))
        else:
            team_alt_color_map = {s: '' for s in _assets_df['school']}
    # Render as cards per conference
    return render_template_string('''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        .conf-card { background: #f8f8f8; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.07); padding: 24px; margin-bottom: 28px; }
        .conf-title { font-size: 1.3em; font-weight: bold; color: #2980b9; margin-bottom: 12px; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; background: #fff; }
        th, td { padding: 8px 10px; border: 1px solid #e0e0e0; text-align: center; }
        th { background: #eaf1fb; color: #2c3e50; }
        tr:nth-child(even) { background: #f4f6fa; }
    </style>
    <div class="container">
        <div style="text-align:right; margin: 12px 0 0 0;">
            <a href="/" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">&#8592; Back to Main Predictions</a>
        </div>
        <h2>2025 Projected Conference Records</h2>
        {% for conf, group in conf_groups.items() %}
        <div class="conf-card">
            <div class="conf-title">
                {% if conf_logos[conf] %}
                    <img src="{{ conf_logos[conf] }}" alt="{{ conf }} logo" style="height:38px;vertical-align:middle;margin-right:10px;">
                {% endif %}
                {{ conf }}
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Team</th>
                        <th>Conf W</th>
                        <th>Conf L</th>
                        <th>Conf T</th>
                        <th>Overall W</th>
                        <th>Overall L</th>
                        <th>Overall T</th>
                    </tr>
                </thead>
                <tbody>
                    {% for _, row in group.iterrows() %}
                    <tr>
                        <td>
                            <span style="color:{{ team_color_map.get(row['Team'], '#2c3e50') }};background:{{ team_alt_color_map.get(row['Team'], '#eaf1fb') }};padding:3px 10px;border-radius:6px;display:inline-block;">{{ row['Team'] }}</span>
                        </td>
                        <td>{{ row['Conf_Wins'] }}</td>
                        <td>{{ row['Conf_Losses'] }}</td>
                        <td>{{ row['Conf_Ties'] }}</td>
                        <td>{{ row['Overall_Wins'] }}</td>
                        <td>{{ row['Overall_Losses'] }}</td>
                        <td>{{ row['Overall_Ties'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endfor %}
    </div>
    ''', conf_groups=conf_groups, conf_logos=conf_logos, team_color_map=team_color_map, team_alt_color_map=team_alt_color_map)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5051))
    app.run(host='0.0.0.0', port=port, debug=False)
