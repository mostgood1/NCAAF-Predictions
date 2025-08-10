


from flask import Flask, render_template_string, request
import pandas as pd
import requests

app = Flask(__name__)

# ...existing code...


from flask import Flask, render_template_string, request
import pandas as pd

app = Flask(__name__)




# Load predictions
pred_df = pd.read_csv("src/data/college_football_schedule_2025_predicted_totals.csv")

# Load team conferences
team_conf_df = pd.read_csv("src/data/team_conferences.csv")
team_conf_df['school_norm'] = team_conf_df['school'].str.strip().str.lower().str.replace('&', 'and').str.replace('  ', ' ')

# Add conference info to predictions
def norm(name):
    return str(name).strip().lower().replace('&', 'and').replace('  ', ' ')
conf_map = dict(zip(team_conf_df['school_norm'], team_conf_df['conference']))
pred_df['home_conference'] = pred_df['home_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))
pred_df['away_conference'] = pred_df['away_team'].apply(lambda x: conf_map.get(norm(x), 'Unknown'))

# Load win margin confidence intervals
try:
    win_margin_conf_df = pd.read_csv("src/data/win_margin_predictions_with_confidence.csv")
    win_margin_conf_df.columns = win_margin_conf_df.columns.str.strip()
except Exception:
    win_margin_conf_df = None

# Load team assets
assets_df = pd.read_csv("src/data/team_assets.csv")
def get_team_asset(team_name):
    row = assets_df[assets_df['school'] == team_name]
    if not row.empty:
        return {
            'logo': row.iloc[0].get('logo', ''),
            'color': row.iloc[0].get('color', ''),
            'alt_color': row.iloc[0].get('alt_color', '')
        }
    return {'logo': '', 'color': '', 'alt_color': ''}

# Load betting lines
lines_df = pd.read_csv("src/data/college_football_betting_lines_last_15_years.csv")
import ast
def get_betting_lines(year, week, home_team, away_team):
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
    weeks = sorted(pred_df['week'].unique())
    selected_week = request.form.get('week', weeks[0])
    # Get all unique dates for the selected week (date only, no time)
    week_games = pred_df[pred_df['week'] == int(selected_week)].copy()
    week_games['date_only'] = week_games['start_date'].str[:10]
    all_dates = sorted(week_games['date_only'].dropna().unique())
    selected_date = request.form.get('date', '')
    selected_conference = request.form.get('conference', '')
    show_all = request.form.get('show_all', '') == 'on'
    
    # Get all conferences for the dropdown
    all_conferences = sorted(set(list(week_games['home_conference'].dropna()) + list(week_games['away_conference'].dropna())))
    all_conferences = [conf for conf in all_conferences if conf != 'Unknown']
    
    # Filter games by date if not showing all
    if selected_date and not show_all:
        filtered_games = week_games[week_games['date_only'] == selected_date]
    else:
        filtered_games = week_games
    
    # Filter games by conference if selected
    if selected_conference:
        filtered_games = filtered_games[
            (filtered_games['home_conference'] == selected_conference) | 
            (filtered_games['away_conference'] == selected_conference)
        ]
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
        import pytz
        from datetime import datetime
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
        <a href="/team-schedules" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">View Team Schedules</a>
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
            <label for="conference">Filter by Conference:</label>
            <select name="conference" id="conference" onchange="document.getElementById('mainForm').submit();">
                <option value="">All Conferences</option>
                {% for conf in all_conferences %}
                <option value="{{conf}}" {% if conf == selected_conference %}selected{% endif %}>{{conf}}</option>
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
    ''', weeks=weeks, selected_week=selected_week, games=games, selected_game=selected_game, game_info=game_info, all_dates=all_dates, selected_date=selected_date, show_all=show_all, all_conferences=all_conferences, selected_conference=selected_conference)


# New route: Projected Conference Records for 2025
@app.route('/conference-records')
def conference_records():
    # Only use 2025 season games
    conf_games = pred_df[pred_df['season'] == 2025].copy()
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

    # Load conference logos
    conf_logo_df = pd.read_csv("src/data/conference_logos.csv")
    conf_logo_map = dict(zip(conf_logo_df['conference'], conf_logo_df['logo_url']))
    # Group by conference for cards
    conferences = conf_rec_df['Conference'].unique()
    conf_groups = {conf: conf_rec_df[conf_rec_df['Conference'] == conf] for conf in conferences}
    conf_logos = {conf: conf_logo_map.get(conf, '') for conf in conferences}
    # Load team assets for colors
    assets_df = pd.read_csv("src/data/team_assets.csv")
    team_color_map = dict(zip(assets_df['school'], assets_df['color']))
    team_alt_color_map = dict(zip(assets_df['school'], assets_df.get('alt_color', ['']*len(assets_df))))
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
                <tr>
                    <th>Team</th>
                    <th>Conf W</th><th>Conf L</th><th>Conf T</th>
                    <th>Overall W</th><th>Overall L</th><th>Overall T</th>
                </tr>
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
            </table>
        </div>
        {% endfor %}
    </div>
    ''', conf_groups=conf_groups, conf_logos=conf_logos, team_color_map=team_color_map, team_alt_color_map=team_alt_color_map)

# New route: Team Schedules
@app.route('/team-schedules', methods=['GET', 'POST'])
def team_schedules():
    # Get all conferences and teams
    all_teams_df = pred_df[pred_df['season'] == 2025].copy()
    all_conferences = sorted(set(list(all_teams_df['home_conference'].dropna()) + list(all_teams_df['away_conference'].dropna())))
    all_conferences = [conf for conf in all_conferences if conf != 'Unknown']
    
    selected_conference = request.form.get('conference', '')
    selected_team = request.form.get('team', '')
    
    # Get teams for selected conference
    if selected_conference:
        conf_teams = set()
        conf_games = all_teams_df[
            (all_teams_df['home_conference'] == selected_conference) | 
            (all_teams_df['away_conference'] == selected_conference)
        ]
        for _, row in conf_games.iterrows():
            if row['home_conference'] == selected_conference:
                conf_teams.add(row['home_team'])
            if row['away_conference'] == selected_conference:
                conf_teams.add(row['away_team'])
        available_teams = sorted(list(conf_teams))
    else:
        available_teams = []
    
    # Get team schedule if team is selected
    team_schedule = None
    team_info = None
    if selected_team:
        # Get all games for the selected team
        team_games = all_teams_df[
            (all_teams_df['home_team'] == selected_team) | 
            (all_teams_df['away_team'] == selected_team)
        ].copy()
        
        # Sort by week
        team_games = team_games.sort_values('week')
        
        # Process each game
        schedule_data = []
        team_record = {'W': 0, 'L': 0, 'T': 0}
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == selected_team
            opponent = game['away_team'] if is_home else game['home_team']
            
            # Get team assets for opponent
            opp_asset = get_team_asset(opponent)
            
            # Determine result
            try:
                home_pts = float(game.get('predicted_home_points', 0))
                away_pts = float(game.get('predicted_away_points', 0))
                team_pts = home_pts if is_home else away_pts
                opp_pts = away_pts if is_home else home_pts
                
                if team_pts > opp_pts:
                    result = 'W'
                    team_record['W'] += 1
                elif opp_pts > team_pts:
                    result = 'L'
                    team_record['L'] += 1
                else:
                    result = 'T'
                    team_record['T'] += 1
            except:
                result = '-'
                team_pts = opp_pts = 0
            
            # Parse date
            game_date = game.get('start_date', '')[:10] if game.get('start_date') else ''
            
            schedule_data.append({
                'week': int(game['week']),
                'date': game_date,
                'opponent': opponent,
                'is_home': is_home,
                'venue': game.get('venue', ''),
                'team_pts': f"{team_pts:.1f}" if team_pts else '',
                'opp_pts': f"{opp_pts:.1f}" if opp_pts else '',
                'result': result,
                'opp_logo': opp_asset['logo'],
                'opp_color': opp_asset['color'],
                'opp_alt_color': opp_asset['alt_color']
            })
        
        team_schedule = schedule_data
        
        # Get team asset info
        team_asset = get_team_asset(selected_team)
        team_info = {
            'name': selected_team,
            'logo': team_asset['logo'],
            'color': team_asset['color'],
            'alt_color': team_asset['alt_color'],
            'record': team_record
        }
    
    return render_template_string('''
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
        .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 24px; }
        form { display: flex; flex-direction: column; gap: 16px; margin-bottom: 32px; }
        label { font-weight: 500; color: #34495e; }
        select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 1em; }
        button { background: #2980b9; color: #fff; border: none; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #3498db; }
        .team-header { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 24px; padding: 20px; background: #f8f8f8; border-radius: 10px; }
        .team-logo { height: 80px; }
        .team-name { font-size: 1.8em; font-weight: bold; padding: 8px 16px; border-radius: 8px; display: inline-block; }
        .record { font-size: 1.2em; color: #2c3e50; margin-top: 8px; }
        .schedule-table { width: 100%; border-collapse: collapse; margin-top: 16px; background: #fff; }
        .schedule-table th, .schedule-table td { padding: 12px 8px; border: 1px solid #e0e0e0; text-align: center; }
        .schedule-table th { background: #eaf1fb; color: #2c3e50; }
        .schedule-table tr:nth-child(even) { background: #f4f6fa; }
        .opponent { display: flex; align-items: center; justify-content: center; gap: 8px; }
        .opp-logo { height: 30px; }
        .opp-name { padding: 2px 8px; border-radius: 4px; font-weight: 500; }
        .result-w { color: #27ae60; font-weight: bold; }
        .result-l { color: #e74c3c; font-weight: bold; }
        .result-t { color: #f39c12; font-weight: bold; }
        .home-indicator { color: #2980b9; font-weight: bold; }
        .away-indicator { color: #7f8c8d; }
    </style>
    <div class="container">
        <div style="text-align:right; margin: 12px 0 0 0;">
            <a href="/" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">&#8592; Back to Main Predictions</a>
            <a href="/conference-records" style="font-size:1.05em; color:#2980b9; text-decoration:underline; margin-right:18px;">View Conference Records</a>
        </div>
        <h2>2025 Team Schedules</h2>
        <form method="post" id="scheduleForm">
            <label for="conference">Select Conference:</label>
            <select name="conference" id="conference" onchange="document.getElementById('scheduleForm').submit();">
                <option value="">Choose a Conference</option>
                {% for conf in all_conferences %}
                <option value="{{conf}}" {% if conf == selected_conference %}selected{% endif %}>{{conf}}</option>
                {% endfor %}
            </select>
            {% if available_teams %}
            <label for="team">Select Team:</label>
            <select name="team" id="team" onchange="document.getElementById('scheduleForm').submit();">
                <option value="">Choose a Team</option>
                {% for team in available_teams %}
                <option value="{{team}}" {% if team == selected_team %}selected{% endif %}>{{team}}</option>
                {% endfor %}
            </select>
            {% endif %}
        </form>
        
        {% if team_info and team_schedule %}
        <div class="team-header">
            <img src="{{team_info['logo']}}" alt="{{team_info['name']}} logo" class="team-logo">
            <div>
                <div class="team-name" style="color:{{team_info['color']}};background:{{team_info['alt_color']}};">{{team_info['name']}}</div>
                <div class="record">Projected Record: {{team_info['record']['W']}}-{{team_info['record']['L']}}{% if team_info['record']['T'] > 0 %}-{{team_info['record']['T']}}{% endif %}</div>
            </div>
        </div>
        
        <table class="schedule-table">
            <tr>
                <th>Week</th>
                <th>Date</th>
                <th>Opponent</th>
                <th>Location</th>
                <th>Venue</th>
                <th>Projected Score</th>
                <th>Result</th>
            </tr>
            {% for game in team_schedule %}
            <tr>
                <td>{{game['week']}}</td>
                <td>{{game['date']}}</td>
                <td>
                    <div class="opponent">
                        <img src="{{game['opp_logo']}}" alt="{{game['opponent']}} logo" class="opp-logo">
                        <span class="opp-name" style="color:{{game['opp_color']}};background:{{game['opp_alt_color']}};">{{game['opponent']}}</span>
                    </div>
                </td>
                <td>
                    {% if game['is_home'] %}
                        <span class="home-indicator">HOME</span>
                    {% else %}
                        <span class="away-indicator">@ AWAY</span>
                    {% endif %}
                </td>
                <td>{{game['venue']}}</td>
                <td>
                    {% if game['team_pts'] and game['opp_pts'] %}
                        {{game['team_pts']}} - {{game['opp_pts']}}
                    {% else %}
                        -
                    {% endif %}
                </td>
                <td>
                    {% if game['result'] == 'W' %}
                        <span class="result-w">W</span>
                    {% elif game['result'] == 'L' %}
                        <span class="result-l">L</span>
                    {% elif game['result'] == 'T' %}
                        <span class="result-t">T</span>
                    {% else %}
                        -
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    ''', all_conferences=all_conferences, selected_conference=selected_conference, 
         available_teams=available_teams, selected_team=selected_team, 
         team_info=team_info, team_schedule=team_schedule)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5051))
    app.run(host='0.0.0.0', port=port, debug=False)
