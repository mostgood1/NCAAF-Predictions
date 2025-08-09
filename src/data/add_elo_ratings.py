import pandas as pd

# Load features
features_df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_lines.csv")

# Initialize ELO ratings for each team
teams = pd.concat([features_df['home_team'], features_df['away_team']]).unique()
elo_dict = {team: 1500 for team in teams}
elo_home = []
elo_away = []
K = 20  # ELO update factor

for idx, row in features_df.iterrows():
    home = row['home_team']
    away = row['away_team']
    home_elo = elo_dict[home]
    away_elo = elo_dict[away]
    elo_home.append(home_elo)
    elo_away.append(away_elo)
    # Calculate expected result
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))
    # Actual result
    home_score = row['home_total_points']
    away_score = row['away_total_points']
    if home_score > away_score:
        actual_home = 1
        actual_away = 0
    elif home_score < away_score:
        actual_home = 0
        actual_away = 1
    else:
        actual_home = 0.5
        actual_away = 0.5
    # Update ELOs
    elo_dict[home] += K * (actual_home - expected_home)
    elo_dict[away] += K * (actual_away - expected_away)


features_df['home_elo'] = elo_home
features_df['away_elo'] = elo_away



# Add recent form (last 3 games average points for home and away teams)
features_df['home_recent_form'] = features_df.groupby('home_team')['home_total_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
features_df['away_recent_form'] = features_df.groupby('away_team')['away_total_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# Add opponent strength: average ELO and win rate of recent 3 opponents
def get_recent_opponent_stats(df, team_col, opp_team_col, opp_elo_col, opp_win_col, window=3):
    avg_elo = []
    win_rate = []
    for idx, row in df.iterrows():
        team = row[team_col]
        week = row['week']
        season = row['season']
        # Find previous games for this team
        prev_games = df[(df[team_col] == team) & ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))]
        prev_games = prev_games.sort_values(['season', 'week'], ascending=[False, False]).head(window)
        # Get opponent ELO and win
        if not prev_games.empty:
            opp_elos = prev_games[opp_elo_col].tolist()
            opp_wins = prev_games[opp_win_col].tolist()
            avg_elo.append(sum(opp_elos)/len(opp_elos))
            win_rate.append(sum(opp_wins)/len(opp_wins))
        else:
            avg_elo.append(None)
            win_rate.append(None)
    return avg_elo, win_rate


# For home team: opponents are away teams, use home_win for opponent win indicator
home_opp_elo, home_opp_win_rate = get_recent_opponent_stats(
    features_df, 'home_team', 'away_team', 'away_elo', 'home_win', window=3)
features_df['home_recent_opp_avg_elo'] = home_opp_elo
features_df['home_recent_opp_win_rate'] = home_opp_win_rate

# For away team: opponents are home teams, use home_win for opponent win indicator
away_opp_elo, away_opp_win_rate = get_recent_opponent_stats(
    features_df, 'away_team', 'home_team', 'home_elo', 'home_win', window=3)
features_df['away_recent_opp_avg_elo'] = away_opp_elo
features_df['away_recent_opp_win_rate'] = away_opp_win_rate

# Add home/away splits (season average points at home/away)
home_avg = features_df.groupby(['home_team', 'season'])['home_total_points'].transform('mean')
away_avg = features_df.groupby(['away_team', 'season'])['away_total_points'].transform('mean')
features_df['home_season_avg_home'] = home_avg
features_df['away_season_avg_away'] = away_avg

# Add categorical encoding for home/away conference and venue
import pandas as pd
lines_df = pd.read_csv("src/data/college_football_betting_lines_last_15_years.csv")
if 'homeConference' in lines_df.columns and 'awayConference' in lines_df.columns:
    home_conf_map = {conf: i for i, conf in enumerate(lines_df['homeConference'].dropna().unique())}
    away_conf_map = {conf: i for i, conf in enumerate(lines_df['awayConference'].dropna().unique())}
    home_conf_encoded = []
    away_conf_encoded = []
    for idx, row in features_df.iterrows():
        match = lines_df[(lines_df['year'] == row['season']) &
                         (lines_df['week'] == row['week']) &
                         (lines_df['homeTeam'] == row['home_team']) &
                         (lines_df['awayTeam'] == row['away_team'])]
        if not match.empty:
            home_conf = match.iloc[0].get('homeConference', None)
            away_conf = match.iloc[0].get('awayConference', None)
            home_conf_encoded.append(home_conf_map.get(home_conf, -1))
            away_conf_encoded.append(away_conf_map.get(away_conf, -1))
        else:
            home_conf_encoded.append(-1)
            away_conf_encoded.append(-1)
    features_df['home_conf_encoded'] = home_conf_encoded
    features_df['away_conf_encoded'] = away_conf_encoded


# Venue encoding (always add column, fill with -1 if not available)
if 'venue' in features_df.columns:
    venue_map = {v: i for i, v in enumerate(features_df['venue'].dropna().unique())}
    features_df['venue_encoded'] = features_df['venue'].map(lambda v: venue_map.get(v, -1))
else:
    features_df['venue_encoded'] = -1

# Placeholder columns for injuries and weather
features_df['home_injuries'] = 0
features_df['away_injuries'] = 0
features_df['weather'] = ''

features_df.to_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv", index=False)
print("Saved features with ELO ratings to src/data/ncaa_games_last_15_years_features_with_elo.csv")
