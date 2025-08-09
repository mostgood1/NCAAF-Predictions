import pandas as pd
import ast

# Load features and betting lines
features_df = pd.read_csv("src/data/ncaa_games_last_15_years_features.csv")
lines_df = pd.read_csv("src/data/college_football_betting_lines_last_15_years.csv")

# Parse betting line features (spread, over/under) from the 'lines' column
spread_list = []
over_under_list = []
for idx, row in features_df.iterrows():
    # Find matching line by year, week, home, away (use correct column names)
    match = lines_df[(lines_df['year'] == row['season']) &
                     (lines_df['week'] == row['week']) &
                     (lines_df['homeTeam'] == row['home_team']) &
                     (lines_df['awayTeam'] == row['away_team'])]
    spread = None
    over_under = None
    if not match.empty:
        # Parse the odds JSON string
        try:
            odds = ast.literal_eval(match.iloc[0]['lines'])
            # Use the first provider's spread and over/under if available
            if isinstance(odds, list) and len(odds) > 0:
                spread = odds[0].get('spread')
                over_under = odds[0].get('overUnder')
        except Exception:
            pass
    spread_list.append(spread)
    over_under_list.append(over_under)


# Add new columns to features_df
features_df['betting_spread'] = spread_list
features_df['betting_over_under'] = over_under_list

# Add categorical encoding for home/away conference and venue
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

# Venue encoding (if available)
if 'venue' in features_df.columns:
    venue_map = {v: i for i, v in enumerate(features_df['venue'].dropna().unique())}
    features_df['venue_encoded'] = features_df['venue'].map(lambda v: venue_map.get(v, -1))

# Save updated features file
features_df.to_csv("src/data/ncaa_games_last_15_years_features_with_lines.csv", index=False)
print("Saved features with betting lines to src/data/ncaa_games_last_15_years_features_with_lines.csv")
