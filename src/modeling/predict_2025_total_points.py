
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load 2025 schedule
games_2025 = pd.read_csv("src/data/college_football_schedule_2025.csv")


# Load historical features for teams with ELO ratings
features_df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")

# For each team, calculate average total points from historical data
team_stats = features_df.groupby("home_team")["home_total_points"].mean().to_dict()
team_stats.update(features_df.groupby("away_team")["away_total_points"].mean().to_dict())

pred_rows = []
# Prepare features for prediction
def get_team_avg(team, stat_dict):
    return stat_dict.get(team, np.mean(list(stat_dict.values())))

pred_rows = []

# Merge betting lines for 2025 games if available, else fill with historical averages
betting_lines_2025 = None
try:
    betting_lines_2025 = pd.read_csv("src/data/college_football_betting_lines_2025.csv")
except Exception:
    betting_lines_2025 = pd.DataFrame()

avg_spread = features_df["betting_spread"].mean()
avg_ou = features_df["betting_over_under"].mean()

for _, row in games_2025.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    week = row["week"]
    season = row["season"]
    home_avg = get_team_avg(home, team_stats)
    away_avg = get_team_avg(away, team_stats)
    # Try to get betting lines for this game
    spread = avg_spread
    over_under = avg_ou
    if not betting_lines_2025.empty:
        match = betting_lines_2025[(betting_lines_2025["homeTeam"] == home) & (betting_lines_2025["awayTeam"] == away) & (betting_lines_2025["week"] == week) & (betting_lines_2025["season"] == season)]
        if not match.empty:
            spread = match.iloc[0].get("betting_spread", avg_spread)
            over_under = match.iloc[0].get("betting_over_under", avg_ou)

    # Get ELO ratings for home and away teams from historical averages
    home_elo = features_df[features_df['home_team'] == home]['home_elo'].mean()
    away_elo = features_df[features_df['away_team'] == away]['away_elo'].mean()
    home_split = features_df[features_df['home_team'] == home]['home_season_avg_home'].mean()
    away_split = features_df[features_df['away_team'] == away]['away_season_avg_away'].mean()
    # Get opponent strength features
    home_recent_opp_avg_elo = features_df[features_df['home_team'] == home]['home_recent_opp_avg_elo'].mean()
    home_recent_opp_win_rate = features_df[features_df['home_team'] == home]['home_recent_opp_win_rate'].mean()
    away_recent_opp_avg_elo = features_df[features_df['away_team'] == away]['away_recent_opp_avg_elo'].mean()
    away_recent_opp_win_rate = features_df[features_df['away_team'] == away]['away_recent_opp_win_rate'].mean()
    features = [home_avg, away_avg, week, season, spread, over_under, home_elo, away_elo, home_split, away_split,
                home_recent_opp_avg_elo, home_recent_opp_win_rate, away_recent_opp_avg_elo, away_recent_opp_win_rate]
    pred_rows.append(features)

X_pred = pd.DataFrame(pred_rows, columns=["home_total_points", "away_total_points", "week", "season", "betting_spread", "betting_over_under", "home_elo", "away_elo", "home_season_avg_home", "away_season_avg_away", "home_recent_opp_avg_elo", "home_recent_opp_win_rate", "away_recent_opp_avg_elo", "away_recent_opp_win_rate"])







# Temporal validation: train on seasons before 2024, test on 2024
train_mask = features_df['season'] < 2024
test_mask = features_df['season'] == 2024


# Train home points model
y_home = features_df["home_total_points"]
X_home = features_df[["away_total_points", "week", "season", "betting_spread", "betting_over_under", "home_elo", "away_elo", "home_season_avg_home", "away_season_avg_away", "home_recent_opp_avg_elo", "home_recent_opp_win_rate", "away_recent_opp_avg_elo", "away_recent_opp_win_rate"]]
model_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_home.fit(X_home[train_mask], y_home[train_mask])
X_pred_home = X_pred[["away_total_points", "week", "season", "betting_spread", "betting_over_under", "home_elo", "away_elo", "home_season_avg_home", "away_season_avg_away", "home_recent_opp_avg_elo", "home_recent_opp_win_rate", "away_recent_opp_avg_elo", "away_recent_opp_win_rate"]]
games_2025["predicted_home_points"] = model_home.predict(X_pred_home)

# Train away points model
y_away = features_df["away_total_points"]
X_away = features_df[["home_total_points", "week", "season", "betting_spread", "betting_over_under", "home_elo", "away_elo", "home_season_avg_home", "away_season_avg_away", "home_recent_opp_avg_elo", "home_recent_opp_win_rate", "away_recent_opp_avg_elo", "away_recent_opp_win_rate"]]
model_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_away.fit(X_away[train_mask], y_away[train_mask])
X_pred_away = X_pred[["home_total_points", "week", "season", "betting_spread", "betting_over_under", "home_elo", "away_elo", "home_season_avg_home", "away_season_avg_away", "home_recent_opp_avg_elo", "home_recent_opp_win_rate", "away_recent_opp_avg_elo", "away_recent_opp_win_rate"]]
games_2025["predicted_away_points"] = model_away.predict(X_pred_away)


# Enforce strict alignment
games_2025["predicted_total_points"] = games_2025["predicted_home_points"] + games_2025["predicted_away_points"]
games_2025["predicted_win_margin"] = games_2025["predicted_home_points"] - games_2025["predicted_away_points"]

# Save predictions
games_2025.to_csv("src/data/college_football_schedule_2025_predicted_totals.csv", index=False)
print("Saved 2025 schedule with predicted total, home, away points, and win margin to src/data/college_football_schedule_2025_predicted_totals.csv")
