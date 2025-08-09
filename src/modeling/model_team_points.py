import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load feature data with ELO ratings
df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")

home_features = [
    "away_total_points", "margin_of_victory", "week", "season", "game_total_points",
    "betting_spread", "betting_over_under",
    "home_conf_encoded", "away_conf_encoded", "venue_encoded",
    "home_elo", "away_elo",
    "home_season_avg_home", "away_season_avg_away",
    "home_recent_opp_avg_elo", "home_recent_opp_win_rate",
    "away_recent_opp_avg_elo", "away_recent_opp_win_rate"
]
X_home = df[home_features]
y_home = df["home_total_points"]

away_features = [
    "home_total_points", "margin_of_victory", "week", "season", "game_total_points",
    "betting_spread", "betting_over_under",
    "home_conf_encoded", "away_conf_encoded", "venue_encoded",
    "home_elo", "away_elo",
    "home_season_avg_home", "away_season_avg_away",
    "home_recent_opp_avg_elo", "home_recent_opp_win_rate",
    "away_recent_opp_avg_elo", "away_recent_opp_win_rate"
]
X_away = df[away_features]
y_away = df["away_total_points"]


# Temporal validation: train on seasons before 2024, test on 2024
train_mask = df['season'] < 2024
test_mask = df['season'] == 2024

Xh_train = X_home[train_mask]
yh_train = y_home[train_mask]
Xh_test = X_home[test_mask]
yh_test = y_home[test_mask]

Xa_train = X_away[train_mask]
ya_train = y_away[train_mask]
Xa_test = X_away[test_mask]
ya_test = y_away[test_mask]

# Home team model
home_model = RandomForestRegressor(n_estimators=100, random_state=42)
home_model.fit(Xh_train, yh_train)
home_pred = home_model.predict(Xh_test)
home_mae = mean_absolute_error(yh_test, home_pred)
home_r2 = r2_score(yh_test, home_pred)
print(f"Home Team Points - MAE: {home_mae:.2f}, R^2: {home_r2:.2f}")

# Away team model
away_model = RandomForestRegressor(n_estimators=100, random_state=42)
away_model.fit(Xa_train, ya_train)
away_pred = away_model.predict(Xa_test)
away_mae = mean_absolute_error(ya_test, away_pred)
away_r2 = r2_score(ya_test, away_pred)
print(f"Away Team Points - MAE: {away_mae:.2f}, R^2: {away_r2:.2f}")
