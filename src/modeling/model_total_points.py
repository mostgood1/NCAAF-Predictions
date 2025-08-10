import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load feature data with ELO ratings
df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")

features = [
    "home_total_points", "away_total_points", "margin_of_victory", "week", "season",
    "betting_spread", "betting_over_under",
    "home_conf_encoded", "away_conf_encoded", "venue_encoded",
    "home_elo", "away_elo",
    "home_season_avg_home", "away_season_avg_away",
    "home_recent_opp_avg_elo", "home_recent_opp_win_rate",
    "away_recent_opp_avg_elo", "away_recent_opp_win_rate"
]
X = df[features]
y = df["game_total_points"]


# Temporal validation: train on seasons before 2024, test on 2024
train_mask = df['season'] < 2024
test_mask = df['season'] == 2024

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Add recency weighting - recent games matter more for prediction accuracy
recency_weights = np.exp((df['season'] - 2010) / 4)  # Exponential weighting
train_weights = recency_weights[train_mask]

# Model - ENHANCED VERSION with better hyperparameters
model = RandomForestRegressor(
    n_estimators=400,        # Increased from 100 
    max_depth=22,            # Added depth control
    min_samples_split=5,     # Better overfitting control
    min_samples_leaf=3,      # Better leaf control
    max_features='sqrt',     # Feature randomness for better generalization
    random_state=42,
    n_jobs=-1               # Use all CPU cores for faster training
)
model.fit(X_train, y_train, sample_weight=train_weights)  # Use weighted training

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Feature importances
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.3f}")
