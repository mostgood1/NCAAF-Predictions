import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load feature data with ELO ratings
df = pd.read_csv("src/data/ncaa_games_last_15_years_features_with_elo.csv")

features = [
    "home_total_points", "away_total_points", "week", "season", "game_total_points",
    "betting_spread", "betting_over_under",
    "home_conf_encoded", "away_conf_encoded", "venue_encoded",
    "home_elo", "away_elo",
    "home_season_avg_home", "away_season_avg_away",
    "home_recent_opp_avg_elo", "home_recent_opp_win_rate",
    "away_recent_opp_avg_elo", "away_recent_opp_win_rate"
]
X = df[features]
y = df["margin_of_victory"]


# Temporal validation: train on seasons before 2024, test on 2024
train_mask = df['season'] < 2024
test_mask = df['season'] == 2024

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Add recency weighting for better predictions
recency_weights = np.exp((df['season'] - 2010) / 4)
train_weights = recency_weights[train_mask]

# Model - ENHANCED VERSION
model = RandomForestRegressor(
    n_estimators=400,        # Increased from 100
    max_depth=20,            # Added depth control for margin prediction
    min_samples_split=5,     # Better overfitting control
    min_samples_leaf=3,      # Better leaf control
    max_features='sqrt',     # Feature randomness
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)
model.fit(X_train, y_train, sample_weight=train_weights)  # Weighted training

# Predict

# Get predictions from each tree for confidence intervals
all_tree_preds = [tree.predict(X_test) for tree in model.estimators_]
import numpy as np
y_pred = np.mean(all_tree_preds, axis=0)
conf_std = np.std(all_tree_preds, axis=0)

# Evaluation

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Output predictions with confidence intervals

# Add home_team and away_team columns for robust matching
results_df = X_test.copy()
results_df['actual_margin'] = y_test.values
results_df['predicted_margin'] = y_pred
results_df['conf_std'] = conf_std
results_df['conf_interval_lower'] = y_pred - 1.96 * conf_std
results_df['conf_interval_upper'] = y_pred + 1.96 * conf_std
if 'home_team' in df.columns and 'away_team' in df.columns:
    results_df['home_team'] = df.loc[test_mask, 'home_team'].values
    results_df['away_team'] = df.loc[test_mask, 'away_team'].values
results_df.to_csv('src/data/win_margin_predictions_with_confidence.csv', index=False)
print(results_df[['predicted_margin', 'conf_std', 'conf_interval_lower', 'conf_interval_upper']].head())

# Feature importances
importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.3f}")
