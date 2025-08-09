import pandas as pd

# Load cleaned data
csv_path = "src/data/ncaa_games_last_15_years_clean.csv"
df = pd.read_csv(csv_path)

# Example feature engineering
# 1. Total points per game
# 2. Margin of victory
# 3. Home win indicator
# 4. Points by half

def get_half_points(line_scores):
    # Assumes line_scores is a list of ints (points per quarter)
    if isinstance(line_scores, str):
        # Convert string representation to list
        import ast
        line_scores = ast.literal_eval(line_scores)
    if not isinstance(line_scores, list):
        return None, None
    first_half = sum(line_scores[:2])
    second_half = sum(line_scores[2:4])
    return first_half, second_half

# Total points
df["home_total_points"] = df["home_line_scores"].apply(lambda x: sum(eval(x)) if isinstance(x, str) else sum(x))
df["away_total_points"] = df["away_line_scores"].apply(lambda x: sum(eval(x)) if isinstance(x, str) else sum(x))
df["game_total_points"] = df["home_total_points"] + df["away_total_points"]

# Margin of victory
# Positive: home team won, Negative: away team won
# Zero: tie
# (Ties are rare in NCAA)
df["margin_of_victory"] = df["home_points"] - df["away_points"]

# Home win indicator
# 1 if home team won, 0 otherwise
df["home_win"] = (df["margin_of_victory"] > 0).astype(int)

# Points by half
home_halves = df["home_line_scores"].apply(get_half_points)
df["home_first_half"] = home_halves.apply(lambda x: x[0])
df["home_second_half"] = home_halves.apply(lambda x: x[1])

away_halves = df["away_line_scores"].apply(get_half_points)
df["away_first_half"] = away_halves.apply(lambda x: x[0])
df["away_second_half"] = away_halves.apply(lambda x: x[1])

# Save engineered features
df.to_csv("src/data/ncaa_games_last_15_years_features.csv", index=False)
print("Feature engineered data saved to src/data/ncaa_games_last_15_years_features.csv")
