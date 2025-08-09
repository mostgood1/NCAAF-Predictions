import pandas as pd

# Load the CSV
csv_path = "src/data/ncaa_games_last_15_years.csv"
df = pd.read_csv(csv_path)

# Basic exploration
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Remove rows with missing scores or teams
clean_df = df.dropna(subset=["home_team", "away_team", "home_points", "away_points", "home_line_scores", "away_line_scores"])
print(f"Rows after cleaning: {clean_df.shape[0]}")

# Save cleaned data
clean_df.to_csv("src/data/ncaa_games_last_15_years_clean.csv", index=False)
print("Cleaned data saved to src/data/ncaa_games_last_15_years_clean.csv")
