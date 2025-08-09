import pandas as pd
from ingestion import DataIngestion

def main():
    year = input("Enter the year to fetch NCAA games for: ")
    try:
        year = int(year)
    except ValueError:
        print("Invalid year. Please enter a valid integer.")
        return
    ingestor = DataIngestion()
    print(f"Fetching games for {year}...")
    df = ingestor.get_games_by_quarter(season=year)
    out_csv = f"ncaa_games_{year}.csv"
    ingestor.save_games_to_csv(df, f"src/data/{out_csv}")
    print(f"Saved {len(df)} games to src/data/{out_csv}")

if __name__ == "__main__":
    main()
