
import pandas as pd
from cfbd.rest import ApiException
from cfbd import Configuration, ApiClient
from cfbd import GamesApi

class DataIngestion:
    def fetch_and_save_last_15_years(self, api_key, out_csv):
        """Fetches all NCAA games with scoring by quarter for the last 15 years and saves to CSV."""
        configuration = Configuration(access_token=api_key)
        with ApiClient(configuration) as api_client:
            api_instance = GamesApi(api_client)
        all_games = []
        for year in range(pd.Timestamp.now().year - 15, pd.Timestamp.now().year):
            print(f"Fetching games for {year}...")
            try:
                games = api_instance.get_games(year=year, season_type="both")
            except ApiException as e:
                print(f"Exception when calling GamesApi->get_games for {year}: {e}")
                continue
            year_count = 0
            for game in games:
                if all([
                    getattr(game, "home_team", None),
                    getattr(game, "away_team", None),
                    getattr(game, "home_points", None) is not None,
                    getattr(game, "away_points", None) is not None,
                    getattr(game, "home_line_scores", None),
                    getattr(game, "away_line_scores", None)
                ]):
                    row = {
                        "season": getattr(game, "season", None),
                        "week": getattr(game, "week", None),
                        "home_team": getattr(game, "home_team", None),
                        "away_team": getattr(game, "away_team", None),
                        "home_points": getattr(game, "home_points", None),
                        "away_points": getattr(game, "away_points", None),
                        "home_line_scores": getattr(game, "home_line_scores", None),
                        "away_line_scores": getattr(game, "away_line_scores", None),
                    }
                    all_games.append(row)
                    year_count += 1
            print(f"Year {year}: {year_count} games added.")
        df = pd.DataFrame(all_games)
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} games to {out_csv}")
    """Handles data loading from CSV, web scraping, or API."""
    def from_csv(self, filepath):
        return pd.read_csv(filepath)

    # Removed from_api and from_web since cfbd-python is used for API calls

    # Removed legacy get_games_by_quarter using requests
    
    def save_games_to_csv(self, df, filename):
        """Saves the DataFrame of games to a CSV file."""
        df.to_csv(filename, index=False)
