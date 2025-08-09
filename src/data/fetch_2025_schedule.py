import cfbd
import pandas as pd

api_key = "2RQ9wRD6/0dFtORvjpF8sxYKzrGtGOh8deL4cm2XzH8ssIWBZ44HZQdyDQA1hFK1"
configuration = cfbd.Configuration(access_token=api_key)

with cfbd.ApiClient(configuration) as api_client:
    api_instance = cfbd.GamesApi(api_client)
    print("Fetching 2025 schedule...")
    games = api_instance.get_games(year=2025, season_type="regular")
    schedule = []
    for game in games:
        row = {
            "season": getattr(game, "season", None),
            "week": getattr(game, "week", None),
            "start_date": getattr(game, "start_date", None),
            "home_team": getattr(game, "home_team", None),
            "away_team": getattr(game, "away_team", None),
            "venue": getattr(game, "venue", None),
            "conference_game": getattr(game, "conference_game", None),
        }
        schedule.append(row)
    df = pd.DataFrame(schedule)
    df.to_csv("src/data/college_football_schedule_2025.csv", index=False)
    print("Saved 2025 schedule to src/data/college_football_schedule_2025.csv")
