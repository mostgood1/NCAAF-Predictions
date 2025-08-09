from ingestion import DataIngestion

if __name__ == "__main__":
    api_key = "2RQ9wRD6/0dFtORvjpF8sxYKzrGtGOh8deL4cm2XzH8ssIWBZ44HZQdyDQA1hFK1"
    out_csv = "src/data/ncaa_games_last_15_years.csv"
    ingestor = DataIngestion()
    ingestor.fetch_and_save_last_15_years(api_key, out_csv)
