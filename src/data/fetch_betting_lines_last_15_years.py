import requests
import pandas as pd
import os
import importlib.util

# Load API key from config/settings.py if available, else from environment
api_key = None
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/settings.py'))
if os.path.exists(config_path):
    spec = importlib.util.spec_from_file_location("settings", config_path)
    settings = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(settings)
        api_key = getattr(settings, 'CFBD_API_KEY', None)
    except Exception:
        pass
if not api_key:
    api_key = os.getenv("CFBD_API_KEY")
if not api_key:
    # Fallback: use hardcoded API key if provided here
    api_key = "2RQ9wRD6/0dFtORvjpF8sxYKzrGtGOh8deL4cm2XzH8ssIWBZ44HZQdyDQA1hFK1"
    if not api_key:
        print("Error: Please set CFBD_API_KEY in config/settings.py, as an environment variable, or provide it in the script.")
        exit(1)

headers = {"Authorization": f"Bearer {api_key}"}
all_lines = []
for year in range(2010, 2025+1):
    print(f"Fetching lines for year {year}...")
    url = f"https://api.collegefootballdata.com/lines?year={year}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        lines = response.json()
        for line in lines:
            line['year'] = year
        all_lines.extend(lines)
    else:
        print(f"Failed to fetch lines for {year}: {response.status_code}")

if all_lines:
    df = pd.DataFrame(all_lines)
    df.to_csv("src/data/college_football_betting_lines_last_15_years.csv", index=False)
    print("Saved betting lines to src/data/college_football_betting_lines_last_15_years.csv")
else:
    print("No betting lines found.")
