

import requests
import pandas as pd
import os
import sys

# Load API key from config/settings.py if available, else from environment

import importlib.util
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

url = "https://api.collegefootballdata.com/teams"
headers = {"Authorization": f"Bearer {api_key}"}


response = requests.get(url, headers=headers)
print(f"Status code: {response.status_code}")
try:
    data = response.json()
    print(f"Full API data: {data}")
except Exception as e:
    print(f"Error parsing JSON: {e}")
    data = None

if isinstance(data, dict) and "teams" in data:
    teams = data["teams"]
elif isinstance(data, list):
    teams = data
else:
    teams = []

print(f"Number of teams found: {len(teams)}")

assets = []
for team in teams:
    if not isinstance(team, dict):
        continue
    assets.append({
        "school": team.get("school", ""),
        "mascot": team.get("mascot", ""),
        "abbreviation": team.get("abbreviation", ""),
        "conference": team.get("conference", ""),
        "logo": team.get("logos", [""])[0] if team.get("logos") else "",
        "color": team.get("color", ""),
        "alt_color": team.get("alt_color", "")
    })

print(f"Number of assets to save: {len(assets)}")
if assets:
    assets_df = pd.DataFrame(assets)
    assets_df.to_csv("src/data/team_assets.csv", index=False)
    print("Saved team assets to src/data/team_assets.csv")
else:
    print("No assets to save.")
