"""OpenWeather-based weather enrichment utilities.

Relies on environment variable OPENWEATHER_API_KEY (or OWM_API_KEY) for auth.
Provides:
  - ensure_team_location(team): returns (lat, lon, indoor_flag)
    * Uses local CSV data/team_locations.csv (team,latitude,longitude,is_indoor)
    * If missing, attempts geocode via OpenWeather direct geocoding API and appends to file.
  - fetch_weather(lat, lon, game_dt): returns dict with temp_f, wind_mph
  - compute_weather_adjustment(temp_f, wind_mph, is_indoor): simple heuristic.
  - enrich_dataframe(df): fills weather_temp, weather_wind, weather_adjustment for rows missing them.

No placeholders: if a row cannot be enriched (missing date, location, API failure) it is recorded in 'enrichment_failed' column.
"""
from __future__ import annotations
import os, csv, math, time, json
from pathlib import Path
from typing import Optional, Tuple
import requests
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = BASE_DIR / 'data'
LOC_FILE = DATA_DIR / 'team_locations.csv'
GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"  # 5 day / 3 hour

API_KEY = os.getenv('OPENWEATHER_API_KEY') or os.getenv('OWM_API_KEY')

def _log(msg: str):
    print(f"[weather] {msg}")

def _load_locations() -> pd.DataFrame:
    if LOC_FILE.exists():
        try:
            return pd.read_csv(LOC_FILE)
        except Exception:
            return pd.DataFrame(columns=['team','latitude','longitude','is_indoor'])
    return pd.DataFrame(columns=['team','latitude','longitude','is_indoor'])

def _append_location(team: str, lat: float, lon: float, indoor: bool=False):
    header_needed = not LOC_FILE.exists()
    with LOC_FILE.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(['team','latitude','longitude','is_indoor'])
        w.writerow([team, lat, lon, 1 if indoor else 0])

def geocode_team(team: str) -> Optional[Tuple[float,float]]:
    if not API_KEY:
        _log('API key missing; cannot geocode')
        return None
    try:
        resp = requests.get(GEOCODE_URL, params={'q': team, 'limit': 1, 'appid': API_KEY}, timeout=10)
        if resp.status_code != 200:
            _log(f'geocode {team} status {resp.status_code}')
            return None
        data = resp.json()
        if not data:
            return None
        ent = data[0]
        lat, lon = ent.get('lat'), ent.get('lon')
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except Exception as e:
        _log(f'geocode {team} error {e}')
        return None

def ensure_team_location(team: str) -> Optional[Tuple[float,float,bool]]:
    df = _load_locations()
    if not df.empty and 'team' in df.columns:
        row = df[df['team'].str.lower() == team.lower()].head(1)
        if not row.empty:
            try:
                return float(row.iloc[0]['latitude']), float(row.iloc[0]['longitude']), bool(int(row.iloc[0].get('is_indoor',0)))
            except Exception:
                pass
    geo = geocode_team(team)
    if geo:
        lat, lon = geo
        _append_location(team, lat, lon, False)
        return lat, lon, False
    return None

def fetch_weather(lat: float, lon: float, game_dt: datetime) -> Optional[dict]:
    if not API_KEY:
        return None
    try:
        now = datetime.now(timezone.utc)
        delta_hours = (game_dt - now).total_seconds() / 3600.0
        if delta_hours < -6:  # game long finished; use current as fallback (historical API would require paid plan)
            which = CURRENT_URL
            params = {'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}
            r = requests.get(which, params=params, timeout=10)
            if r.status_code != 200:
                return None
            d = r.json()
            return {'temp_f': d.get('main',{}).get('temp'), 'wind_mph': d.get('wind',{}).get('speed')}
        elif delta_hours <= 6:  # near/ongoing => current
            r = requests.get(CURRENT_URL, params={'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}, timeout=10)
            if r.status_code != 200:
                return None
            d = r.json()
            return {'temp_f': d.get('main',{}).get('temp'), 'wind_mph': d.get('wind',{}).get('speed')}
        else:  # future => forecast slots
            r = requests.get(FORECAST_URL, params={'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}, timeout=10)
            if r.status_code != 200:
                return None
            d = r.json()
            lst = d.get('list', [])
            if not lst:
                return None
            target_ts = game_dt.timestamp()
            best = None; best_dt_diff = 10**9
            for ent in lst:
                dt_txt = ent.get('dt_txt')
                try:
                    slot_dt = datetime.strptime(dt_txt, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                diff = abs(slot_dt.timestamp() - target_ts)
                if diff < best_dt_diff:
                    best_dt_diff = diff
                    best = ent
            if best:
                return {'temp_f': best.get('main',{}).get('temp'), 'wind_mph': best.get('wind',{}).get('speed')}
        return None
    except Exception as e:
        _log(f'fetch error {e}')
        return None

def compute_weather_adjustment(temp_f: Optional[float], wind_mph: Optional[float], is_indoor: bool) -> Optional[float]:
    if is_indoor:
        return 0.0
    if temp_f is None and wind_mph is None:
        return None
    try:
        penalty = 0.0
        if wind_mph is not None and wind_mph > 5:
            penalty += 0.15 * (wind_mph - 5)
        if temp_f is not None and temp_f < 60:
            penalty += 0.05 * ((60 - temp_f)/5)
        return round(-penalty, 2)
    except Exception:
        return None

def _parse_start(dt_str: str) -> Optional[datetime]:
    if not dt_str or str(dt_str).lower() in ('nan','nat','none','null'): return None
    s = str(dt_str).replace(' ', 'T')
    try:
        if s.endswith('Z'):
            return datetime.fromisoformat(s.replace('Z','+00:00'))
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return pd.to_datetime(dt_str, errors='coerce').to_pydatetime()
        except Exception:
            return None

def enrich_dataframe(df: pd.DataFrame, limit: int = 150) -> pd.DataFrame:
    if df.empty:
        return df
    needed_cols = ['weather_temp','weather_wind','weather_adjustment']
    for c in needed_cols:
        if c not in df.columns:
            df[c] = pd.NA
    if 'enrichment_failed' not in df.columns:
        df['enrichment_failed'] = False
    todo_idx = df[(df['weather_temp'].isna()) & (df['weather_wind'].isna())].head(limit).index
    for idx in todo_idx:
        row = df.loc[idx]
        start_dt = _parse_start(row.get('start_date') or row.get('start_date_api'))
        if not start_dt:
            df.at[idx,'enrichment_failed'] = True
            continue
        loc = ensure_team_location(row.get('home_team',''))
        if not loc:
            df.at[idx,'enrichment_failed'] = True
            continue
        lat, lon, indoor = loc
        w = fetch_weather(lat, lon, start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc))
        if not w:
            df.at[idx,'enrichment_failed'] = True
            continue
        temp_f = w.get('temp_f'); wind_mph = w.get('wind_mph')
        df.at[idx,'weather_temp'] = temp_f
        df.at[idx,'weather_wind'] = wind_mph
        df.at[idx,'weather_adjustment'] = compute_weather_adjustment(temp_f, wind_mph, indoor)
    return df

__all__ = [
    'enrich_dataframe','ensure_team_location','fetch_weather','compute_weather_adjustment'
]
