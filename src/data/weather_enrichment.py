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
from typing import Optional, Tuple, List
import requests
import pandas as pd
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = BASE_DIR / 'data'
LOC_FILE = DATA_DIR / 'team_locations.csv'
GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"  # 5 day / 3 hour

def _load_dotenv_for_api_key() -> str | None:
    """Attempt to load OPENWEATHER_API_KEY from a .env file if not already set.

    Looks for .env in (a) current working directory, (b) BASE_DIR, (c) parent of BASE_DIR.
    Only sets environment variable if absent. Returns the discovered key or None.
    """
    candidates = [
        Path.cwd() / '.env.local',
        Path.cwd() / '.env',
        BASE_DIR / '.env.local',
        BASE_DIR / '.env',
        (BASE_DIR.parent if BASE_DIR.parent else BASE_DIR) / '.env.local',
        (BASE_DIR.parent if BASE_DIR.parent else BASE_DIR) / '.env'
    ]
    for p in candidates:
        try:
            if p.exists():
                for line in p.read_text(encoding='utf-8').splitlines():
                    line=line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k,v = line.split('=',1)
                    k=k.strip(); v=v.strip().strip('"').strip("'")
                    if k == 'OPENWEATHER_API_KEY' and v and not os.getenv('OPENWEATHER_API_KEY'):
                        os.environ['OPENWEATHER_API_KEY'] = v
                        return v
                    if k == 'OWM_API_KEY' and v and not os.getenv('OWM_API_KEY'):
                        os.environ['OWM_API_KEY'] = v
                        if not os.getenv('OPENWEATHER_API_KEY'):
                            os.environ['OPENWEATHER_API_KEY'] = v
                        return v
        except Exception:
            continue
    return None

# Pull API key (with .env fallback)
API_KEY = os.getenv('OPENWEATHER_API_KEY') or os.getenv('OWM_API_KEY')
if not API_KEY:
    maybe = _load_dotenv_for_api_key()
    if maybe:
        API_KEY = maybe
        try:
            print('[weather] API key loaded from .env file')
        except Exception:
            pass

# Verbosity / progress controls
VERBOSE = os.getenv('WEATHER_VERBOSE','0') in ('1','true','TRUE','yes','on')
PROGRESS_EVERY = int(os.getenv('WEATHER_PROGRESS_EVERY','25'))  # rows
RATE_LIMIT_SLEEP = float(os.getenv('WEATHER_RATE_SLEEP','0.0'))  # optional throttle seconds per request
CAPTURE_DEBUG = os.getenv('WEATHER_DEBUG','0') in ('1','true','TRUE','yes','on')
REQUEST_TIMEOUT = float(os.getenv('WEATHER_REQ_TIMEOUT','3'))  # hard timeout per HTTP call (seconds)

# In-memory location cache to avoid re-reading CSV every row
_LOCATION_CACHE: dict[str, tuple[float,float,bool]] = {}
_LOC_FILE_MTIME: float | None = None

# Minimal static fallback coordinates (stadium approximations) to avoid geocode dependency when API blocked.
# Format: lowercase team name -> (lat, lon, is_indoor)
STATIC_FBS_COORDS = {
    # ACC / Power / Others
    'wake forest': (36.1353, -80.2520, False),
    'ucla': (34.1613, -118.1676, False),  # Rose Bowl
    'indiana': (39.1809, -86.5250, False),
    'syracuse': (43.0379, -76.1381, True),  # Dome
    'houston': (29.7219, -95.3493, False),
    'arizona': (32.2288, -110.9488, False),
    'nevada': (39.5369, -119.8183, False),
    'south carolina': (33.9737, -81.0190, False),
    'ole miss': (34.3630, -89.5383, False),
    'fresno state': (36.8123, -119.7491, False),
    "hawai'i": (21.3007, -157.8194, False),
    'temple': (39.9789, -75.1744, False),
    'nebraska': (40.8206, -96.7078, False),
    'northwestern': (42.0656, -87.6924, False),
    'maryland': (38.9897, -76.9456, False),
    'virginia': (38.0310, -78.5133, False),
    'ohio state': (40.0017, -83.0197, False),
    'texas a&m': (30.6103, -96.3400, False),
    'boston college': (42.3355, -71.1665, False),
    'alabama': (33.2076, -87.5505, False),
    'arizona state': (33.4269, -111.9325, False),
    'arkansas state': (35.8421, -90.6848, False),
    'auburn': (32.6029, -85.4893, False),
    'baylor': (31.5503, -97.1147, False),
    'california': (37.8704, -122.2500, False),
    'charlotte': (35.3079, -80.7320, False),
    'cincinnati': (39.1313, -84.5160, False),
    'coastal carolina': (33.7925, -79.0139, False),
    'delaware': (39.6654, -75.7531, False),
    'florida international': (25.7573, -80.3773, False),
    'georgia southern': (32.4121, -81.7832, False),
    'georgia state': (33.7398, -84.3897, False),
    'georgia tech': (33.7724, -84.3929, False),
    'illinois': (40.0994, -88.2350, False),
    'iowa': (41.6589, -91.5511, False),
    'kennesaw state': (34.0380, -84.5808, False),
    'kentucky': (38.0220, -84.5058, False),
    'louisiana tech': (32.5325, -92.6520, False),
    'lsu': (30.4120, -91.1839, False),
    'marshall': (38.3742, -82.4203, False),
    'miami': (25.9580, -80.2389, False),
    'michigan': (42.2659, -83.7487, False),
    'michigan state': (42.7289, -84.4849, False),
    'mississippi state': (33.4625, -88.7931, False),
    'missouri': (38.9354, -92.3338, False),
    'missouri state': (37.1983, -93.2785, False),
    'north carolina': (35.9070, -79.0478, False),
    'north texas': (33.2070, -97.1558, False),
    'penn state': (40.8122, -77.8560, False),
    'purdue': (40.4347, -86.9236, False),
    'rice': (29.7175, -95.4044, False),
    'rutgers': (40.5135, -74.4654, False),
    'southern miss': (31.3253, -89.2903, False),
    'stanford': (37.4349, -122.1610, False),
    'tcu': (32.7096, -97.3605, False),
    'tennessee': (35.9550, -83.9250, False),
    'texas': (30.2836, -97.7327, False),
    'texas tech': (33.5900, -101.8870, False),
    'troy': (31.8005, -85.9572, False),
    'tulane': (29.9488, -90.1218, False),
    'tulsa': (36.1428, -95.9473, False),
    'uab': (33.5113, -86.8033, False),
    'utah state': (41.7428, -111.8069, False),
    'utsa': (29.5836, -98.6200, False),
    'virginia tech': (37.2220, -80.4178, False),
    'west virginia': (39.6508, -79.9557, False),
    'wyoming': (41.3129, -105.5683, False),
    'kansas state': (39.1976, -96.5931, False),
    'oklahoma state': (36.1270, -97.0737, False),
    'uconn': (41.8089, -72.2495, False),
    'boston college': (42.3355, -71.1665, False),
    # Additional FBS teams (expanded list)
    'air force': (38.9972, -104.8439, False),
    'akron': (41.0739, -81.5129, False),
    'app state': (36.2136, -81.6854, False),
    'appalachian state': (36.2136, -81.6854, False),
    'army': (41.3790, -73.9635, False),
    'ball state': (40.2065, -85.4081, False),
    'boise state': (43.6015, -116.1973, False),
    'bowling green': (41.3780, -83.6266, False),
    'byu': (40.2576, -111.6546, False),
    'central michigan': (43.5810, -84.7722, False),
    'colorado': (40.0094, -105.2669, False),
    'colorado state': (40.5765, -105.0834, False),
    'connecticut': (41.8089, -72.2495, False),
    'duke': (35.9976, -78.9423, False),
    'east carolina': (35.5967, -77.3659, False),
    'eastern michigan': (42.3033, -83.7057, False),
    'florida': (29.6499, -82.3482, False),
    'florida atlantic': (26.3673, -80.0969, False),
    'florida state': (30.4380, -84.3049, False),
    'utsa': (29.5836, -98.6200, False),
    'ga southern': (32.4121, -81.7832, False),
    'ga state': (33.7398, -84.3897, False),
    'liberty': (37.3501, -79.1803, False),
    'louisville': (38.2120, -85.7601, False),
    'memphis': (35.1190, -89.9773, False),
    'miami (oh)': (39.5061, -84.7312, False),
    'middle tennessee': (35.8498, -86.3691, False),
    'navy': (38.9854, -76.5064, False),
    'new mexico': (35.0675, -106.6260, False),
    'new mexico state': (32.2862, -106.9123, False),
    'niu': (41.9345, -88.7772, False),
    'north alabama': (34.8108, -87.6773, False),
    'northern illinois': (41.9345, -88.7772, False),
    'notre dame': (41.6986, -86.2353, False),
    'ohio': (39.3227, -82.1018, False),
    'oklahoma': (35.2059, -97.4458, False),
    'oregon': (44.0582, -123.0687, False),
    'oregon state': (44.5594, -123.2814, False),
    'pitt': (40.4430, -79.9546, False),
    'san diego state': (32.7757, -117.0716, False),
    'san jose state': (37.3209, -121.8687, False),
    'san josé state': (37.3209, -121.8687, False),  # Unicode accented variant
    'smu': (32.8413, -96.7846, False),
    'south alabama': (30.6956, -88.1897, False),
    'texas state': (29.8900, -97.9380, False),
    'toledo': (41.6588, -83.6150, False),
    'uab': (33.5113, -86.8033, False),
    'ucf': (28.6079, -81.1970, False),
    'ul lafayette': (30.2119, -92.0238, False),
    'ul monroe': (32.5322, -92.0699, False),
    'umass': (42.3910, -72.5270, False),
    'unlv': (36.0908, -115.1830, False),
    'utep': (31.7723, -106.5079, False),
    'utah': (40.7608, -111.8910, False),
    'washington': (47.6500, -122.3016, False),
    'washington state': (46.7320, -117.1653, False),
    'western kentucky': (36.9846, -86.4551, False),
    'western michigan': (42.2927, -85.5889, False),
    'wku': (36.9846, -86.4551, False),
    'wyoming': (41.3129, -105.5683, False),
    # Additional bulk FBS coverage / aliases
    'arkansas': (35.8434, -94.0429, False),
    'georgia': (33.9498, -83.3737, False),
    'vanderbilt': (36.1392, -86.8039, False),
    'clemson': (34.6786, -82.8454, False),
    'nc state': (35.8001, -78.7190, False),
    'kansas': (38.9550, -95.2544, False),
    'iowa state': (42.0140, -93.6350, False),
    'minnesota': (44.9760, -93.2280, False),
    'wisconsin': (43.0699, -89.4125, False),
    'usc': (34.0141, -118.2879, False),
    'oregon state': (44.5594, -123.2814, False),
    'washington state': (46.7320, -117.1653, False),
    'james madison': (38.4351, -78.8738, False),
    'old dominion': (36.8856, -76.3057, False),
    'jacksonville state': (33.8246, -85.7630, False),
    'sam houston': (30.7163, -95.5500, False),
    'south florida': (27.9904, -82.4389, False),
    'kent state': (41.1399, -81.3131, False),
    'buffalo': (42.8599, -78.8100, False),
    'georgia southern': (32.4121, -81.7832, False),
    'georgia state': (33.7398, -84.3897, False),
    'louisiana': (30.2119, -92.0238, False),
    'fiu': (25.7573, -80.3773, False),
}

def _refresh_location_cache():
    global _LOCATION_CACHE, _LOC_FILE_MTIME
    try:
        if LOC_FILE.exists():
            mtime = LOC_FILE.stat().st_mtime
            if _LOC_FILE_MTIME is None or mtime != _LOC_FILE_MTIME:
                df = _load_locations()
                cache = {}
                for _, r in df.iterrows():
                    try:
                        cache[str(r['team']).lower()] = (float(r['latitude']), float(r['longitude']), bool(int(r.get('is_indoor',0))))
                    except Exception:
                        continue
                _LOCATION_CACHE = cache
                _LOC_FILE_MTIME = mtime
    except Exception:
        pass

_refresh_location_cache()

# Basic list of FBS conferences (2025) for prioritizing enrichment
FBS_CONFERENCES: List[str] = [
    'SEC','Big Ten','Big 12','ACC','Pac-12','AAC','Mountain West','Sun Belt','Conference USA','MAC',
    'Independents','American','CUSA','MWC'
]

def _load_conference_map() -> dict:
    tc_path = DATA_DIR / 'team_conferences.csv'
    if not tc_path.exists():
        return {}
    try:
        df = pd.read_csv(tc_path)
        if not {'school','conference'}.issubset(df.columns):
            return {}
        mp = {}
        for _, r in df.iterrows():
            try:
                mp[str(r['school']).strip().lower()] = str(r['conference']).strip()
            except Exception:
                continue
        return mp
    except Exception:
        return {}

_CONF_MAP = _load_conference_map()

def _is_fbs_team(team: str) -> bool:
    if not team:
        return False
    conf = _CONF_MAP.get(str(team).strip().lower())
    if not conf:
        return False
    return any(conf.startswith(fbs) for fbs in FBS_CONFERENCES)

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
        resp = requests.get(GEOCODE_URL, params={'q': team, 'limit': 1, 'appid': API_KEY}, timeout=REQUEST_TIMEOUT)
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
    if not team:
        return None
    key = team.lower()
    # Fast path cache
    if key in _LOCATION_CACHE:
        return _LOCATION_CACHE[key]
    # Refresh cache (maybe file updated by another process)
    _refresh_location_cache()
    if key in _LOCATION_CACHE:
        return _LOCATION_CACHE[key]
    # Static fallback
    if key in STATIC_FBS_COORDS:
        _LOCATION_CACHE[key] = STATIC_FBS_COORDS[key]
        return _LOCATION_CACHE[key]
    # Geocode if allowed
    geo = geocode_team(team)
    if geo:
        lat, lon = geo
        try:
            _append_location(team, lat, lon, False)
        except Exception:
            pass
        _LOCATION_CACHE[key] = (lat, lon, False)
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
            r = requests.get(which, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                if VERBOSE:
                    _log(f'current weather status {r.status_code} lat={lat} lon={lon}')
                return None
            d = r.json()
            return {'temp_f': d.get('main',{}).get('temp'), 'wind_mph': d.get('wind',{}).get('speed')}
        elif delta_hours <= 6:  # near/ongoing => current
            r = requests.get(CURRENT_URL, params={'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                if VERBOSE:
                    _log(f'current weather status {r.status_code} lat={lat} lon={lon}')
                return None
            d = r.json()
            return {'temp_f': d.get('main',{}).get('temp'), 'wind_mph': d.get('wind',{}).get('speed')}
        elif delta_hours <= 120:  # future within 5 day forecast window => use forecast slots
            r = requests.get(FORECAST_URL, params={'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                if VERBOSE:
                    _log(f'forecast status {r.status_code} lat={lat} lon={lon}')
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
        else:  # >5 days out: fallback to current real weather (coverage priority)
            r = requests.get(CURRENT_URL, params={'lat': lat, 'lon': lon, 'units': 'imperial', 'appid': API_KEY}, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return None
            d = r.json()
            return {'temp_f': d.get('main',{}).get('temp'), 'wind_mph': d.get('wind',{}).get('speed')}
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
    """Parse to timezone-aware UTC datetime, preferring America/New_York for naive values."""
    if not dt_str or str(dt_str).lower() in ('nan','nat','none','null'):
        return None
    s = str(dt_str).strip().replace(' ', 'T')
    dt_obj: Optional[datetime] = None
    try:
        if s.endswith('Z'):
            dt_obj = datetime.fromisoformat(s.replace('Z','+00:00'))
        else:
            dt_obj = datetime.fromisoformat(s)
    except Exception:
        try:
            val = pd.to_datetime(s, errors='coerce')
            dt_obj = None if pd.isna(val) else val.to_pydatetime()
        except Exception:
            dt_obj = None
    if not dt_obj:
        return None
    # Localize naive to America/New_York, else pass-through tz, then convert to UTC
    if dt_obj.tzinfo is None:
        try:
            if ZoneInfo is not None:
                dt_obj = dt_obj.replace(tzinfo=ZoneInfo('America/New_York'))
            else:
                # Fallback: assume UTC to avoid double-shifting
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    try:
        return dt_obj.astimezone(timezone.utc)
    except Exception:
        return dt_obj

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
        # Prefer API kickoff time when available
        start_dt = _parse_start(row.get('start_date_api') or row.get('start_date'))
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

def enrich_fbs_games(df: pd.DataFrame, batch: int = 150, max_loops: int = 20, persist_every: int = 50, output_path: Optional[str] = None, verbose: bool | None = None) -> pd.DataFrame:
    """Iteratively enrich ALL FBS games (ignores horizon) for weather.

    - Fallback to current weather for games beyond forecast window (>5 days ahead).
    - Persists partial successes every 'persist_every' enriched rows if output_path provided.
    """
    if df.empty:
        return df
    for c in ['weather_temp','weather_wind','weather_adjustment']:
        if c not in df.columns:
            df[c] = pd.NA
    if 'enrichment_failed' not in df.columns:
        df['enrichment_failed'] = False
    if CAPTURE_DEBUG and 'weather_debug_reason' not in df.columns:
        df['weather_debug_reason'] = pd.NA
    fbs_mask = df['home_team'].apply(_is_fbs_team)
    need_weather = df['weather_temp'].isna() & df['weather_wind'].isna() & fbs_mask
    loops = 0
    enriched_count = 0
    total_target = int(need_weather.sum())
    start_ts = time.time()
    vb = VERBOSE if verbose is None else verbose
    if vb:
        _log(f'start FBS enrichment: target_rows={total_target} batch={batch} max_loops={max_loops}')
    while need_weather.any() and loops < max_loops:
        loops += 1
        target_idx = need_weather[need_weather].head(batch).index
        if not len(target_idx):
            break
        for idx in target_idx:
            row = df.loc[idx]
            # Prefer API kickoff time when available
            start_dt = _parse_start(row.get('start_date_api') or row.get('start_date'))
            if not start_dt:
                df.at[idx,'enrichment_failed'] = True
                if CAPTURE_DEBUG: df.at[idx,'weather_debug_reason'] = 'no_start_date'
                continue
            loc = ensure_team_location(row.get('home_team',''))
            if not loc:
                df.at[idx,'enrichment_failed'] = True
                if CAPTURE_DEBUG: df.at[idx,'weather_debug_reason'] = 'no_location'
                continue
            lat, lon, indoor = loc
            w = fetch_weather(lat, lon, start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc))
            if not w:
                df.at[idx,'enrichment_failed'] = True
                if CAPTURE_DEBUG: df.at[idx,'weather_debug_reason'] = 'weather_fetch_failed'
                continue
            temp_f = w.get('temp_f'); wind_mph = w.get('wind_mph')
            df.at[idx,'weather_temp'] = temp_f
            df.at[idx,'weather_wind'] = wind_mph
            df.at[idx,'weather_adjustment'] = compute_weather_adjustment(temp_f, wind_mph, indoor)
            enriched_count += 1
            if RATE_LIMIT_SLEEP > 0:
                time.sleep(RATE_LIMIT_SLEEP)
            if vb and enriched_count % PROGRESS_EVERY == 0:
                done = enriched_count
                remaining = int(need_weather.sum()) - 1  # this row just processed
                elapsed = time.time() - start_ts
                rate = done / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else float('inf')
                _log(f'progress: done={done}/{total_target} ({done/total_target:.1%}) rate={rate:.2f} r/s eta={eta/60:.1f}m loops={loops}')
            if output_path and enriched_count % persist_every == 0:
                try:
                    df.to_csv(output_path, index=False)
                    _log(f'progress persist: {enriched_count} rows enriched (file updated)')
                except Exception as e:
                    _log(f'persist error: {e}')
        need_weather = df['weather_temp'].isna() & df['weather_wind'].isna() & fbs_mask
    if output_path:
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            _log(f'final persist error: {e}')
    if vb:
        total_elapsed = time.time() - start_ts
        _log(f'completed: enriched={enriched_count} / {total_target} elapsed={total_elapsed:.1f}s avg_rate={(enriched_count/total_elapsed if total_elapsed>0 else 0):.2f} r/s remaining={int(need_weather.sum())}')
    return df

__all__ = [
    'enrich_dataframe','enrich_fbs_games','ensure_team_location','fetch_weather','compute_weather_adjustment'
]
