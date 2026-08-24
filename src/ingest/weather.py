"""
Ingest historical weather data from Open-Meteo (free API, no key required).
"""
import os
import pandas as pd
import requests
from pathlib import Path


# NFL stadiums with lat/lon and roof type
STADIUM_INFO = {
    'ARZ': {'lat': 33.7577, 'lon': -112.3633, 'roof': 'retractable'},
    'ATL': {'lat': 33.7490, 'lon': -84.3880, 'roof': 'retractable'},
    'BAL': {'lat': 39.2781, 'lon': -76.6169, 'roof': 'open'},
    'BUF': {'lat': 42.7735, 'lon': -78.7870, 'roof': 'open'},
    'CAR': {'lat': 35.1358, 'lon': -80.8530, 'roof': 'open'},
    'CHI': {'lat': 41.8623, 'lon': -87.6165, 'roof': 'open'},
    'CIN': {'lat': 39.0954, 'lon': -84.5163, 'roof': 'open'},
    'CLE': {'lat': 41.5054, 'lon': -81.6995, 'roof': 'open'},
    'DAL': {'lat': 32.8975, 'lon': -97.0382, 'roof': 'retractable'},
    'DEN': {'lat': 39.7439, 'lon': -104.8202, 'roof': 'open'},
    'DET': {'lat': 42.6394, 'lon': -83.1797, 'roof': 'dome'},
    'GB': {'lat': 44.5013, 'lon': -88.0622, 'roof': 'open'},
    'HOU': {'lat': 29.6847, 'lon': -95.4107, 'roof': 'retractable'},
    'IND': {'lat': 39.7604, 'lon': -86.1613, 'roof': 'dome'},
    'JAX': {'lat': 30.3239, 'lon': -81.6373, 'roof': 'open'},
    'KC': {'lat': 39.0487, 'lon': -94.4837, 'roof': 'open'},
    'LA': {'lat': 33.9535, 'lon': -118.2398, 'roof': 'open'},
    'LAR': {'lat': 33.9535, 'lon': -118.2398, 'roof': 'dome'},
    'LV': {'lat': 36.0837, 'lon': -115.1537, 'roof': 'dome'},
    'MIA': {'lat': 25.9581, 'lon': -80.2389, 'roof': 'open'},
    'MIN': {'lat': 44.9736, 'lon': -93.2618, 'roof': 'dome'},
    'NE': {'lat': 42.0895, 'lon': -71.2642, 'roof': 'open'},
    'NO': {'lat': 29.9511, 'lon': -90.0815, 'roof': 'dome'},
    'NYG': {'lat': 40.8149, 'lon': -74.0740, 'roof': 'open'},
    'NYJ': {'lat': 40.8149, 'lon': -74.0740, 'roof': 'open'},
    'PHI': {'lat': 39.9576, 'lon': -75.1675, 'roof': 'open'},
    'PIT': {'lat': 40.4465, 'lon': -80.0157, 'roof': 'open'},
    'SF': {'lat': 37.4038, 'lon': -121.9690, 'roof': 'open'},
    'SEA': {'lat': 47.5951, 'lon': -122.3316, 'roof': 'open'},
    'TB': {'lat': 27.9759, 'lon': -82.5033, 'roof': 'dome'},
    'TEN': {'lat': 36.1627, 'lon': -86.7816, 'roof': 'open'},
    'WAS': {'lat': 38.9076, 'lon': -77.1186, 'roof': 'open'},
}


def get_stadium_info(team):
    """
    Get lat/lon and roof type for a team.

    Args:
        team: 2-3 letter team code (e.g., 'GB', 'NYG')

    Returns:
        dict with 'lat', 'lon', 'roof' keys, or None if not found
    """
    return STADIUM_INFO.get(team)


def fetch_historical_weather(lat, lon, date, cache_dir="data/raw/weather"):
    """
    Fetch historical weather from Open-Meteo archive API.

    Args:
        lat, lon: latitude, longitude
        date: date string in YYYY-MM-DD format
        cache_dir: directory to cache JSON responses

    Returns:
        dict with 'temp_f', 'wind_mph', 'precip_in' for the given date, or None on error
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date,
        'end_date': date,
        'hourly': 'temperature_2m,wind_speed_10m,precipitation,snowfall',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'auto',
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract midday (12:00 UTC) weather as proxy for game-time conditions
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        temps = hourly.get('temperature_2m', [])
        winds = hourly.get('wind_speed_10m', [])
        precips = hourly.get('precipitation', [])

        if len(times) > 12:
            # Noon UTC is a reasonable proxy
            return {
                'temp_f': temps[12],
                'wind_mph': winds[12],
                'precip_in': precips[12],
            }
        else:
            # Use first available
            return {
                'temp_f': temps[0] if temps else None,
                'wind_mph': winds[0] if winds else None,
                'precip_in': precips[0] if precips else None,
            }
    except Exception as e:
        print(f"Warning: Failed to fetch weather for ({lat}, {lon}, {date}): {e}")
        return None


def add_weather_to_games(schedule_df, cache_dir="data/raw/weather"):
    """
    Add weather columns to schedule DataFrame.
    Only fetches for outdoor/retractable-roof games.

    Args:
        schedule_df: DataFrame with columns home_team, away_team, game_date (or similar)
        cache_dir: directory to cache weather data

    Returns:
        schedule_df with added columns temp_f, wind_mph, precip_in
    """
    result = schedule_df.copy()
    result['temp_f'] = None
    result['wind_mph'] = None
    result['precip_in'] = None

    # Iterate over games and fetch weather for outdoor stadiums
    for idx, row in result.iterrows():
        home_team = row.get('home_team')
        stadium_info = get_stadium_info(home_team)

        if stadium_info is None or stadium_info['roof'] == 'dome':
            # Skip domes; use None/median impute later
            continue

        # Get game date; format varies by source
        date = row.get('gameday') or row.get('game_date') or row.get('date')
        if date is None:
            continue

        # Ensure date is YYYY-MM-DD
        if hasattr(date, 'strftime'):
            date = date.strftime('%Y-%m-%d')
        else:
            date = str(date)[:10]

        weather = fetch_historical_weather(
            stadium_info['lat'],
            stadium_info['lon'],
            date,
            cache_dir=cache_dir
        )

        if weather:
            result.loc[idx, 'temp_f'] = weather['temp_f']
            result.loc[idx, 'wind_mph'] = weather['wind_mph']
            result.loc[idx, 'precip_in'] = weather['precip_in']

    return result


if __name__ == "__main__":
    # Quick test
    info = get_stadium_info('GB')
    print(f"Green Bay info: {info}")

    weather = fetch_historical_weather(info['lat'], info['lon'], '2024-01-14')
    print(f"Weather for 2024-01-14 in Green Bay: {weather}")
