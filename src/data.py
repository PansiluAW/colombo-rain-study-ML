"""Download a reproducible Colombo ERA5 extract and construct next-day rows."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["rain_today", "humidity_today", "temp_max_today", "temp_min_today",
            "wind_today", "pressure_today", "rain_mean_3d", "rain_mean_7d",
            "pressure_change", "season_sin", "season_cos"]
LABELS = ["Today's rain (mm)", "Today's mean humidity (%)", "Today's maximum temperature (C)",
          "Today's minimum temperature (C)", "Today's maximum wind (km/h)",
          "Today's mean pressure (hPa)", "Mean rain over last 3 days (mm)",
          "Mean rain over last 7 days (mm)", "Pressure change since yesterday (hPa)",
          "Target-date season (sine)", "Target-date season (cosine)"]

def download():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    query = dict(latitude=6.9271, longitude=79.8612, start_date='2014-01-01',
        end_date='2025-12-31', models='era5', timezone='Asia/Colombo',
        daily=','.join(['temperature_2m_max','temperature_2m_min','precipitation_sum',
        'relative_humidity_2m_mean','wind_speed_10m_max','surface_pressure_mean']))
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=90) as response:
        content = response.read()
    result = json.loads(content)
    if result.get('error'): raise RuntimeError(result)
    (folder/'colombo_era5_raw.json').write_bytes(content)
    pd.DataFrame(result['daily']).to_csv(folder/'colombo_daily.csv', index=False)
    provenance = dict(source='Open-Meteo Historical Weather API; ECMWF ERA5 reanalysis',
        requested_coordinates=[6.9271,79.8612], returned_coordinates=[result['latitude'],result['longitude']],
        timezone=result['timezone'], units=result['daily_units'], url=url,
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_sha256=hashlib.sha256(content).hexdigest(), license='CC BY 4.0',
        attribution='Weather data by Open-Meteo.com, using ECMWF ERA5.',
        modifications='Daily extract converted to CSV; lagged/rolling and seasonal predictors derived.')
    (folder/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Downloaded {len(result["daily"]["time"])} daily records')

def make_frame():
    d = pd.read_csv(ROOT/'data/colombo_daily.csv', parse_dates=['time'])
    assert d.time.is_unique and d.time.is_monotonic_increasing
    assert (d.time.diff().dropna() == pd.Timedelta(days=1)).all()
    assert d.drop(columns='time').notna().all().all(), 'Missing raw values require review'
    assert (d.precipitation_sum >= 0).all()
    x = pd.DataFrame({'observation_date': d.time, 'target_date': d.time + pd.Timedelta(days=1)})
    for feature, raw in [('rain_today','precipitation_sum'),('humidity_today','relative_humidity_2m_mean'),
        ('temp_max_today','temperature_2m_max'),('temp_min_today','temperature_2m_min'),
        ('wind_today','wind_speed_10m_max'),('pressure_today','surface_pressure_mean')]:
        x[feature] = d[raw]
    x['rain_mean_3d'] = d.precipitation_sum.rolling(3, min_periods=3).mean()
    x['rain_mean_7d'] = d.precipitation_sum.rolling(7, min_periods=7).mean()
    x['pressure_change'] = d.surface_pressure_mean.diff()
    angle = 2*np.pi*(x.target_date.dt.dayofyear-1)/365.25
    x['season_sin'], x['season_cos'] = np.sin(angle), np.cos(angle)
    next_rain = d.precipitation_sum.shift(-1)
    x['target_rain_mm'] = next_rain
    x['wet_tomorrow'] = (next_rain >= 1.0).astype(int)
    # Drop the first six incomplete windows and the final unavailable target.
    return x.dropna().reset_index(drop=True)

def splits(d):
    return [d[d.target_date < '2022-01-01'].copy(),
            d[(d.target_date >= '2022-01-01') & (d.target_date < '2024-01-01')].copy(),
            d[d.target_date >= '2024-01-01'].copy()]

if __name__ == '__main__': download()
