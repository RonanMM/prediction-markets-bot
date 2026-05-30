import numpy as np
np.NaN = np.nan
import pandas as pd
from meteostat import Daily, Stations
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.polymarket_weather.config import CITIES

STATION_OVERRIDE = {
    "London": "EGLC0",
    "New York City": "72503",
    "Chicago": "72530",
    "Seoul": "47113",
    "Hong Kong": "45007"
}

def fetch_historical_truth():
    os.makedirs("data/weather", exist_ok=True)
    end_date = datetime.now() - timedelta(days=1)
    start_date = datetime(2015, 1, 1)

    for city, config in CITIES.items():
        station_id = STATION_OVERRIDE.get(city, config['station_id'])
        logger.info(f"Fetching truth for {city} (Station: {station_id})")
        
        data = Daily(station_id, start_date, end_date)
        data = data.fetch()
        
        if data.empty:
            raise ValueError(f"Downloaded empty dataset for {city} (Station: {station_id})")
        
        df = data.reset_index()
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'date_local'})
            
        df = df[['date_local', 'tmax']]
        df = df.rename(columns={'tmax': 'temp_max_c'})
        
        df['date_local'] = df['date_local'].dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['temp_max_c'])
        
        if len(df) <= 3000:
            logger.critical(f"WARNING: Dataset for {city} has {len(df)} rows, expected > 3000.")
            
        city_slug = city.replace(' ', '_').lower()
        output_path = f"data/weather/{city_slug}_historical_actuals.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    fetch_historical_truth()
