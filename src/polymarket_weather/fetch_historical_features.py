import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.polymarket_weather.config import CITIES

def fetch_historical_features():
    os.makedirs("data/weather", exist_ok=True)
    start_date = "2015-01-01"
    end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    for city, config in CITIES.items():
        logger.info(f"Fetching features for {city}")
        params = {
            "latitude": config["lat"],
            "longitude": config["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "timezone": "auto"
        }
        
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        
        data = resp.json()
        if "daily" not in data or not data["daily"].get("time"):
            raise ValueError(f"Downloaded empty feature dataset for {city}")
        
        df = pd.DataFrame({
            "date_local": data["daily"]["time"],
            "grid_temp_max_c": data["daily"]["temperature_2m_max"]
        })
        
        df = df.dropna(subset=['grid_temp_max_c'])
        
        if len(df) <= 3000:
            raise ValueError(f"Feature dataset for {city} has only {len(df)} rows, expected > 3000")
        
        city_slug = city.replace(' ', '_').lower()
        output_path = f"data/weather/{city_slug}_historical_features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    fetch_historical_features()
