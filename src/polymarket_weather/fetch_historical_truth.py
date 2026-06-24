import numpy as np
np.NaN = np.nan
import pandas as pd
from meteostat import daily, Parameter
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CITIES
from resolution_anchors import RESOLUTION_ANCHORS  # single source of truth for station ids

def fetch_historical_truth():
    os.makedirs("data/weather", exist_ok=True)
    end_date = datetime.now() - timedelta(days=1)
    start_date = datetime(2015, 1, 1)

    for city, config in CITIES.items():
        # Truth-label station comes from the anchor (falls back to config's ICAO if absent).
        station_id = RESOLUTION_ANCHORS.get(city, {}).get("meteostat_id", config['station_id'])
        logger.info(f"Fetching truth for {city} (Station: {station_id})")
        
        data = daily(station_id, start_date, end_date,
                     parameters=[Parameter.TMAX, Parameter.TMIN]).fetch()
        if data.empty:
            raise ValueError(f"Downloaded empty dataset for {city} (Station: {station_id})")

        df = data.reset_index().rename(columns={
            'time': 'date_local', 'tmax': 'temp_max_c', 'tmin': 'temp_min_c'})
        df = df[['date_local', 'temp_max_c', 'temp_min_c']]
        df['date_local'] = pd.to_datetime(df['date_local']).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['temp_max_c'])
        
        if len(df) <= 3000:
            logger.critical(f"WARNING: Dataset for {city} has {len(df)} rows, expected > 3000.")
            
        city_slug = city.replace(' ', '_').lower()
        output_path = f"data/weather/{city_slug}_historical_actuals.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    fetch_historical_truth()
