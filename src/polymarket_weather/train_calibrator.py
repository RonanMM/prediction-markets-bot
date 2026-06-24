import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CITIES

# Canonical model dir: src/polymarket_weather/models — the SAME dir ml_calibrator.py loads from,
# anchored to this file so training updates the live models regardless of the working directory.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def train_calibrator():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for city in CITIES.keys():
        logger.info(f"Training model for {city}")
        city_slug = city.replace(' ', '_').lower()
        actuals_path = f"data/weather/{city_slug}_historical_actuals.csv"
        features_path = f"data/weather/{city_slug}_historical_features.csv"
        
        try:
            df_actuals = pd.read_csv(actuals_path)
            df_features = pd.read_csv(features_path)
        except Exception as e:
            logger.error(f"Missing data for {city}: {e}")
            continue
            
        df_actuals['date_local'] = pd.to_datetime(df_actuals['date_local']).dt.strftime('%Y-%m-%d')
        df_features['date_local'] = pd.to_datetime(df_features['date_local']).dt.strftime('%Y-%m-%d')
        
        df = pd.merge(df_features, df_actuals, on='date_local', how='inner')
        
        if len(df) <= 2500:
            raise ValueError(f"Merged dataset for {city} has {len(df)} rows, expected > 2500")
            
        df['day_of_year'] = pd.to_datetime(df['date_local']).dt.dayofyear
        
        X = df[['grid_temp_max_c', 'day_of_year']]
        y = df['temp_max_c']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        logger.info(f"{city} Model RMSE: {rmse:.4f}")
        
        joblib.dump(model, os.path.join(MODELS_DIR, f"{city_slug}_calibrator.joblib"))
        with open(os.path.join(MODELS_DIR, f"{city_slug}_sigma.json"), "w") as f:
            json.dump({"sigma": rmse}, f)
            
if __name__ == "__main__":
    train_calibrator()
