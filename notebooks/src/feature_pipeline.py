import requests
import pandas as pd
import hopsworks
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 1. Setup & Login in hopsworks.ai
load_dotenv()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)
fs = project.get_feature_store()

# 2. Get the existing Feature Group from hopsworks.ai
aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

def get_latest_data():
    # It Fetch only the last 2 days to ensure no gaps
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": "pm2_5,pm10",
        "past_days": 2, 
        "timezone": "auto"
    }
    
    # Weather for features(karachi city)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days": 2,
        "forecast_days": 0
    }

    aq_resp = requests.get(url, params=params).json()
    w_resp = requests.get(weather_url, params=weather_params).json()
    
    df_aq = pd.DataFrame(aq_resp["hourly"])
    df_w = pd.DataFrame(w_resp["hourly"])
    
    df = pd.merge(df_aq, df_w, on="time")
    df['city'] = 'karachi'
    df['time'] = pd.to_datetime(df['time']).apply(lambda x: int(x.timestamp() * 1000))
    return df

latest_df = get_latest_data()

def add_features(df):
    # Convert back to datetime for extracting time parts
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    
    # 1. Time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # 2. Lag Feature: What was the PM2.5 24 hours ago?
    # Note: For real-time pipelines, you'd usually pull the lag from the Feature Store
    # For now, we calculate it from our fetched 2-day window
    df['pm2_5_lag_24h'] = df['pm2_5'].shift(24)
    
    # Drop rows with NaN (the first 24 hours of our 2-day window will be empty)
    df = df.dropna()
    
    # Drop the helper column
    df = df.drop(columns=['datetime'])
    return df

# Apply engineering and upload
final_df = add_features(latest_df)
aqi_fg.insert(final_df)
print("Hourly features uploaded to Hopsworks!")