import requests
import pandas as pd
import hopsworks
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:

    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs = project.get_feature_store()

    # the existing Feature Group from hopsworks.ai
    aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

    def get_latest_data():
        # the last 2 days to ensure no gaps
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
        
        df = pd.merge(df_w, df_aq, on="time", how="left")
        df['city'] = 'karachi'
        df['time'] = pd.to_datetime(df['time']).apply(lambda x: int(x.timestamp() * 1000))
        return df

    latest_df = get_latest_data()

    # table name in the feature group
    aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

    # Fetch data Raw from API
    latest_df = get_latest_data() 

    try:
        aqi_fg.insert(latest_df)
    except Exception as e:
        print(f"Error: {e}")
except Exception as e:
    print(f"Feature_pipeline error: {e}")
    