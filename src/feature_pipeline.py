import requests
import pandas as pd
import hopsworks
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_latest_data():
    """
    Fetches the last 2 days of real observed AQI and Weather data.
    """
    logging.info("Fetching observed data from Open-Meteo...")
    
    lat, lon = 24.8607, 67.0011
    
    # AQI API - Ground truth PM2.5
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5", 
        "past_days": 2, 
        "timezone": "auto"
    }
    
    # Weather API - Observed conditions
    # We use forecast_days=0 because we only want historical data for the feature store
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days": 2, 
        "forecast_days": 0, 
        "timezone": "auto"
    }

    try:
        aq_resp = requests.get(aq_url, params=aq_params).json()
        w_resp = requests.get(w_url, params=w_params).json()

        df_aq = pd.DataFrame(aq_resp["hourly"])
        df_w = pd.DataFrame(w_resp["hourly"])

        # Inner merge ensures we only keep rows where both AQI and Weather exist
        df = pd.merge(df_w, df_aq, on="time", how="inner")
        
        # Add metadata
        df['city'] = 'karachi'
        
        # Convert time to Unix Milliseconds (Integer) for Hopsworks compatibility
        df['time'] = pd.to_datetime(df['time']).apply(lambda x: int(x.timestamp() * 1000))
        
        logging.info(f"Prepared {len(df)} rows of data.")
        return df
    except Exception as e:
        logging.error(f"API Fetch Error: {e}")
        raise

def main():
    try:
        load_dotenv()
        
        # 2. CONNECT TO HOPSWORKS
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME")
        )
        fs = project.get_feature_store()

        # 3. GET DATA
        latest_df = get_latest_data()

        # 4. CREATE/GET FEATURE GROUP
        # Note: If you haven't deleted the old group yet, do it in the UI now!
        aqi_fg = fs.get_or_create_feature_group(
            name="karachi_aqi_weather",
            version=1,
            primary_key=['city', 'time'], # Composite key for better organization
            event_time='time',
            online_enabled=True,          # Required to fix the 'Binder Error' in inference
            description="Karachi AQI and Weather historical data"
        )

        # 5. INSERT DATA
        # insert() will handle both initial upload and future updates (upserts)
        logging.info("Uploading data to Hopsworks...")
        aqi_fg.insert(latest_df)
        
        logging.info(" Feature Pipeline completed successfully.")

    except Exception as e:
        logging.error(f" Pipeline Failed: {e}")
        raise

if __name__ == "__main__":
    main()