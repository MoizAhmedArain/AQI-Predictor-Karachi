import os
import joblib
import pandas as pd
import requests
import hopsworks
import logging
from datetime import datetime
from dotenv import load_dotenv
import time
import logging

#  logging to use UTC
logging.Formatter.converter = time.gmtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
       
        load_dotenv()
        logging.info("Connecting to Hopsworks...")
        
        project = hopsworks.login()
        mr = project.get_model_registry()
        fs = project.get_feature_store()

        # 2. LOAD MODEL & SCALER
        logging.info("Downloading model artifact (v2)...")
        model_meta = mr.get_model("karachi_aqi_model", version=1)
        model_dir = model_meta.download()

        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        logging.info("Model and Scaler loaded successfully.")
        time.sleep(3)

        # 3. FEATURE STORE DATA RETRIEVAL
        logging.info("Accessing Feature Group: karachi_aqi_weather")
        aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

        hist_df = None
        for attempt in range(3):
            try:
                hist_df = aqi_fg.read(read_options={"use_hive": False})
                if not hist_df.empty:
                    break
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(5)

        if hist_df is None or hist_df.empty:
            raise Exception("No data found in Feature Group!")

        # lowercase
        hist_df.columns = [c.lower() for c in hist_df.columns]

        # historical time is treated as UTC
        hist_df['time'] = pd.to_datetime(hist_df['time'], utc=True)

        hist_df = hist_df.sort_values('time').reset_index(drop=True)
        logging.info(f"Historical data loaded. Rows: {len(hist_df)}")

        # 4. FETCH WEATHER FORECAST
        logging.info("Fetching 72-hour weather forecast...")
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 24.8607,
            "longitude": 67.0011,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "forecast_days": 3,
            "timezone": "UTC"   #  UTC instead of auto
        }

        resp = requests.get(weather_url, params=params, timeout=30)
        resp.raise_for_status()
        future_weather_df = pd.DataFrame(resp.json()["hourly"])

        # 5. GENERATE ALIGNED FORECAST
        logging.info("Running inference loop...")
        predictions_list = []
        current_window = hist_df.copy()

        for i in range(len(future_weather_df)):
            weather_row = future_weather_df.iloc[i]

            #  Parse forecast time as UTC
            target_time = pd.to_datetime(weather_row['time'], utc=True)

            pm25_lag_1 = current_window['pm2_5'].iloc[-1]
            pm25_lag_24 = current_window['pm2_5'].iloc[-24] if len(current_window) >= 24 else pm25_lag_1

            input_data = {
                'temperature_2m': weather_row['temperature_2m'],
                'relative_humidity_2m': weather_row['relative_humidity_2m'],
                'wind_speed_10m': weather_row['wind_speed_10m'],
                'hour': target_time.hour,
                'pm2_5_lag_1h': pm25_lag_1,
                'pm2_5_lag_24h': pm25_lag_24
            }

            feature_order = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour', 'pm2_5_lag_1h', 'pm2_5_lag_24h']
            features_df = pd.DataFrame([input_data])[feature_order]

            scaled_features = scaler.transform(features_df)
            prediction = model.predict(scaled_features)[0]

            predictions_list.append({
                'city': 'Karachi',
                'prediction_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_pm2_5': round(float(prediction), 2),
                'forecast_hour_out': i + 1
            })

            new_row = pd.DataFrame([{
                'time': int(target_time.timestamp() * 1000),
                'pm2_5': prediction,
                'city': 'Karachi'
            }])

            current_window = pd.concat([current_window, new_row], ignore_index=True)

        predictions_final_df = pd.DataFrame(predictions_list)

        # 6. UPLOAD TO HOPSWORKS
        logging.info("Uploading results to Feature Group: aqi_predictions")
        pred_fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=['city', 'prediction_time'],
            description="72-hour forecast for Karachi AQI",
            online_enabled=True
        )
        
        for attempt in range(3):
            try:
                pred_fg.insert(predictions_final_df)
                print("Insert successful")
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed. Retrying...")
                time.sleep(10)
                logging.info(" SUCCESS! Batch inference completed.")

    except Exception as e:
        logging.error(f"PIPELINE FAILED: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
