import os
import joblib
import pandas as pd
import requests
import hopsworks
import logging
from datetime import datetime
from dotenv import load_dotenv
import time

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        # 1. INITIALIZATION
        load_dotenv()
        logging.info("Connecting to Hopsworks...")
        
        # MODEL SESSION
        project_model = hopsworks.login()
        mr = project_model.get_model_registry()

        model_meta = mr.get_model("karachi_aqi_model", version=2)
        model_dir = model_meta.download()

        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

        project_model.logout()


        # FEATURE STORE SESSION
        project_fs = hopsworks.login()
        fs = project_fs.get_feature_store()

        aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

        for attempt in range(3):
            try:
                hist_df = aqi_fg.read(read_options={"use_hive": False})
                break
            except:
                time.sleep(5)

        hist_df = hist_df.sort_values('time').reset_index(drop=True)

        project_fs.logout()
        logging.info("Done Feature Store connection and data retrieval.")


        # 4. FETCH WEATHER FORECAST (Open-Meteo)
        logging.info("Fetching 72-hour weather forecast...")
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 24.8607,
            "longitude": 67.0011,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "forecast_days": 3,
            "timezone": "auto"
        }
        resp = requests.get(weather_url, params=params, timeout=30)
        resp.raise_for_status()
        future_weather_df = pd.DataFrame(resp.json()["hourly"])
        
        logging.info(f" Weather forecast fetched for {len(future_weather_df)} hours.")

        # 5. GENERATE ALIGNED FORECAST (Sliding Window)
        logging.info("Running inference loop...")
        predictions_list = []
        current_window = hist_df.copy()

        for i in range(len(future_weather_df)):
            weather_row = future_weather_df.iloc[i]
            target_time = pd.to_datetime(weather_row['time'])

            # Calculation of features (Lag and Hour)
            # Safety: If history is shorter than 24h, use the last available PM2.5 for the 24h lag
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

            # Define the EXACT order the model was trained on
            feature_order = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour', 'pm2_5_lag_1h', 'pm2_5_lag_24h']
            features_df = pd.DataFrame([input_data])[feature_order]

            # Scale and Predict
            scaled_features = scaler.transform(features_df)
            prediction = model.predict(scaled_features)[0]

            # Store the prediction results
            predictions_list.append({
                'city': 'Karachi',
                'prediction_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_pm2_5': round(float(prediction), 2),
                'forecast_hour_out': i + 1
            })

            # Update the window so the next iteration can use this prediction as a lag
            new_row = pd.DataFrame([{
                'time': int(target_time.timestamp() * 1000),
                'pm2_5': prediction,
                'city': 'Karachi'
            }])
            current_window = pd.concat([current_window, new_row], ignore_index=True)

        predictions_final_df = pd.DataFrame(predictions_list)

        # 6. UPLOAD TO HOPSWORKS
        logging.info("Uploading results to Hopsworks...")
        pred_fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=['city', 'prediction_time'],
            description="72-hour forecast for Karachi AQI",
            online_enabled=True
        )
        
        # Write the batch of predictions
        pred_fg.insert(predictions_final_df, write_options={"wait_for_job": False})
        
        logging.info(" SUCCESS! Batch inference completed.")

    except Exception as e:
        logging.error(f" PIPELINE FAILED: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
