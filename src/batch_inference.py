import os
import joblib
import pandas as pd
import requests
import hopsworks
import logging
from datetime import datetime
from dotenv import load_dotenv

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    try:
        # 1. INITIALIZATION
        load_dotenv() # Ensure .env is loaded
        logging.info("Connecting to Hopsworks...")
        
        # Explicit login is safer for local/CI environments
        project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()

        # 2. LOAD MODEL & SCALER (v2)
        logging.info("Downloading model artifact (v2)...")
        model_meta = mr.get_model("karachi_aqi_model", version=2)
        model_dir = model_meta.download()
        
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or Scaler file missing from download directory.")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("Model and Scaler loaded successfully.")

        # 3. LOAD HISTORICAL DATA (Using Feature View for Stability)
        logging.info("Retrieving historical AQI data...")
        try:
            # We use the View to avoid the DuckDB 'time' column Binder error
            fv = fs.get_feature_view(name="karachi_aqi_view", version=1)
            hist_df = fv.get_batch_data()
        except Exception as e:
            logging.warning(f"Feature View read failed ({e}). Falling back to direct Feature Group read...")
            aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
            hist_df = aqi_fg.read()

        if hist_df is None or hist_df.empty:
            raise Exception("Historical data is empty. Cannot generate lags.")

        hist_df = hist_df.sort_values('time').reset_index(drop=True)
        logging.info(f" History loaded. Latest PM2.5: {hist_df['pm2_5'].iloc[-1]}")

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
        logging.info(" Running inference loop...")
        predictions_list = []
        # 'current_window' stores the history + new predictions to calculate lags
        current_window = hist_df.copy()

        for i in range(len(future_weather_df)):
            weather_row = future_weather_df.iloc[i]
            target_time = pd.to_datetime(weather_row['time'])

            # Prepare features exactly in the order the model saw during training
            input_data = {
                'temperature_2m': weather_row['temperature_2m'],
                'relative_humidity_2m': weather_row['relative_humidity_2m'],
                'wind_speed_10m': weather_row['wind_speed_10m'],
                'hour': target_time.hour,
                'pm2_5_lag_1h': current_window['pm2_5'].iloc[-1],
                'pm2_5_lag_24h': current_window['pm2_5'].iloc[-24] if len(current_window) >= 24 else current_window['pm2_5'].iloc[-1]
            }

            features_df = pd.DataFrame([input_data])[[
                'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                'hour', 'pm2_5_lag_1h', 'pm2_5_lag_24h'
            ]]

            # Predict
            scaled_features = scaler.transform(features_df)
            prediction = model.predict(scaled_features)[0]

            # Save prediction
            predictions_list.append({
                'city': 'Karachi',
                'prediction_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_pm2_5': round(float(prediction), 2),
                'forecast_hour_out': i + 1
            })

            # Update the window so the NEXT hour can use THIS prediction as its 'lag_1h'
            new_row = pd.DataFrame([{
                'time': int(target_time.timestamp() * 1000),
                'pm2_5': prediction
            }])
            current_window = pd.concat([current_window, new_row], ignore_index=True)

        predictions_final_df = pd.DataFrame(predictions_list)

        # 6. UPLOAD TO HOPSWORKS
        logging.info(" Uploading results to Hopsworks...")
        pred_fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=['city', 'prediction_time'],
            description="72-hour forecast based on real Open-Meteo weather data",
            online_enabled=True
        )
        pred_fg.insert(predictions_final_df)
        
        logging.info("✨ SUCCESS! Batch inference completed and uploaded.")

    except Exception as e:
        logging.error(f" PIPELINE FAILED: {str(e)}")
        raise e

if __name__ == "__main__":
    main()