import os
import joblib
import pandas as pd
import requests
import hopsworks
import logging
from datetime import datetime
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    try:
        
        load_dotenv()

        logging.info("Connecting to Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()

    
        logging.info("Downloading model from registry...")

        model_meta = mr.get_model("karachi_aqi_model", version=1)
        model_dir = model_meta.download()

        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(model_path):
            raise Exception("Model file missing")

        if not os.path.exists(scaler_path):
            raise Exception("Scaler file missing")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        logging.info("Model and Scaler loaded successfully")

        logging.info("Loading historical AQI data...")

        aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
        hist_df = aqi_fg.read().sort_values('time').reset_index(drop=True)

        if hist_df.empty:
            raise Exception("Historical dataset is empty")

        if hist_df['pm2_5'].isna().all():
            raise Exception("Historical PM2.5 column is all null")

        logging.info(f"Historical data loaded. Last PM2.5: {hist_df['pm2_5'].iloc[-1]}")

        def get_future_weather():
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": 24.8607,
                "longitude": 67.0011,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "forecast_days": 3,
                "timezone": "auto"
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if "hourly" not in data:
                    raise Exception("Weather API response missing 'hourly'")

                return pd.DataFrame(data["hourly"])

            except Exception as e:
                logging.error(f"Weather API failed: {e}")
                raise

        future_weather_df = get_future_weather()

        if future_weather_df.empty:
            raise Exception("Weather forecast data empty")

        logging.info(f"Future weather fetched for {len(future_weather_df)} hours")

        def generate_aligned_forecast(model, scaler, history_df, weather_forecast):

            predictions_list = []
            current_window = history_df.copy()

            for i in range(len(weather_forecast)):

                weather_row = weather_forecast.iloc[i]
                target_time = pd.to_datetime(weather_row['time'])

                data = {
                    'temperature_2m': weather_row['temperature_2m'],
                    'relative_humidity_2m': weather_row['relative_humidity_2m'],
                    'wind_speed_10m': weather_row['wind_speed_10m'],
                    'hour': target_time.hour,
                    'pm2_5_lag_1h': current_window['pm2_5'].iloc[-1],
                    'pm2_5_lag_24h': current_window['pm2_5'].iloc[-24] if len(current_window) >= 24 else current_window['pm2_5'].iloc[-1]
                }

                features_df = pd.DataFrame([data])[[
                    'temperature_2m',
                    'relative_humidity_2m',
                    'wind_speed_10m',
                    'hour',
                    'pm2_5_lag_1h',
                    'pm2_5_lag_24h'
                ]]

                scaled_features = scaler.transform(features_df)
                prediction = model.predict(scaled_features)[0]

                predictions_list.append({
                    'city': 'Karachi',
                    'prediction_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_pm2_5': round(float(prediction), 2),
                    'forecast_hour_out': i + 1
                })

                new_row = {
                    'time': int(target_time.timestamp() * 1000),
                    'pm2_5': prediction
                }

                current_window = pd.concat(
                    [current_window, pd.DataFrame([new_row])],
                    ignore_index=True
                )

            return pd.DataFrame(predictions_list)

        logging.info("Generating AQI forecasts...")

        predictions_df = generate_aligned_forecast(
            model,
            scaler,
            hist_df,
            future_weather_df
        )

        if predictions_df.empty:
            raise Exception("Prediction dataframe is empty")

        
        logging.info("Uploading forecasts to Hopsworks...")

        aqi_predictions_fg = fs.get_or_create_feature_group(
            name="aqi_predictions",
            version=1,
            primary_key=['city', 'prediction_time'],
            description="Authentic 72-hour AQI predictions aligned with Open-Meteo timestamps",
            online_enabled=True
        )

        aqi_predictions_fg.insert(predictions_df)

        logging.info("Forecast upload completed successfully")

    except Exception as e:
        logging.error(f"Inference pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
