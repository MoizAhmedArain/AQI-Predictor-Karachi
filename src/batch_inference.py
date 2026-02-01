import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

model_meta = mr.get_model("karachi_aqi_model", version=2)
model_dir = model_meta.download()
model = joblib.load(os.path.join(model_dir, "model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
print(" Model (v2) and Scaler loaded successfully!")

# GET LATEST DATA 
aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
df = aqi_fg.read()
df = df.sort_values('time').reset_index(drop=True)
print(f"Retrieved recent data. Latest PM2.5: {df['pm2_5'].iloc[-1]}")

#THE FEATURE BUILDER FUNCTION 
def create_forecast_row(current_df, target_time):
    last_row = current_df.iloc[-1]
    data = {
        'temperature_2m': last_row['temperature_2m'],
        'relative_humidity_2m': last_row['relative_humidity_2m'],
        'wind_speed_10m': last_row['wind_speed_10m'],
        'hour': target_time.hour,
        'pm2_5_lag_1h': last_row['pm2_5'],
        'pm2_5_lag_24h': current_df['pm2_5'].iloc[-24] if len(current_df) >= 24 else last_row['pm2_5']
    }
    feature_names = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour', 'pm2_5_lag_1h', 'pm2_5_lag_24h']
    features_df = pd.DataFrame([data])[feature_names]
    return scaler.transform(features_df)

# THE 72-HOUR LOOP 
print(f" Generating 72-hour forecast...")
predictions_list = []
current_data_window = df.copy()
start_time = datetime.now()

for i in range(72):
    target_time = start_time + timedelta(hours=i+1)
    input_features = create_forecast_row(current_data_window, target_time)
    prediction = model.predict(input_features)[0]
    
    predictions_list.append({
        'city': 'Karachi',
        'prediction_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
        'predicted_pm2_5': round(float(prediction), 2),
        'forecast_hour_out': i + 1
    })
    
    # Add this prediction to the window for the next hour's lag
    new_row = current_data_window.iloc[-1].copy()
    new_row['pm2_5'] = prediction
    new_row['time'] = int(target_time.timestamp() * 1000)
    current_data_window = pd.concat([current_data_window, pd.DataFrame([new_row])], ignore_index=True)

predictions_df = pd.DataFrame(predictions_list)

# UPLOAD TO HOPSWORKS 
print(" Uploading forecasts to Hopsworks...")
aqi_predictions_fg = fs.get_or_create_feature_group(
    name="aqi_predictions",
    version=1,
    primary_key=['city', 'prediction_time'],
    description="72-hour AQI predictions for Karachi",
    online_enabled=True
)
aqi_predictions_fg.insert(predictions_df)
print(" Done! Karachi's future air quality is now live in the cloud.")