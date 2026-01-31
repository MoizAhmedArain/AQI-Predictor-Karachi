import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os


project = hopsworks.login(
    api_key_value="",
    project="Project1_AQI_predict"
)
fs = project.get_feature_store()

# 2. Get the RAW feature group from hopsworks.ai
aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

# 3. Define the Query so We can select all columns. This query is "Live" as the raw data grows, 
# the query automatically includes the new rows!
query = aqi_fg.select_all()

feature_view = fs.get_feature_view(name="karachi_aqi_view", version=1)

if feature_view is None:
    feature_view = fs.create_feature_view(
        name="karachi_aqi_view",
        version=1,
        query=query,
        labels=["pm2_5"]
    )
    print("Feature View created successfully!")
else:
    print("Feature View already exists.")



feature_view = fs.get_feature_view(name="karachi_aqi_view", version=1)

# 2. Fetch the Raw Data
# We get the full history to create our lags
df = feature_view.get_batch_data()
  
# 2. Get the RAW feature group (This contains ALL columns including pm2_5)
aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
df = aqi_fg.read() # This reads everything into a Pandas DataFrame

# --- DEBUG CHECK ---
print("--- COLUMNS FOUND ---")
print(df.columns.tolist()) # You will definitely see pm2_5 here now!

# 3. "On-the-Fly" Feature Engineering
def prepare_data(data):
    data = data.sort_values('time').reset_index(drop=True)
    
    # We use the raw name from your Feature Group
    target_col = 'pm2_5' 
    
    # Time-based features
    data['datetime'] = pd.to_datetime(data['time'], unit='ms')
    data['hour'] = data['datetime'].dt.hour
    
    # Creating the Lags (The Intelligence)
    data['pm2_5_lag_1h'] = data[target_col].shift(1)
    data['pm2_5_lag_24h'] = data[target_col].shift(24)
    
    data = data.dropna()
    
    # Drop columns model shouldn't see
    features = data.drop(columns=[target_col, 'time', 'datetime', 'city', 'pm10'], errors='ignore')
    target = data[target_col]
    
    return features, target

X, y = prepare_data(df)

# 4. Chronological Split (Better for time-series than random split)
# We train on the past and test on the most recent data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 5. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Check Performance
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model Training Complete! Mean Squared Error: {mse:.2f}")

# 7. Save and Register the Model
model_dir = "aqi_model"
if not os.path.exists(model_dir): os.mkdir(model_dir)
joblib.dump(model, f"{model_dir}/model.pkl")

mr = project.get_model_registry()
karachi_model = mr.python.create_model(
    name="karachi_aqi_model",
    metrics={"mse": mse},
    description="Predicts PM2.5 using time features and 1h/24h lags."
)
karachi_model.save(model_dir)
print(" Model successfully uploaded to the Hopsworks Model Registry")