import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

load_dotenv()

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

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

df = feature_view.get_batch_data()

aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)
df = aqi_fg.read()

print("--- COLUMNS FOUND ---")
print(df.columns.tolist())

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

split_idx = int(len(X) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- 2. Scaling (Vital for Linear Regression) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_mse = float('inf')
best_model_name = ""

print("\n Results for better model selection:")

print("DEBUG: Exact column order for model training:")
print(X.columns.tolist())

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
    r2 = r2_score(y_test, preds)
    
    results[name] = {"RMSE": rmse, "R2": r2}
    
    print(f"--- {name} ---")
    print(f"   RMSE: {rmse:.2f} (Average error in PM2.5 units)")
    print(f"   R2 Score: {r2:.2f}")

    if rmse < best_mse: # best_mse is our threshold variable
            best_mse = rmse
            best_model = model
            best_model_name = name

print(f"\n The Winner is: {best_model_name} with RMSE {best_mse:.2f}")

# --- 4. Save and Register the Winner ---
model_dir = "aqi_model"
if not os.path.exists(model_dir): os.mkdir(model_dir)

#Save the model AND the scaler (we'll need the scaler for predictions tomorrow!)
joblib.dump(best_model, f"{model_dir}/model.pkl")
joblib.dump(scaler, f"{model_dir}/scaler.pkl")

mr = project.get_model_registry()
karachi_model = mr.python.create_model(
    name="karachi_aqi_model",
    metrics={"mse": best_mse},
    description=f"Best model ({best_model_name}) found during Day 4 tournament."
)
karachi_model.save(model_dir)

print(f" {best_model_name} successfully uploaded to Hopsworks!")
