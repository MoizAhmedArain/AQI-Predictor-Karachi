import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import numpy as np
import joblib

# -------------------------
# 1. Login to Hopsworks
# -------------------------
project = hopsworks.login(
    project="Project1_AQI_predict"
)
fs = project.get_feature_store()

# -------------------------
# 2. Get Feature Group
# -------------------------
fg = fs.get_feature_group(
    name="karachi_aqi_features",
    version=1
)

# -------------------------
# 3. Load training data
# -------------------------
df = fg.read()

# -------------------------
# 4. Prepare X / y
# -------------------------
target = "aqi"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Train models
# -------------------------
models = {
    "ridge": Ridge(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    results[name] = {"rmse": rmse, "mae": mae}

    joblib.dump(model, f"models/{name}.pkl")

# -------------------------
# 6. Print results
# -------------------------
print("Model evaluation results:")
for model, metrics in results.items():
    print(model, metrics)
