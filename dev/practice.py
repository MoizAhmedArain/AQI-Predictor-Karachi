import hopsworks
import pandas as pd
import logging
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


# 1. Setup
project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()
mr = project.get_model_registry()

# 2. Get Metadata for Comparison
print("\n--- 🔍 DIAGNOSTIC REPORT ---")

# A. Check Feature View (V2)
fv = fs.get_feature_view(name="karachi_aqi_view", version=2)
fv_schema = [f.name for f in fv.schema]
print(f"1. Feature View Schema (Order in UI): {fv_schema}")

# B. Check Model Registry (V2)
model_meta = mr.get_model("karachi_aqi_model", version=2)
# Updated Diagnostic snippet
try:
    # Accessing the schema in Hopsworks 4.0
    model_schema = [f.name for f in model_meta.model_schema.input_schema]
    print(f"2. Model Input Schema: {model_schema}")
except Exception as e:
    print(f"2. Could not read model schema: {e}")
    # Fallback: Check what your code expects
    model_schema = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour', 'pm2_5_lag_1h', 'pm2_5_lag_24h']
    print(f"   Using sequence from your script: {model_schema}")

print("3. Fetching Sample Batch Data to check for Dictionary Error...")
test_df = fv.get_batch_data()
print(f"   Actual Column Names returned: {list(test_df.columns)}")

# C. Check the "Dictionary Problem" in the actual Data
print("3. Fetching Sample Batch Data...")
test_df = fv.get_batch_data()
actual_columns = list(test_df.columns)

print(f"   Actual Column Names returned by Python: {actual_columns}")

# --- ANALYSES ---
print("\n--- ⚖️ FINAL VERIFICATION ---")

# Problem 1: Dictionary Check
has_dict_problem = any("{" in str(c) for c in actual_columns)
print(f"[PROBLEM 1] Dictionary metadata in names? {'❌ YES' if has_dict_problem else '✅ NO'}")

# Problem 2: Sequence Check
# We ignore 'time' and 'city' for the model check
model_features_in_fv = [c for c in actual_columns if c in model_schema]
is_sequence_correct = model_features_in_fv == [c for c in model_schema if c in actual_columns]
print(f"[PROBLEM 2] Column sequence matches model? {'✅ YES' if is_sequence_correct else '❌ NO (Mismatch detected)'}")

# Problem 3: Missing Features
missing_from_fv = [c for c in model_schema if c not in actual_columns]
print(f"[PROBLEM 3] Features missing from Feature View: {missing_from_fv if missing_from_fv else '✅ None'}")

if missing_from_fv:
    print(f"   (Note: These must be calculated manually in your batch_inference loop)")