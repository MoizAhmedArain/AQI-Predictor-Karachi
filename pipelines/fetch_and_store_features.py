import requests
import pandas as pd
from datetime import datetime
import hopsworks
import os

# CONFIG
CITY = "karachi"
AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")

if AQICN_TOKEN is None:
    raise RuntimeError("AQICN_API_TOKEN environment variable not set")

#"""URL = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"""

# FETCH DATA FROM API
response = requests.get(AQICN_TOKEN)
data = response.json()

if data["status"] != "ok":
    raise RuntimeError("Failed to fetch AQI data")

aqi_value = data["data"]["aqi"]
timestamp = datetime.utcnow()


# FEATURE ENGINEERING
features = {
    "timestamp": [timestamp],
    "aqi": [aqi_value],
    "hour": [timestamp.hour],
    "day": [timestamp.day],
    "month": [timestamp.month]
}

df = pd.DataFrame(features)

# CONNECT TO HOPSWORKS
project = hopsworks.login(
    project="Project1_AQI_predict"
)
fs = project.get_feature_store()

# CREATE / GET FEATURE GROUP
feature_group = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    primary_key=["timestamp"],
    description="Hourly AQI features for Karachi"
)


# INSERT FEATURES
feature_group.insert(df)

print("Features successfully written to Hopsworks in Feature Store")
