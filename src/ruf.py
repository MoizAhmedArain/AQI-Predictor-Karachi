import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")    
)
fs = project.get_feature_store()

# 1. Get the Feature Group object
# Based on your screenshot, the name is 'karachi_aqi_weather'
aqi_fg = fs.get_feature_group(name="karachi_aqi_weather", version=1)

# 2. Use .read() with the 'python' engine explicitly
# This avoids the Hive error by using the modern Arrow-based reader
print("Reading data using the Python engine...")
hist_df = aqi_fg.read(read_options={"use_hive": False})
hist_df = hist_df.sort_values('time').reset_index(drop=True)

if not hist_df.empty:
    print(f"✅ SUCCESS! Retrieved {len(hist_df)} rows.")
    print(hist_df.head())
else:
    print("❌ Data retrieved but DataFrame is empty.")