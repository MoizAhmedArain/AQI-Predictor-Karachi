import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )

fs = project.get_feature_store()

# This is the reliable way to see what's inside
print("--- Checking Feature Groups ---")
# If get_feature_groups() fails, we can check specific names
names_to_check = ["karachi_aqi_weather", "karachi_aqi_view"]

for name in names_to_check:
    try:
        fg = fs.get_feature_group(name=name, version=1)
        print(f"✅ Found Group: {name}")
    except:
        print(f"❌ Could not find Group: {name}")

print("\n--- Checking Feature Views ---")
try:
    # Most reliable way to list views
    views = fs.get_feature_views()
    for fv in views:
        print(f"VIEW: {fv.name} (Version: {fv.version})")
except Exception as e:
    print(f"Error listing views: {e}")