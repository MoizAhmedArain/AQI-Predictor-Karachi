import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("HOPSWORKS_API_KEY"))
print(os.getenv("HOPSWORKS_PROJECT_NAME"))