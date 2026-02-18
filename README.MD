### Karachi AQI Predictor ###

# Project Goal
The goal of this project is to provide a reliable 7 hour forecast of PM2.5 air quality levels in Karachi, Pakistan. By integrating real time weather data with historical air quality observations, the system uses machine learning to help residents and policymakers anticipate hazardous air quality conditions.

# System Architecture
This project follows a modern MLOps design, utilizing a "Feature Store" architecture to decouple data engineering from model inference.

Feature Pipeline: Automated script that fetches observed weather and AQI data from the Open-Meteo API and updates the Hopsworks Feature Store.

Model Registry: Stores a trained Scikit-Learn model and its corresponding Scaler (v2).

Batch Inference Pipeline: Triggered hourly via GitHub Actions. It retrieves the latest features, calculates rolling lags (1h and 24h), generates a 72-hour forecast using a sliding window approach, and stores predictions back in Hopsworks.

UI/Dashboard: A Streamlit application that visualizes the predicted PM2.5 trends and provides a data-driven outlook for the city.

# Tech Stack
Data Source: Open-Meteo API (Weather & Air Quality)

Feature Store: Hopsworks

Automation: GitHub Actions (Cron scheduling)

Machine Learning: Python, Scikit-Learn, Pandas

Frontend: Streamlit Cloud

#  How to Run
1. Setup Environment
Ensure you have the required dependencies installed:

Bash
pip install -r requirements.txt
2. Configure Secrets
Create a .env file (or set GitHub Secrets) with the following:

HOPSWORKS_API_KEY: Your project API key.

HOPSWORKS_PROJECT_NAME: Your project name (e.g., Karachi_AQI).

3. Run Pipelines
Update Features: python feature_pipeline.py

Generate Forecasts: python batch_inference.py

Launch UI: streamlit run streamlit_app.py

#  Key Features
Sliding Window Inference: The model uses its own predictions as lags to generate a continuous 72-hour forecast.

Robust Data Access: Bypasses metadata overhead by accessing Feature Groups directly for high-performance reading.

Automated Lifecycle: From data ingestion to prediction, the entire process is hands-off via GitHub Actions.

#  Disclaimer
This project is for educational purposes. AQI predictions are based on historical trends and weather forecasts provided by Open-Me
