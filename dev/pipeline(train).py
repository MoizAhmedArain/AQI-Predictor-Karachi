import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
import logging
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_data(data):
    data = data.sort_values('time').reset_index(drop=True)

    target_col = 'pm2_5'

    data['datetime'] = pd.to_datetime(data['time'], unit='ms')
    data['hour'] = data['datetime'].dt.hour

    data['pm2_5_lag_1h'] = data[target_col].shift(1)
    data['pm2_5_lag_24h'] = data[target_col].shift(24)

    data = data.dropna()

    features = data.drop(
        columns=[target_col, 'time', 'datetime', 'city', 'pm10'],
        errors='ignore'
    )
    target = data[target_col]

    return features, target


def main():
    try:
        logging.info("Loading environment variables...")
        load_dotenv()

        logging.info("Connecting to Hopsworks...")
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )

        fs = project.get_feature_store()

        logging.info("Loading Feature Group...")
        aqi_fg = fs.get_feature_group(
            name="karachi_aqi_weather",
            version=1
        )

        logging.info("Reading data...")
        df = aqi_fg.read()

        logging.info("Preparing data...")
        X, y = prepare_data(df)

        split_idx = int(len(X) * 0.8)
        X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logging.info("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        logging.info("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        logging.info(f"Model Performance -> RMSE: {rmse:.2f}, R2: {r2:.2f}")

        logging.info("Saving model artifacts...")
        model_dir = "aqi_model"
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, f"{model_dir}/model.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler.pkl")

        logging.info("Registering model to Hopsworks...")
        mr = project.get_model_registry()

        karachi_model = mr.python.create_model(
            name="karachi_aqi_model",
            metrics={"rmse": rmse},
            description="Random Forest AQI prediction model (production pipeline)"
        )

        karachi_model.save(model_dir)

        logging.info("Model successfully uploaded to Hopsworks")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
