import os
import joblib
import pandas as pd

# =============================
# Load model artifacts
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))

# =============================
# Feature order (HARUS SAMA)
# =============================
FEATURE_COLUMNS = [
    'relative_humidity',
    'dew_point',
    'rain (mm)',
    'snowfall (cm)',
    'pressure_msl (hPa)',
    'surface_pressure (hPa)',
    'cloud_cover (%)',
    'cloud_cover_low (%)',
    'cloud_cover_mid (%)',
    'cloud_cover_high (%)',
    'vapour_pressure_deficit (kPa)',
    'wind_speed_10m (km/h)',
    'wind_direction',
    'is_Day'
]

# =============================
# Prediction function
# =============================
def predict(input_data: dict) -> float:

    df = pd.DataFrame([input_data])

    # Tambahkan kolom yang hilang
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Urutkan sesuai scaler
    df = df[scaler.feature_names_in_]

    # Scaling
    df_scaled = scaler.transform(df)

    # Feature selection
    df_selected = selector.transform(df_scaled)

    # Predict
    prediction = model.predict(df_selected)

    return f"{round(float(prediction[0]), 2)} °C"


# =============================
# Local testing
# =============================
if __name__ == "__main__":
    sample_input = {
        'relative_humidity': 80,
        'dew_point': 24,
        'rain (mm)': 2.5,
        'snowfall (cm)': 0,
        'pressure_msl (hPa)': 1010,
        'surface_pressure (hPa)': 1008,
        'cloud_cover (%)': 75,
        'cloud_cover_low (%)': 40,
        'cloud_cover_mid (%)': 20,
        'cloud_cover_high (%)': 10,
        'vapour_pressure_deficit (kPa)': 0.5,
        'wind_speed_10m (km/h)': 12,
        'wind_direction': 180,
        'is_Day': 1
    }

    print("Prediksi suhu:", predict(sample_input))
