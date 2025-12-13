import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor


def load_and_merge_data(data_dir):
    file1 = os.path.join(data_dir, "Weather_Data_1980-2024.csv")
    file2 = os.path.join(data_dir, "Weather_dataset.csv")

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Feature engineering is_Day
    df2["time"] = pd.to_datetime(df2["time"])
    df2["hour"] = df2["time"].dt.hour
    df2["is_Day"] = df2["hour"].apply(lambda h: 1 if 6 <= h <= 17 else 0)
    df2 = df2.drop(columns=["hour"])

    # Samakan kolom
    df2 = df2[df1.columns]

    # Gabungkan
    df = pd.concat([df1, df2], ignore_index=True)

    # Cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    # Sorting
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    return df


def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")
    MODEL_DIR = os.path.join(BASE_DIR, "../models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================
    # Load & preprocess data
    # =========================
    df = load_and_merge_data(DATA_DIR)

    target = "temperature"
    drop_cols = ["time", target]

    X = df.drop(columns=drop_cols)
    y = df[target]

    # =========================
    # Split data
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # Feature selection
    # =========================
    selector = SelectKBest(score_func=f_regression, k=10)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    # =========================
    # Final model (deploy)
    # =========================
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_sel, y_train)

    # =========================
    # Save artifacts
    # =========================
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"), compress=3)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"), compress=3)
    joblib.dump(selector, os.path.join(MODEL_DIR, "selector.pkl"), compress=3)

    print("✅ Training selesai. Model, scaler, dan selector disimpan.")


if __name__ == "__main__":
    train()
