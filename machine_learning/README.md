# Machine Learning Module

Folder ini berisi seluruh pipeline Machine Learning yang digunakan untuk
memprediksi suhu udara (temperature) berdasarkan data cuaca historis.
Model dikembangkan menggunakan algoritma Random Forest Regressor.

---

## Struktur Folder

machine_learning/
├── training/
│   └── train_model.py        # Script training model
├── inference/
│   └── predict.py            # Script inference / prediksi suhu
├── models/
│   ├── model.pkl             # Model Random Forest terlatih
│   ├── scaler.pkl            # StandardScaler
│   └── selector.pkl          # Feature selector (SelectKBest)
├── notebooks/
│   └── training.ipynb        # Notebook eksperimen & EDA
├── requirements.txt
└── README.md

---

## Dataset

Model dilatih menggunakan data cuaca historis:

- Weather_Data_1980-2024.csv
- Weather_dataset.csv

Dataset tidak disertakan di repository GitHub karena ukuran besar.
Data digunakan hanya pada tahap training dan eksperimen.

---

## Target Prediksi

- **Target variable:** `temperature`
- **Satuan:** derajat Celcius (°C)

---

## Fitur yang Digunakan

Model menggunakan fitur cuaca berikut:

- relative_humidity  
- dew_point  
- rain (mm)  
- snowfall (cm)  
- pressure_msl (hPa)  
- surface_pressure (hPa)  
- cloud_cover (%)  
- cloud_cover_low (%)  
- cloud_cover_mid (%)  
- cloud_cover_high (%)  
- vapour_pressure_deficit (kPa)  
- wind_speed_10m (km/h)  
- wind_direction  
- is_Day  

---

## Metode & Preprocessing

Tahapan Machine Learning yang digunakan:

- Scaling: **StandardScaler**
- Feature Selection: **SelectKBest (f_regression)**
- Algoritma: **Random Forest Regressor**
- Hyperparameter tuning: **RandomizedSearchCV**
- Evaluasi: RMSE, MAE, dan R² Score

Model akhir menggunakan konfigurasi yang lebih kecil agar ringan saat deployment.

---

## Training Model

Untuk melakukan training ulang model:

1. Pastikan virtual environment aktif
2. Pastika
