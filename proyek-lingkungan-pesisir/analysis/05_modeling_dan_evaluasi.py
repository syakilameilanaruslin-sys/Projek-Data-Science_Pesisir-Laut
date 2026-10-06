"""
05_modeling_dan_evaluasi.py
Tahap 6: Modeling + Evaluasi untuk menjawab Pertanyaan Model/AI:
"Bagaimana prediksi kondisi lingkungan pesisir (anomali SST) untuk
 beberapa hari ke depan menggunakan model machine learning?"

Model dibangun TERPISAH per wilayah (karena karakteristik tiap wilayah beda,
seperti terbukti di analisis korelasi sebelumnya).

Perbandingan 3 model:
1. Baseline Naive (prediksi = nilai 24 jam sebelumnya) - patokan minimum
2. Linear Regression - baseline statistik
3. XGBoost - model utama yang diharapkan lebih baik

Menggunakan data_siap_model.csv hasil dari 04_feature_engineering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ========== KONFIGURASI ==========

RASIO_TEST = 0.15   # 15% data terakhir (kronologis) dipakai untuk testing

KOLOM_FITUR = [
    "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "jam", "hari_dalam_minggu", "bulan",
]

# ========== BACA DATA (sudah siap, tinggal pakai) ==========

df_siap_model = pd.read_csv("data_siap_model.csv")
df_siap_model["waktu"] = pd.to_datetime(df_siap_model["waktu"])

print(f"Data dimuat: {len(df_siap_model)} baris, {df_siap_model['wilayah'].nunique()} wilayah")


# ========== FUNGSI EVALUASI ==========

def hitung_metrik(y_asli, y_prediksi):
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mae = mean_absolute_error(y_asli, y_prediksi)
    mask = y_asli != 0  # hindari pembagian dengan nol untuk MAPE
    mape = np.mean(np.abs((y_asli[mask] - y_prediksi[mask]) / y_asli[mask])) * 100
    return rmse, mae, mape


# ========== MODELING & EVALUASI PER WILAYAH ==========

hasil_semua_wilayah = []

for wilayah in df_siap_model["wilayah"].unique():
    print(f"\n{'='*60}")
    print(f"WILAYAH: {wilayah}")
    print(f"{'='*60}")

    df_fitur = df_siap_model[df_siap_model["wilayah"] == wilayah].sort_values("waktu").reset_index(drop=True)

    # Split time-series (bukan random split!) - data awal untuk train, data akhir untuk test
    n = len(df_fitur)
    batas_test = int(n * (1 - RASIO_TEST))

    train = df_fitur.iloc[:batas_test]
    test = df_fitur.iloc[batas_test:]

    X_train, y_train = train[KOLOM_FITUR], train["target"]
    X_test, y_test = test[KOLOM_FITUR], test["target"]

    print(f"Data training: {len(train)} baris | Data testing: {len(test)} baris")

    # --- Model 1: Baseline Naive ---
    prediksi_naive = test["lag_1"].values
    rmse_naive, mae_naive, mape_naive = hitung_metrik(y_test.values, prediksi_naive)

    # --- Model 2: Linear Regression ---
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    prediksi_lr = model_lr.predict(X_test)
    rmse_lr, mae_lr, mape_lr = hitung_metrik(y_test.values, prediksi_lr)

    # --- Model 3: XGBoost (dengan pencarian parameter otomatis) ---
    # Target sangat persisten/autokorelatif, jadi kita cari titik kompleksitas
    # yang pas: cukup sederhana untuk tidak overfit, tapi tidak terlalu
    # sederhana sampai underfit. Dicari otomatis lewat validasi internal
    # (bukan random search, supaya proses ini konsisten & bisa dijelaskan
    # metodologinya di laporan).

    # Split train jadi train-internal + validasi (masih time-series, urut)
    batas_val = int(len(train) * 0.85)
    X_tr, y_tr = X_train.iloc[:batas_val], y_train.iloc[:batas_val]
    X_val, y_val = X_train.iloc[batas_val:], y_train.iloc[batas_val:]

    kandidat_parameter = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "reg_alpha": 0, "reg_lambda": 1},
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1},
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "reg_alpha": 0.5, "reg_lambda": 1.5},
    ]

    param_terbaik, rmse_val_terbaik = None, np.inf
    for param in kandidat_parameter:
        model_coba = XGBRegressor(**param, random_state=42, objective="reg:squarederror")
        model_coba.fit(X_tr, y_tr)
        prediksi_val = model_coba.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, prediksi_val))
        if rmse_val < rmse_val_terbaik:
            rmse_val_terbaik = rmse_val
            param_terbaik = param

    print(f"Parameter XGBoost terbaik (dari validasi): {param_terbaik} (RMSE validasi: {rmse_val_terbaik:.4f})")

    # Latih ulang model final dengan parameter terbaik, pakai SELURUH data train
    model_xgb = XGBRegressor(**param_terbaik, random_state=42, objective="reg:squarederror")
    model_xgb.fit(X_train, y_train)
    prediksi_xgb = model_xgb.predict(X_test)
    rmse_xgb, mae_xgb, mape_xgb = hitung_metrik(y_test.values, prediksi_xgb)

    # --- Cetak ringkasan perbandingan ---
    print(f"\n{'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE (%)':>10}")
    print(f"{'Baseline Naive':<20} {rmse_naive:>10.4f} {mae_naive:>10.4f} {mape_naive:>10.2f}")
    print(f"{'Linear Regression':<20} {rmse_lr:>10.4f} {mae_lr:>10.4f} {mape_lr:>10.2f}")
    print(f"{'XGBoost':<20} {rmse_xgb:>10.4f} {mae_xgb:>10.4f} {mape_xgb:>10.2f}")

    hasil_semua_wilayah.append({
        "wilayah": wilayah,
        "rmse_naive": rmse_naive, "mae_naive": mae_naive, "mape_naive": mape_naive,
        "rmse_lr": rmse_lr, "mae_lr": mae_lr, "mape_lr": mape_lr,
        "rmse_xgb": rmse_xgb, "mae_xgb": mae_xgb, "mape_xgb": mape_xgb,
    })

    # --- Plot: prediksi vs aktual (pakai model terbaik, XGBoost) ---
    plt.figure(figsize=(12, 5))
    plt.plot(test["waktu"].values, y_test.values, label="Aktual", linewidth=1.2)
    plt.plot(test["waktu"].values, prediksi_xgb, label="Prediksi (XGBoost)", linewidth=1.2, linestyle="--")
    plt.title(f"Prediksi vs Aktual - {wilayah}")
    plt.xlabel("Waktu")
    plt.ylabel("Anomali SST")
    plt.legend()
    plt.tight_layout()
    nama_file = f"prediksi_{wilayah.replace(' ', '_')}.png"
    plt.savefig(nama_file, dpi=150)
    plt.close()
    print(f"[OK] Grafik prediksi disimpan: {nama_file}")

# ========== SIMPAN RINGKASAN HASIL ==========

df_hasil = pd.DataFrame(hasil_semua_wilayah)
df_hasil.to_csv("hasil_evaluasi_model.csv", index=False)

print(f"\n{'='*60}")
print("=== RINGKASAN PERBANDINGAN SEMUA WILAYAH ===")
print(f"{'='*60}")
print(df_hasil.to_string(index=False))

print("\n[OK] Ringkasan lengkap disimpan ke hasil_evaluasi_model.csv")
print("\n=== SELESAI ===")
print("Model terbaik (RMSE terendah) yang dipilih untuk tiap wilayah bisa dilihat dari tabel di atas.")