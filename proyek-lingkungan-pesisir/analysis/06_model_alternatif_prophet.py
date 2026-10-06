"""
06_model_alternatif_prophet.py
Mencoba model alternatif (Prophet) yang secara eksplisit menangkap
TREN JANGKA PANJANG + MUSIMAN HARIAN pada data SST - dua pola yang
terlihat jelas di EDA awal tapi tidak ditangkap baik oleh pendekatan
lag-based (XGBoost) sebelumnya.

Berbeda dari 05_modeling_dan_evaluasi.py yang memprediksi "nilai h+24 jam
dari sekarang" per baris, di sini Prophet memprediksi deret waktu masa
depan secara langsung (multi-step forecasting) - framing yang lebih
natural untuk pertanyaan "prediksi beberapa hari ke depan".

Menggunakan data_dengan_indeks_risiko.csv (bukan data_siap_model.csv,
karena Prophet tidak butuh lag features manual).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========== KONFIGURASI ==========

TARGET_KOLOM = "sst_anomali"
RASIO_TEST = 0.15

# ========== BACA DATA ==========

df = pd.read_csv("data_dengan_indeks_risiko.csv")
df["waktu"] = pd.to_datetime(df["waktu"])
df = df.sort_values(["wilayah", "waktu"]).reset_index(drop=True)


def hitung_metrik(y_asli, y_prediksi):
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mae = mean_absolute_error(y_asli, y_prediksi)
    mask = y_asli != 0
    mape = np.mean(np.abs((y_asli[mask] - y_prediksi[mask]) / y_asli[mask])) * 100
    return rmse, mae, mape


hasil_semua_wilayah = []

for wilayah in df["wilayah"].unique():
    print(f"\n{'='*60}")
    print(f"WILAYAH: {wilayah} (Prophet)")
    print(f"{'='*60}")

    df_wilayah = df[df["wilayah"] == wilayah][["waktu", TARGET_KOLOM]].reset_index(drop=True)

    n = len(df_wilayah)
    batas_test = int(n * (1 - RASIO_TEST))
    train = df_wilayah.iloc[:batas_test]
    test = df_wilayah.iloc[batas_test:]

    # Prophet wajib nama kolom "ds" (waktu) dan "y" (nilai)
    df_prophet = train.rename(columns={"waktu": "ds", TARGET_KOLOM: "y"})

    # Model Prophet: nonaktifkan musiman harian (data SST hampir tidak berubah
    # dalam 1 hari, jadi musiman harian cuma menambah noise, bukan sinyal asli).
    # changepoint_prior_scale diturunkan jauh supaya tren tidak diekstrapolasi
    # terlalu agresif ke masa depan (mengurangi risiko overshoot).
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
    )
    model.fit(df_prophet)

    # Forecast sepanjang periode test
    df_masa_depan = test[["waktu"]].rename(columns={"waktu": "ds"})
    hasil_forecast = model.predict(df_masa_depan)

    prediksi_prophet = hasil_forecast["yhat"].values
    y_test = test[TARGET_KOLOM].values

    rmse_prophet, mae_prophet, mape_prophet = hitung_metrik(y_test, prediksi_prophet)

    # --- Baseline Naive untuk perbandingan yang FAIR dalam framing ini ---
    # (nilai terakhir sebelum periode test, dibawa terus sepanjang periode forecast -
    #  ini baseline yang jauh lebih "jujur"/susah dikalahkan dibanding lag_1 per jam)
    nilai_awal_test = train[TARGET_KOLOM].iloc[-1]
    prediksi_naive = np.full(len(test), nilai_awal_test)
    rmse_naive, mae_naive, mape_naive = hitung_metrik(y_test, prediksi_naive)

    print(f"\n{'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE (%)':>10}")
    print(f"{'Baseline Naive':<20} {rmse_naive:>10.4f} {mae_naive:>10.4f} {mape_naive:>10.2f}")
    print(f"{'Prophet':<20} {rmse_prophet:>10.4f} {mae_prophet:>10.4f} {mape_prophet:>10.2f}")

    menang = "Prophet MENANG" if rmse_prophet < rmse_naive else "Naive masih lebih baik"
    print(f"==> {menang}")

    hasil_semua_wilayah.append({
        "wilayah": wilayah,
        "rmse_naive": rmse_naive, "mae_naive": mae_naive, "mape_naive": mape_naive,
        "rmse_prophet": rmse_prophet, "mae_prophet": mae_prophet, "mape_prophet": mape_prophet,
        "kesimpulan": menang,
    })

    # --- Plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(test["waktu"].values, y_test, label="Aktual", linewidth=1.2)
    plt.plot(test["waktu"].values, prediksi_prophet, label="Prediksi (Prophet)", linewidth=1.2, linestyle="--")
    plt.axhline(nilai_awal_test, color="gray", linestyle=":", label="Baseline Naive (konstan)")
    plt.title(f"Prediksi Prophet vs Aktual vs Naive - {wilayah}")
    plt.xlabel("Waktu")
    plt.ylabel(TARGET_KOLOM)
    plt.legend()
    plt.tight_layout()
    nama_file = f"prediksi_prophet_{wilayah.replace(' ', '_')}.png"
    plt.savefig(nama_file, dpi=150)
    plt.close()
    print(f"[OK] Grafik disimpan: {nama_file}")

# ========== RINGKASAN ==========

df_hasil = pd.DataFrame(hasil_semua_wilayah)
df_hasil.to_csv("hasil_evaluasi_prophet.csv", index=False)

print(f"\n{'='*60}")
print("=== RINGKASAN PERBANDINGAN PROPHET vs NAIVE ===")
print(f"{'='*60}")
print(df_hasil.to_string(index=False))
print("\n[OK] Disimpan ke hasil_evaluasi_prophet.csv")