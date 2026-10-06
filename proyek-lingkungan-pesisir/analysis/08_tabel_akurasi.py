"""
08_tabel_akurasi.py
Menyusun ulang hasil_evaluasi_model.csv (dari 05_modeling_dan_evaluasi.py)
jadi satu tabel rapi: Wilayah | Model | RMSE | MAE | MAPE | Akurasi (%)

Akurasi dihitung sebagai (100 - MAPE), pendekatan sederhana untuk
menerjemahkan error persentase jadi angka yang mudah dibaca awam -
BUKAN metrik akurasi klasifikasi, karena ini tetap kasus regresi.
"""

import pandas as pd

# ========== BACA HASIL EVALUASI ==========

df = pd.read_csv("hasil_evaluasi_model.csv")

# ========== SUSUN ULANG KE FORMAT PANJANG (long format) ==========

daftar_baris = []

for _, baris in df.iterrows():
    wilayah = baris["wilayah"]

    daftar_baris.append({
        "wilayah": wilayah, "model": "Naive",
        "rmse": baris["rmse_naive"], "mae": baris["mae_naive"], "mape": baris["mape_naive"],
    })
    daftar_baris.append({
        "wilayah": wilayah, "model": "Linear Regression",
        "rmse": baris["rmse_lr"], "mae": baris["mae_lr"], "mape": baris["mape_lr"],
    })
    daftar_baris.append({
        "wilayah": wilayah, "model": "XGBoost",
        "rmse": baris["rmse_xgb"], "mae": baris["mae_xgb"], "mape": baris["mape_xgb"],
    })

df_tabel = pd.DataFrame(daftar_baris)
df_tabel["akurasi (%)"] = 100 - df_tabel["mape"]

# ========== TANDAI MODEL TERBAIK PER WILAYAH (RMSE terendah) ==========

df_tabel["terbaik"] = ""
for wilayah in df_tabel["wilayah"].unique():
    idx_terbaik = df_tabel[df_tabel["wilayah"] == wilayah]["rmse"].idxmin()
    df_tabel.loc[idx_terbaik, "terbaik"] = "<-- terbaik (RMSE)"

# ========== CETAK RAPI ==========

pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 90)
print("TABEL RINGKASAN AKURASI MODEL - PREDIKSI ANOMALI SST (h+24 jam)")
print("=" * 90)

for wilayah in df_tabel["wilayah"].unique():
    print(f"\n--- {wilayah} ---")
    subset = df_tabel[df_tabel["wilayah"] == wilayah][
        ["model", "rmse", "mae", "mape", "akurasi (%)", "terbaik"]
    ]
    print(subset.to_string(index=False))

print("\n" + "=" * 90)
print("RATA-RATA AKURASI SEMUA WILAYAH PER MODEL")
print("=" * 90)
rata_rata = df_tabel.groupby("model")["akurasi (%)"].mean().sort_values(ascending=False)
print(rata_rata.to_string())

# ========== SIMPAN ==========

df_tabel.to_csv("tabel_akurasi_lengkap.csv", index=False)
print("\n[OK] Tabel lengkap disimpan ke tabel_akurasi_lengkap.csv")