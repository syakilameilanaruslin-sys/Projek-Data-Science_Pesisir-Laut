"""
04_feature_engineering.py
Tahap 5: Feature Engineering - menyiapkan fitur untuk modeling.

Membuat lag features, rolling average, dan komponen waktu dari
data_dengan_indeks_risiko.csv, lalu simpan hasilnya ke CSV baru
supaya bisa langsung dipakai di tahap modeling (05_modeling_dan_evaluasi.py)
tanpa perlu hitung ulang tiap kali eksperimen model.
"""

import pandas as pd

# ========== KONFIGURASI ==========

TARGET_KOLOM = "sst_anomali"   # variabel yang mau diprediksi
JAM_KE_DEPAN = 24               # prediksi 24 jam (1 hari) ke depan

KOLOM_FITUR = [
    "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "jam", "hari_dalam_minggu", "bulan",
]

# ========== BACA DATA ==========

df = pd.read_csv("data_dengan_indeks_risiko.csv")
df["waktu"] = pd.to_datetime(df["waktu"])
df = df.sort_values(["wilayah", "waktu"]).reset_index(drop=True)

print(f"Data awal: {len(df)} baris, {df['wilayah'].nunique()} wilayah")


# ========== FUNGSI FEATURE ENGINEERING ==========

def buat_fitur(df_wilayah, target_kolom):
    df_fitur = df_wilayah.copy()

    # Lag features - nilai beberapa jam sebelumnya (prediktor utama untuk time series)
    for lag in [1, 3, 6, 12, 24]:
        df_fitur[f"lag_{lag}"] = df_fitur[target_kolom].shift(lag)

    # Rolling average & std - menangkap tren jangka pendek & menengah
    df_fitur["roll_mean_24"] = df_fitur[target_kolom].rolling(24).mean()
    df_fitur["roll_mean_168"] = df_fitur[target_kolom].rolling(168).mean()  # 7 hari
    df_fitur["roll_std_24"] = df_fitur[target_kolom].rolling(24).std()

    # Komponen waktu - menangkap pola siklus harian/mingguan/musiman
    df_fitur["jam"] = df_fitur["waktu"].dt.hour
    df_fitur["hari_dalam_minggu"] = df_fitur["waktu"].dt.dayofweek
    df_fitur["bulan"] = df_fitur["waktu"].dt.month

    # Target: nilai JAM_KE_DEPAN jam ke depan (yang mau diprediksi model)
    df_fitur["target"] = df_fitur[target_kolom].shift(-JAM_KE_DEPAN)

    # Buang baris yang punya NaN (akibat lag/rolling di awal & target shift di akhir)
    jumlah_sebelum = len(df_fitur)
    df_fitur = df_fitur.dropna().reset_index(drop=True)
    jumlah_sesudah = len(df_fitur)

    print(f"  {df_wilayah['wilayah'].iloc[0]}: {jumlah_sebelum} -> {jumlah_sesudah} baris "
          f"(terbuang {jumlah_sebelum - jumlah_sesudah} akibat lag/rolling/target shift)")

    return df_fitur


# ========== JALANKAN PER WILAYAH, GABUNG HASILNYA ==========

print(f"\nMembuat fitur untuk target: '{TARGET_KOLOM}', prediksi h+{JAM_KE_DEPAN} jam\n")

hasil_semua_wilayah = []
for wilayah in df["wilayah"].unique():
    df_wilayah = df[df["wilayah"] == wilayah].reset_index(drop=True)
    df_fitur = buat_fitur(df_wilayah, TARGET_KOLOM)
    hasil_semua_wilayah.append(df_fitur)

df_siap_model = pd.concat(hasil_semua_wilayah, ignore_index=True)

# ========== VALIDASI HASIL ==========

print(f"\nTotal data siap model: {len(df_siap_model)} baris")
print(f"\nCek missing value pada kolom fitur:")
print(df_siap_model[KOLOM_FITUR + ["target"]].isnull().sum())

print(f"\nContoh 5 baris pertama (kolom fitur utama):")
print(df_siap_model[["waktu", "wilayah", TARGET_KOLOM] + KOLOM_FITUR[:5] + ["target"]].head())

# ========== SIMPAN ==========

df_siap_model.to_csv("data_siap_model.csv", index=False)
print(f"\n[OK] Data siap model disimpan ke data_siap_model.csv")
print(f"Kolom fitur yang tersedia: {KOLOM_FITUR}")
print("\n=== SELESAI ===")
print("Lanjut ke 05_modeling_dan_evaluasi.py untuk tahap modeling")