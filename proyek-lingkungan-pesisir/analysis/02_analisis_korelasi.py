"""
02_analisis_korelasi.py
Tahap 3: Analisis korelasi antar variabel untuk menjawab Pertanyaan Analisis 1:
"Apakah ada korelasi signifikan antara anomali SST/gelombang tinggi
 dengan penurunan kualitas udara di wilayah pesisir?"

Menggunakan data_gabungan.csv hasil dari 01_join_dan_eda.py
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ========== BACA DATA ==========

df = pd.read_csv("data_gabungan.csv")
df["waktu"] = pd.to_datetime(df["waktu"])

print(f"Total baris data: {len(df)}")
print(f"Wilayah: {df['wilayah'].unique()}")

# ========== HITUNG ANOMALI SST ==========
# Anomali = selisih nilai aktual dengan rata-rata historisnya per wilayah
# Ini penting karena yang dianalisis bukan SST mentah, tapi PENYIMPANGANnya

df["sst_anomali"] = df.groupby("wilayah")["sea_surface_temperature"].transform(
    lambda x: x - x.mean()
)

# ========== KORELASI PEARSON & SPEARMAN (per wilayah) ==========

def hitung_korelasi(df, kolom_a, kolom_b, nama_a, nama_b):
    print(f"\n=== Korelasi {nama_a} vs {nama_b} ===")
    hasil = []
    for wilayah in df["wilayah"].unique():
        subset = df[df["wilayah"] == wilayah]
        r_pearson, p_pearson = stats.pearsonr(subset[kolom_a], subset[kolom_b])
        r_spearman, p_spearman = stats.spearmanr(subset[kolom_a], subset[kolom_b])

        signifikan = "Signifikan (p<0.05)" if p_pearson < 0.05 else "Tidak signifikan"
        kekuatan = interpretasi_kekuatan(r_pearson)

        print(f"{wilayah:20s} | Pearson r={r_pearson:6.3f} (p={p_pearson:.4f}) | "
              f"Spearman r={r_spearman:6.3f} (p={p_spearman:.4f}) | {kekuatan} | {signifikan}")

        hasil.append({
            "wilayah": wilayah,
            "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman,
            "kekuatan": kekuatan, "signifikan": signifikan,
        })
    return pd.DataFrame(hasil)


def interpretasi_kekuatan(r):
    r_abs = abs(r)
    if r_abs < 0.1:
        return "Sangat lemah/tidak ada"
    elif r_abs < 0.3:
        return "Lemah"
    elif r_abs < 0.5:
        return "Sedang"
    elif r_abs < 0.7:
        return "Kuat"
    else:
        return "Sangat kuat"


# Korelasi 1: Anomali SST vs PM2.5
hasil_sst_pm25 = hitung_korelasi(df, "sst_anomali", "pm2_5", "Anomali SST", "PM2.5")

# Korelasi 2: Anomali SST vs PM10
hasil_sst_pm10 = hitung_korelasi(df, "sst_anomali", "pm10", "Anomali SST", "PM10")

# Korelasi 3: Tinggi gelombang vs PM2.5
hasil_gel_pm25 = hitung_korelasi(df, "wave_height", "pm2_5", "Tinggi Gelombang", "PM2.5")

# Korelasi 4: Tinggi gelombang vs PM10
hasil_gel_pm10 = hitung_korelasi(df, "wave_height", "pm10", "Tinggi Gelombang", "PM10")

# ========== HEATMAP KORELASI (semua variabel sekaligus, per wilayah) ==========

kolom_analisis = [
    "sea_surface_temperature", "wave_height",
    "precipitation", "windspeed_10m",
    "pm2_5", "pm10",
]

for wilayah in df["wilayah"].unique():
    subset = df[df["wilayah"] == wilayah][kolom_analisis]
    matriks_korelasi = subset.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriks_korelasi, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Matriks Korelasi - {wilayah}")
    plt.tight_layout()
    nama_file = f"korelasi_{wilayah.replace(' ', '_')}.png"
    plt.savefig(nama_file, dpi=150)
    plt.close()
    print(f"[OK] Heatmap disimpan: {nama_file}")

# ========== SIMPAN RINGKASAN HASIL KE CSV ==========

ringkasan_semua = pd.concat([
    hasil_sst_pm25.assign(pasangan="SST_anomali vs PM2.5"),
    hasil_sst_pm10.assign(pasangan="SST_anomali vs PM10"),
    hasil_gel_pm25.assign(pasangan="Gelombang vs PM2.5"),
    hasil_gel_pm10.assign(pasangan="Gelombang vs PM10"),
])
ringkasan_semua.to_csv("hasil_korelasi.csv", index=False)
print("\n[OK] Ringkasan hasil korelasi disimpan ke hasil_korelasi.csv")

print("\n=== SELESAI ===")
print("Cek file korelasi_<wilayah>.png untuk heatmap visual")
print("Cek hasil_korelasi.csv untuk tabel ringkasan lengkap")