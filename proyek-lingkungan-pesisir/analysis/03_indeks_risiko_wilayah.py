"""
03_indeks_risiko_wilayah.py
Tahap 4: Menjawab Pertanyaan Analisis 2:
"Wilayah mana yang memiliki tingkat risiko lingkungan pesisir tertinggi
 berdasarkan kombinasi ketiga variabel (SST, gelombang, kualitas udara)?"

Metode: normalisasi tiap variabel (0-1) lalu digabung jadi satu skor indeks risiko.
Menggunakan data_gabungan.csv hasil dari 01_join_dan_eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# ========== BACA DATA ==========

df = pd.read_csv("data_gabungan.csv")
df["waktu"] = pd.to_datetime(df["waktu"])

# ========== HITUNG ANOMALI SST (sama seperti tahap analisis korelasi) ==========

df["sst_anomali"] = df.groupby("wilayah")["sea_surface_temperature"].transform(
    lambda x: x - x.mean()
)
df["sst_anomali_abs"] = df["sst_anomali"].abs()  # yang penting besarnya penyimpangan, bukan arahnya

# ========== NORMALISASI VARIABEL (Min-Max Scaling 0-1) ==========
# Supaya variabel dengan satuan/skala berbeda (SST dalam °C, gelombang dalam m,
# PM dalam µg/m³) bisa digabung secara adil jadi satu skor.

def normalisasi(kolom):
    return (kolom - kolom.min()) / (kolom.max() - kolom.min())

df["sst_anomali_norm"] = normalisasi(df["sst_anomali_abs"])
df["gelombang_norm"] = normalisasi(df["wave_height"])
df["pm25_norm"] = normalisasi(df["pm2_5"])
df["pm10_norm"] = normalisasi(df["pm10"])

# ========== HITUNG INDEKS RISIKO GABUNGAN ==========
# Bobot sama rata (masing-masing 1/4) - bisa disesuaikan kalau ada justifikasi
# teoritis untuk memberi bobot berbeda pada variabel tertentu.

df["indeks_risiko"] = (
    df["sst_anomali_norm"] * 0.25 +
    df["gelombang_norm"] * 0.25 +
    df["pm25_norm"] * 0.25 +
    df["pm10_norm"] * 0.25
)

# ========== RANKING WILAYAH BERDASARKAN RATA-RATA INDEKS RISIKO ==========

ranking = df.groupby("wilayah")["indeks_risiko"].agg(["mean", "std", "max"]).sort_values("mean", ascending=False)
ranking.columns = ["rata_rata_risiko", "std_risiko", "risiko_maksimum"]

print("=== RANKING WILAYAH BERDASARKAN INDEKS RISIKO LINGKUNGAN PESISIR ===\n")
print(ranking)

print("\n=== INTERPRETASI ===")
for i, (wilayah, baris) in enumerate(ranking.iterrows(), start=1):
    print(f"Peringkat {i}: {wilayah} (skor rata-rata: {baris['rata_rata_risiko']:.4f})")

# ========== KATEGORISASI STATUS RISIKO (opsional, untuk mempermudah interpretasi) ==========

def kategori_risiko(skor):
    if skor < 0.25:
        return "Rendah"
    elif skor < 0.5:
        return "Sedang"
    elif skor < 0.75:
        return "Tinggi"
    else:
        return "Sangat Tinggi"

df["status_risiko"] = df["indeks_risiko"].apply(kategori_risiko)

print("\n=== DISTRIBUSI STATUS RISIKO PER WILAYAH ===")
distribusi = df.groupby(["wilayah", "status_risiko"]).size().unstack(fill_value=0)
print(distribusi)

# ========== VISUALISASI 1: Bar chart ranking rata-rata risiko ==========

plt.figure(figsize=(8, 5))
warna = ["#d62728" if i == 0 else "#ff7f0e" if i == 1 else "#2ca02c" for i in range(len(ranking))]
plt.bar(ranking.index, ranking["rata_rata_risiko"], color=warna)
plt.title("Ranking Indeks Risiko Lingkungan Pesisir per Wilayah")
plt.ylabel("Rata-rata Indeks Risiko (0-1)")
plt.xlabel("Wilayah")
for i, v in enumerate(ranking["rata_rata_risiko"]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("ranking_indeks_risiko.png", dpi=150)
plt.close()
print("\n[OK] Grafik disimpan: ranking_indeks_risiko.png")

# ========== VISUALISASI 2: Tren indeks risiko dari waktu ke waktu ==========

plt.figure(figsize=(12, 5))
for wilayah in df["wilayah"].unique():
    subset = df[df["wilayah"] == wilayah]
    # Rolling average 24 jam (1 hari) supaya tren lebih mudah dibaca, tidak terlalu noisy
    subset_harian = subset.set_index("waktu")["indeks_risiko"].rolling("24h").mean()
    plt.plot(subset_harian.index, subset_harian.values, label=wilayah, linewidth=1)
plt.title("Tren Indeks Risiko Lingkungan Pesisir (Rolling 24 Jam)")
plt.xlabel("Waktu")
plt.ylabel("Indeks Risiko (0-1)")
plt.legend()
plt.tight_layout()
plt.savefig("tren_indeks_risiko.png", dpi=150)
plt.close()
print("[OK] Grafik disimpan: tren_indeks_risiko.png")

# ========== SIMPAN HASIL ==========

df.to_csv("data_dengan_indeks_risiko.csv", index=False)
ranking.to_csv("ranking_wilayah.csv")

print("\n=== SELESAI ===")
print("File tersimpan: data_dengan_indeks_risiko.csv, ranking_wilayah.csv")
print("Grafik tersimpan: ranking_indeks_risiko.png, tren_indeks_risiko.png")