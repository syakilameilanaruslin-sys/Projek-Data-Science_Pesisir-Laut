"""
01_join_dan_eda.py
Tahap 1: Join 3 tabel dari MySQL jadi satu dataset
Tahap 2: EDA dasar (statistik + visualisasi tren) untuk menjawab
         pertanyaan deskriptif dari 4 pertanyaan penelitian.

Dijalankan manual (bukan lewat Airflow), taruh di folder analysis/
"""

import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt

# ========== KONFIGURASI KONEKSI ==========

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "lingkungan_pesisir",
}

# ========== TAHAP 1: AMBIL & JOIN 3 TABEL ==========

def ambil_dan_join():
    koneksi = mysql.connector.connect(**DB_CONFIG)

    df_marine = pd.read_sql("SELECT * FROM tabel_oseanografi", koneksi)
    df_cuaca = pd.read_sql("SELECT * FROM tabel_cuaca", koneksi)
    df_udara = pd.read_sql("SELECT * FROM tabel_kualitas_udara", koneksi)

    koneksi.close()

    print(f"Jumlah baris - Oseanografi: {len(df_marine)}, Cuaca: {len(df_cuaca)}, Udara: {len(df_udara)}")

    # Pastikan tipe datetime benar sebelum join
    for df in [df_marine, df_cuaca, df_udara]:
        df["waktu"] = pd.to_datetime(df["waktu"])

    # Join berdasarkan waktu + wilayah
    df_gabung = df_marine.merge(df_cuaca, on=["waktu", "wilayah"], how="inner") \
                          .merge(df_udara, on=["waktu", "wilayah"], how="inner")

    print(f"Jumlah baris setelah join (inner): {len(df_gabung)}")

    # Cek missing value setelah join
    print("\nCek missing value setelah join:")
    print(df_gabung.isnull().sum())

    # Sort biar urut secara kronologis per wilayah
    df_gabung = df_gabung.sort_values(["wilayah", "waktu"]).reset_index(drop=True)

    return df_gabung


# ========== TAHAP 2: EDA DASAR (Pertanyaan Deskriptif) ==========

def statistik_dasar(df_gabung):
    print("\n=== STATISTIK DASAR PER WILAYAH ===")
    kolom_numerik = [
        "sea_surface_temperature", "wave_height",
        "precipitation", "windspeed_10m",
        "pm2_5", "pm10",
    ]
    ringkasan = df_gabung.groupby("wilayah")[kolom_numerik].agg(["mean", "min", "max", "std"])
    print(ringkasan)
    return ringkasan


def plot_tren(df_gabung, kolom, judul, satuan=""):
    """Plot tren satu variabel untuk semua wilayah dalam satu grafik."""
    plt.figure(figsize=(12, 5))
    for wilayah in df_gabung["wilayah"].unique():
        subset = df_gabung[df_gabung["wilayah"] == wilayah]
        plt.plot(subset["waktu"], subset[kolom], label=wilayah, linewidth=1)
    plt.title(judul)
    plt.xlabel("Waktu")
    plt.ylabel(f"{kolom} {satuan}")
    plt.legend()
    plt.tight_layout()
    nama_file = f"tren_{kolom}.png"
    plt.savefig(nama_file, dpi=150)
    print(f"[OK] Grafik disimpan: {nama_file}")
    plt.close()


def jalankan_eda(df_gabung):
    statistik_dasar(df_gabung)

    print("\n=== MEMBUAT GRAFIK TREN (untuk jawab pertanyaan deskriptif) ===")
    plot_tren(df_gabung, "sea_surface_temperature", "Tren Suhu Permukaan Laut (SST)", "(°C)")
    plot_tren(df_gabung, "wave_height", "Tren Tinggi Gelombang", "(m)")
    plot_tren(df_gabung, "pm2_5", "Tren Kualitas Udara (PM2.5)", "(µg/m³)")
    plot_tren(df_gabung, "precipitation", "Tren Curah Hujan", "(mm)")


# ========== MAIN ==========

if __name__ == "__main__":
    df_gabung = ambil_dan_join()

    # Simpan hasil join ke CSV supaya bisa dipakai lagi di script berikutnya
    # (analisis korelasi, modeling) tanpa perlu query ulang ke MySQL setiap saat
    df_gabung.to_csv("data_gabungan.csv", index=False)
    print("\n[OK] Data gabungan disimpan ke data_gabungan.csv")

    jalankan_eda(df_gabung)

    print("\n=== SELESAI ===")
    print("Cek folder ini untuk melihat grafik hasil EDA (file .png)")