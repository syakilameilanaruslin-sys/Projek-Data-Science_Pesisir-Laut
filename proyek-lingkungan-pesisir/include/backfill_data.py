"""
backfill_data.py
Script untuk mengisi data HISTORIS ke 3 tabel MySQL sebelum DAG harian mulai jalan.
Dijalankan MANUAL SEKALI SAJA di awal (di luar Airflow), bukan lewat DAG.

Setelah ini jalan dan berhasil, DAG harian (@daily) yang menangani penambahan
data baru setiap hari ke depannya secara otomatis.
"""

import requests
import pandas as pd
import mysql.connector

# ========== KONFIGURASI ==========

daftar_wilayah = {
    "Kendari": (-3.97, 122.51),
    "Kolaka": (-4.05, 121.62),
    "Konawe Selatan": (-4.28, 122.35),
}

# Rentang backfill - sesuaikan sesuai kebutuhan
START_DATE = "2026-01-01"
END_DATE = "2026-09-24"

# Kredensial MySQL - samakan dengan airflow_settings.yaml
DB_CONFIG = {
    "host": "localhost",       # ganti sesuai host container MySQL kamu
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "lingkungan_pesisir",
}


# ========== BUAT TABEL (jaga-jaga kalau backfill dijalankan sebelum DAG) ==========

def buat_tabel(koneksi):
    kursor = koneksi.cursor()
    kursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_oseanografi (
            waktu DATETIME, wilayah VARCHAR(50),
            sea_surface_temperature FLOAT, wave_height FLOAT,
            PRIMARY KEY (waktu, wilayah)
        )
    """)
    kursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_cuaca (
            waktu DATETIME, wilayah VARCHAR(50),
            precipitation FLOAT, windspeed_10m FLOAT,
            PRIMARY KEY (waktu, wilayah)
        )
    """)
    kursor.execute("""
        CREATE TABLE IF NOT EXISTS tabel_kualitas_udara (
            waktu DATETIME, wilayah VARCHAR(50),
            pm2_5 FLOAT, pm10 FLOAT,
            PRIMARY KEY (waktu, wilayah)
        )
    """)
    koneksi.commit()
    kursor.close()
    print("[OK] 3 tabel siap (dibuat kalau belum ada).")


# ========== EXTRACT (sama seperti tes_api.py yang sudah terbukti jalan) ==========

def tarik_data_marine():
    url = "https://marine-api.open-meteo.com/v1/marine"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "sea_surface_temperature,wave_height",
            "start_date": START_DATE, "end_date": END_DATE,
            "timezone": "Asia/Makassar",
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            print(f"[GAGAL] Marine - {nama} - status {respons.status_code}")
            continue
        df = pd.DataFrame(respons.json()["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
        print(f"[OK] Marine - {nama}: {len(df)} baris")
    return pd.concat(hasil, ignore_index=True) if hasil else None


def tarik_data_cuaca():
    url = "https://archive-api.open-meteo.com/v1/archive"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,windspeed_10m",
            "start_date": START_DATE, "end_date": END_DATE,
            "timezone": "Asia/Makassar",
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            print(f"[GAGAL] Weather - {nama} - status {respons.status_code}")
            continue
        df = pd.DataFrame(respons.json()["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
        print(f"[OK] Weather - {nama}: {len(df)} baris")
    return pd.concat(hasil, ignore_index=True) if hasil else None


def tarik_data_udara():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "pm2_5,pm10",
            "start_date": START_DATE, "end_date": END_DATE,
            "timezone": "Asia/Makassar",
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            print(f"[GAGAL] Air Quality - {nama} - status {respons.status_code}")
            continue
        df = pd.DataFrame(respons.json()["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
        print(f"[OK] Air Quality - {nama}: {len(df)} baris")
    return pd.concat(hasil, ignore_index=True) if hasil else None


# ========== LOAD (masing-masing ke tabelnya sendiri, pakai upsert) ==========

def simpan_marine(koneksi, df):
    kursor = koneksi.cursor()
    for _, b in df.iterrows():
        kursor.execute("""
            INSERT INTO tabel_oseanografi (waktu, wilayah, sea_surface_temperature, wave_height)
            VALUES (%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                sea_surface_temperature=VALUES(sea_surface_temperature),
                wave_height=VALUES(wave_height)
        """, (b["time"], b["wilayah"], b["sea_surface_temperature"], b["wave_height"]))
    koneksi.commit()
    kursor.close()
    print(f"[OK] {len(df)} baris disimpan ke tabel_oseanografi")


def simpan_cuaca(koneksi, df):
    kursor = koneksi.cursor()
    for _, b in df.iterrows():
        kursor.execute("""
            INSERT INTO tabel_cuaca (waktu, wilayah, precipitation, windspeed_10m)
            VALUES (%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                precipitation=VALUES(precipitation),
                windspeed_10m=VALUES(windspeed_10m)
        """, (b["time"], b["wilayah"], b["precipitation"], b["windspeed_10m"]))
    koneksi.commit()
    kursor.close()
    print(f"[OK] {len(df)} baris disimpan ke tabel_cuaca")


def simpan_udara(koneksi, df):
    kursor = koneksi.cursor()
    for _, b in df.iterrows():
        kursor.execute("""
            INSERT INTO tabel_kualitas_udara (waktu, wilayah, pm2_5, pm10)
            VALUES (%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                pm2_5=VALUES(pm2_5),
                pm10=VALUES(pm10)
        """, (b["time"], b["wilayah"], b["pm2_5"], b["pm10"]))
    koneksi.commit()
    kursor.close()
    print(f"[OK] {len(df)} baris disimpan ke tabel_kualitas_udara")


# ========== MAIN ==========

if __name__ == "__main__":
    print(f"=== BACKFILL DATA {START_DATE} sampai {END_DATE} ===\n")

    koneksi = mysql.connector.connect(**DB_CONFIG)
    buat_tabel(koneksi)

    print("\n--- Menarik data dari API ---")
    df_marine = tarik_data_marine()
    df_cuaca = tarik_data_cuaca()
    df_udara = tarik_data_udara()

    print("\n--- Menyimpan ke MySQL ---")
    if df_marine is not None:
        simpan_marine(koneksi, df_marine)
    if df_cuaca is not None:
        simpan_cuaca(koneksi, df_cuaca)
    if df_udara is not None:
        simpan_udara(koneksi, df_udara)

    koneksi.close()
    print("\n=== BACKFILL SELESAI ===")