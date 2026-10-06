from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
import requests
import pandas as pd

daftar_wilayah = {
    "Kendari": (-3.97, 122.51),
    "Kolaka": (-4.05, 121.62),
    "Konawe Selatan": (-4.28, 122.35),
}

# ============ CREATE TABLE ============

def buat_tabel():
    hook = MySqlHook(mysql_conn_id="mysql_default")
    koneksi = hook.get_conn()
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

# ============ EXTRACT (persis dari tes_api.py yang sudah terbukti jalan) ============

def ambil_data_marine(**kontex):
    url = "https://marine-api.open-meteo.com/v1/marine"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "sea_surface_temperature,wave_height",
            "timezone": "Asia/Makassar",
            "past_days": 1,   # ambil kemarin+hari ini supaya tidak ada celah data
            "forecast_days": 1,
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            raise Exception(f"Marine API gagal - {nama} - status {respons.status_code}")
        data = respons.json()
        df = pd.DataFrame(data["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
    df_gabung = pd.concat(hasil, ignore_index=True)
    kontex["ti"].xcom_push(key="data_marine", value=df_gabung.to_json())

def ambil_data_cuaca(**kontex):
    url = "https://api.open-meteo.com/v1/forecast"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "precipitation,windspeed_10m",
            "timezone": "Asia/Makassar",
            "past_days": 1,
            "forecast_days": 1,
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            raise Exception(f"Weather API gagal - {nama} - status {respons.status_code}")
        data = respons.json()
        df = pd.DataFrame(data["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
    df_gabung = pd.concat(hasil, ignore_index=True)
    kontex["ti"].xcom_push(key="data_cuaca", value=df_gabung.to_json())

def ambil_data_udara(**kontex):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hasil = []
    for nama, (lat, lon) in daftar_wilayah.items():
        param = {
            "latitude": lat, "longitude": lon,
            "hourly": "pm2_5,pm10",
            "timezone": "Asia/Makassar",
            "past_days": 1,
            "forecast_days": 1,
        }
        respons = requests.get(url, params=param, timeout=30)
        if respons.status_code != 200:
            raise Exception(f"Air Quality API gagal - {nama} - status {respons.status_code}")
        data = respons.json()
        df = pd.DataFrame(data["hourly"])
        df["wilayah"] = nama
        hasil.append(df)
    df_gabung = pd.concat(hasil, ignore_index=True)
    kontex["ti"].xcom_push(key="data_udara", value=df_gabung.to_json())

# ============ LOAD (masing-masing ke tabelnya sendiri) ============

def simpan_marine(**kontex):
    data_json = kontex["ti"].xcom_pull(key="data_marine", task_ids="ambil_data_marine")
    if data_json is None:
        raise Exception("Data marine kosong (XCom None) - kemungkinan task ambil_data_marine gagal/tidak jalan")
    df = pd.read_json(data_json)
    hook = MySqlHook(mysql_conn_id="mysql_default")
    koneksi = hook.get_conn()
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

def simpan_cuaca(**kontex):
    data_json = kontex["ti"].xcom_pull(key="data_cuaca", task_ids="ambil_data_cuaca")
    if data_json is None:
        raise Exception("Data cuaca kosong (XCom None) - kemungkinan task ambil_data_cuaca gagal/tidak jalan")
    df = pd.read_json(data_json)
    hook = MySqlHook(mysql_conn_id="mysql_default")
    koneksi = hook.get_conn()
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

def simpan_udara(**kontex):
    data_json = kontex["ti"].xcom_pull(key="data_udara", task_ids="ambil_data_udara")
    if data_json is None:
        raise Exception("Data udara kosong (XCom None) - kemungkinan task ambil_data_udara gagal/tidak jalan")
    df = pd.read_json(data_json)
    hook = MySqlHook(mysql_conn_id="mysql_default")
    koneksi = hook.get_conn()
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

# ============ DAG ============

with DAG(
    dag_id="etl_lingkungan_pesisir",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",   # <-- ini yang bikin data nambah otomatis tiap hari
    catchup=False,
) as dag:

    task_buat_tabel = PythonOperator(task_id="buat_tabel", python_callable=buat_tabel)

    task_marine = PythonOperator(task_id="ambil_data_marine", python_callable=ambil_data_marine)
    task_cuaca = PythonOperator(task_id="ambil_data_cuaca", python_callable=ambil_data_cuaca)
    task_udara = PythonOperator(task_id="ambil_data_udara", python_callable=ambil_data_udara)

    task_simpan_marine = PythonOperator(task_id="simpan_marine", python_callable=simpan_marine)
    task_simpan_cuaca = PythonOperator(task_id="simpan_cuaca", python_callable=simpan_cuaca)
    task_simpan_udara = PythonOperator(task_id="simpan_udara", python_callable=simpan_udara)

    task_buat_tabel >> [task_marine, task_cuaca, task_udara]
    task_marine >> task_simpan_marine
    task_cuaca >> task_simpan_cuaca
    task_udara >> task_simpan_udara