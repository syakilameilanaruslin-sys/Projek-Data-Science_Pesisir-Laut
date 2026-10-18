from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard Lingkungan Pesisir Sulawesi Tenggara",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent.parent
ANALYSIS = BASE / "analysis"
MODELS_DIR = ANALYSIS / "models"

WILAYAH = ["Kendari", "Kolaka", "Konawe Selatan"]
FITUR = [
    "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "jam", "hari_dalam_minggu", "bulan",
]
TARGET = "sst_anomali"
JAM_HORIZON = 24

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_risk() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS / "data_dengan_indeks_risiko.csv", parse_dates=["waktu"])
    return df.sort_values(["wilayah", "waktu"]).reset_index(drop=True)


@st.cache_data
def load_model_data() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS / "data_siap_model.csv", parse_dates=["waktu"])
    return df.sort_values(["wilayah", "waktu"]).reset_index(drop=True)


@st.cache_data
def load_ranking() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / "ranking_wilayah.csv")


@st.cache_data
def load_evaluasi() -> pd.DataFrame:
    path = ANALYSIS / "tabel_akurasi_lengkap.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(ANALYSIS / "hasil_evaluasi_model.csv")


@st.cache_data
def load_korelasi() -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / "hasil_korelasi.csv")


@st.cache_resource
def load_models() -> dict:
    models = {}
    for p in MODELS_DIR.glob("*.pkl"):
        models[p.stem] = joblib.load(p)
    return models


df = load_risk()
df_model = load_model_data()
ranking = load_ranking()
evaluasi = load_evaluasi()
korelasi = load_korelasi()
models = load_models()
rata_sst = df.groupby("wilayah")["sea_surface_temperature"].mean().to_dict()


AWALAN_MODEL = {"Linear Regression": "lr", "XGBoost": "xgb"}


def nama_file_model(wilayah: str, jenis: str) -> str:
    return f"{AWALAN_MODEL[jenis]}_{wilayah.replace(' ', '_')}"


def status_anomali(anomali: float) -> tuple[str, str]:
    magnitudo = abs(anomali)
    arah = "positif (pemanasan)" if anomali >= 0 else "negatif (pendinginan)"
    if magnitudo < 0.25:
        return "Normal", "🟢"
    if magnitudo < 0.75:
        return f"Anomali ringan {arah}", "🟡"
    if magnitudo < 1.25:
        return f"Anomali sedang {arah}", "🟠"
    return f"Anomali kuat {arah}", "🔴"


def bangun_fitur(deret: pd.Series, ts: pd.Timestamp) -> dict:
    if len(deret) < 168:
        raise ValueError("Butuh minimal 168 jam data historis sebelum timestamp terpilih.")
    return {
        "lag_1": deret.iloc[-2],
        "lag_3": deret.iloc[-4],
        "lag_6": deret.iloc[-7],
        "lag_12": deret.iloc[-13],
        "lag_24": deret.iloc[-25],
        "roll_mean_24": deret.iloc[-24:].mean(),
        "roll_mean_168": deret.iloc[-168:].mean(),
        "roll_std_24": deret.iloc[-24:].std(),
        "jam": ts.hour,
        "hari_dalam_minggu": ts.dayofweek,
        "bulan": ts.month,
    }


def prediksi(fitur_row: pd.DataFrame, jenis: str, wilayah: str) -> float:
    if jenis == "Naive (Persistence)":
        return float(fitur_row["lag_1"].iloc[0])
    model = models[nama_file_model(wilayah, jenis)]
    return float(model.predict(fitur_row[FITUR])[0])


def halaman_beranda():
    st.title("🌊 Dashboard Lingkungan Pesisir Sulawesi Tenggara")
    st.caption(
        "Analisis risiko lingkungan & prediksi anomali *Sea Surface Temperature* (SST) "
        "24 jam ke depan di Kendari, Kolaka, dan Konawe Selatan — data hourly Open-Meteo "
        "(Jan–Sep 2026)."
    )

    terakhir = df["waktu"].max()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Periode Data", f"{df['waktu'].min():%d %b %Y} – {terakhir:%d %b %Y}")
    k2.metric("Total Rekaman", f"{len(df):,} baris")
    k3.metric("Wilayah Pantau", len(WILAYAH))
    k4.metric("Frekuensi", "Per jam")

    st.subheader("🏆 Peringkat Indeks Risiko Lingkungan")
    tertinggi = ranking.iloc[0]
    st.info(
        f"**{tertinggi['wilayah']}** memiliki rata-rata indeks risiko tertinggi "
        f"(**{tertinggi['rata_rata_risiko']:.3f}**) dari kombinasi anomali SST, tinggi gelombang, "
        f"PM2.5, dan PM10."
    )
    fig = px.bar(
        ranking,
        x="wilayah",
        y="rata_rata_risiko",
        error_y="std_risiko",
        color="rata_rata_risiko",
        color_continuous_scale="OrRd",
        labels={"wilayah": "Wilayah", "rata_rata_risiko": "Rata-rata Indeks Risiko"},
    )
    fig.update_layout(coloraxis_showscale=False, height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("📌 Ringkasan Kinerja Model Prediksi")
    best = evaluasi[evaluasi["terbaik"].notna()] if "terbaik" in evaluasi.columns else evaluasi
    cols = st.columns(3)
    for i, wilayah in enumerate(WILAYAH):
        baris = best[best["wilayah"] == wilayah]
        if baris.empty:
            baris = evaluasi[evaluasi["wilayah"] == wilayah]
        b = baris.iloc[0]
        with cols[i]:
            st.markdown(f"**{wilayah}**")
            st.write(f"Model terbaik: **{b['model']}**")
            st.write(f"Akurasi: **{b['akurasi (%)']:.2f}%** · RMSE: {b['rmse']:.4f}")

    st.subheader("❓ Pertanyaan Analisis")
    st.markdown(
        """
        1. Bagaimana tren SST, tinggi gelombang, dan kualitas udara (PM2.5/PM10) dari waktu ke waktu?
        2. Apakah anomali SST / gelombang tinggi berkorelasi dengan degradasi kualitas udara?
        3. Wilayah mana yang memiliki indeks risiko lingkungan komposit tertinggi?
        4. Seberapa akur model *machine learning* memprediksi anomali SST 24 jam ke depan?
        """
    )


def halaman_tren():
    st.title("📈 Tren Waktu Parameter Lingkungan")
    pilihan = st.multiselect(
        "Pilih parameter",
        ["sea_surface_temperature", "wave_height", "pm2_5", "pm10", "precipitation", "windspeed_10m", "indeks_risiko"],
        default=["sea_surface_temperature", "pm2_5"],
    )
    wilayah_terpilih = st.multiselect("Pilih wilayah", WILAYAH, default=WILAYAH)
    if not pilihan or not wilayah_terpilih:
        st.warning("Pilih minimal satu parameter dan satu wilayah.")
        return

    sub = df[df["wilayah"].isin(wilayah_terpilih)]
    rentang = st.slider(
        "Rentang waktu",
        min_value=sub["waktu"].min().to_pydatetime(),
        max_value=sub["waktu"].max().to_pydatetime(),
        value=(sub["waktu"].min().to_pydatetime(), sub["waktu"].max().to_pydatetime()),
    )
    sub = sub[(sub["waktu"] >= rentang[0]) & (sub["waktu"] <= rentang[1])]

    for kolom in pilihan:
        agregat = (
            sub.set_index("waktu")
            .groupby("wilayah")[kolom]
            .resample("D")
            .mean()
            .reset_index()
        )
        fig = px.line(
            agregat,
            x="waktu",
            y=kolom,
            color="wilayah",
            title=f"{kolom} (rata-rata harian)",
            labels={"waktu": "Waktu", kolom: kolom, "wilayah": "Wilayah"},
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")


def halaman_risiko():
    st.title("⚠️ Indeks Risiko Lingkungan Komposit")
    st.markdown(
        "Indeks = rata-rata bobot sama (0,25) dari normalisasi min-max: **anomali SST absolut**, "
        "**tinggi gelombang**, **PM2.5**, dan **PM10**. Status: Rendah <0,25 · Sedang <0,5 · "
        "Tinggi <0,75 · Sangat Tinggi ≥0,75."
    )

    wilayah = st.selectbox("Wilayah", WILAYAH)
    sub = df[df["wilayah"] == wilayah]

    k1, k2, k3 = st.columns(3)
    k1.metric("Rata-rata Indeks", f"{sub['indeks_risiko'].mean():.3f}")
    k2.metric("Maksimum", f"{sub['indeks_risiko'].max():.3f}")
    k3.metric("% Status ≥ Tinggi", f"{(sub['status_risiko'].isin(['Tinggi', 'Sangat Tinggi'])).mean() * 100:.1f}%")

    agregat = sub.set_index("waktu")["indeks_risiko"].resample("D").mean().reset_index()
    fig = px.area(agregat, x="waktu", y="indeks_risiko", title=f"Indeks Risiko Harian — {wilayah}")
    fig.add_hline(y=0.25, line_dash="dot", annotation_text="Sedang")
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="Tinggi")
    fig.add_hline(y=0.75, line_dash="dot", annotation_text="Sangat Tinggi")
    fig.update_layout(height=400, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    distribusi = sub["status_risiko"].value_counts().reindex(
        ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    ).fillna(0).reset_index()
    distribusi.columns = ["status_risiko", "jumlah"]
    fig2 = px.bar(
        distribusi,
        x="status_risiko",
        y="jumlah",
        color="status_risiko",
        color_discrete_map={"Rendah": "#2e7d32", "Sedang": "#f9a825", "Tinggi": "#ef6c00", "Sangat Tinggi": "#c62828"},
        title="Distribusi Status Risiko",
    )
    fig2.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10))
    st.plotly_chart(fig2, width="stretch")


def halaman_korelasi():
    st.title("🔗 Korelasi Anomali SST & Gelombang terhadap Kualitas Udara")
    st.dataframe(
        korelasi.style.format(
            {"pearson_r": "{:.3f}", "pearson_p": "{:.2e}", "spearman_r": "{:.3f}", "spearman_p": "{:.2e}"}
        ),
        width="stretch",
    )

    st.subheader("Heatmap Korelasi per Wilayah")
    for wilayah in WILAYAH:
        cols = ["sea_surface_temperature", "wave_height", "pm2_5", "pm10", "indeks_risiko"]
        matriks = df[df["wilayah"] == wilayah][cols].corr()
        fig = px.imshow(
            matriks,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{wilayah}",
        )
        fig.update_layout(height=420, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.info(
        "Temuan: korelasi anomali SST–PM10 dan gelombang–PM2.5 bertanda negatif dengan kekuatan "
        "lemah, konsisten dengan pola angin musim & hujan pencuci polutan (scavenging) di pesisir."
    )


def halaman_evaluasi():
    st.title("🤖 Evaluasi Model Prediksi Anomali SST (+24 Jam)")
    st.markdown(
        "Target: **anomali SST 24 jam ke depan**. Split kronologis 85/15. "
        "Akurasi = 100 − MAPE."
    )
    st.dataframe(
        evaluasi.style.format({"rmse": "{:.4f}", "mae": "{:.4f}", "mape": "{:.2f}", "akurasi (%)": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )

    metrik = st.radio("Metrik", ["akurasi (%)", "rmse", "mae", "mape"], horizontal=True)
    fig = px.bar(
        evaluasi,
        x="wilayah",
        y=metrik,
        color="model",
        barmode="group",
        title=f"Perbandingan {metrik.upper()} antar Model",
        text_auto=".3f",
    )
    fig.update_layout(height=420, margin=dict(t=40, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Plot Prediksi vs Aktual (Data Uji)")
    wilayah = st.selectbox("Wilayah", WILAYAH, key="eval_wilayah")
    gambar = ANALYSIS / f"prediksi_{wilayah.replace(' ', '_')}.png"
    if gambar.exists():
        st.image(str(gambar), caption=f"Prediksi vs Aktual — {wilayah}", width="stretch")
    else:
        st.warning(f"Gambar {gambar.name} tidak ditemukan.")

    with st.expander("Coba model alternatif Prophet"):
        st.markdown("Prophet (seasonality mingguan) dibandingkan baseline Naive:")
        st.dataframe(pd.read_csv(ANALYSIS / "hasil_evaluasi_prophet.csv"), width="stretch")


def halaman_prediksi():
    st.title("🔮 Prediksi Interaktif Anomali SST (+24 Jam)")
    st.markdown(
        "Pilih wilayah, model, dan waktu acuan (*timestamp*). Fitur lag/rolling dihitung ulang "
        "dari deret anomali SST historis, lalu model memprediksi anomali SST pada **t + 24 jam**."
    )

    kol_w, kol_m, kol_mode = st.columns([1.2, 1.2, 1.4])
    with kol_w:
        wilayah = st.selectbox("Wilayah", WILAYAH)
    with kol_m:
        jenis_model = st.selectbox("Model", ["Linear Regression", "XGBoost", "Naive (Persistence)"])
    with kol_mode:
        mode = st.radio(
            "Mode",
            ["Rentang waktu historis (ada nilai aktual)", "Prediksi mendatang (dari data terakhir)"],
            horizontal=True,
        )

    deret = df[df["wilayah"] == wilayah].set_index("waktu")[TARGET].sort_index()
    tersedia = any(
        f"{prefix}_{wilayah.replace(' ', '_')}" in models for prefix in ("lr", "xgb")
    )
    if not tersedia:
        st.error("Model pickel tidak ditemukan di analysis/models/. Jalankan dashboard/train_models.py.")
        return

    if mode.startswith("Rentang"):
        df_w = df_model[df_model["wilayah"] == wilayah].sort_values("waktu")
        idx = st.slider(
            "Posisi waktu acuan dalam data",
            min_value=0,
            max_value=len(df_w) - 1,
            value=len(df_w) - 1 - JAM_HORIZON,
        )
        ts = df_w.iloc[idx]["waktu"]
        deret_sampai_t = deret.loc[:ts]
        fitur_row = pd.DataFrame([bangun_fitur(deret_sampai_t, ts)])
        pred = prediksi(fitur_row, jenis_model, wilayah)
        target_waktu = ts + pd.Timedelta(hours=JAM_HORIZON)
        aktual_row = deret.loc[target_waktu] if target_waktu in deret.index else None
    else:
        ts = deret.index.max()
        deret_sampai_t = deret
        fitur_row = pd.DataFrame([bangun_fitur(deret_sampai_t, ts)])
        pred = prediksi(fitur_row, jenis_model, wilayah)
        target_waktu = ts + pd.Timedelta(hours=JAM_HORIZON)
        aktual_row = None
        st.caption(f"Data historis terakhir: **{ts:%d %b %Y %H:%M}** → prediksi untuk **{target_waktu:%d %b %Y %H:%M}**")

    st.divider()
    sst_pred = rata_sst[wilayah] + pred
    status, emoji = status_anomali(pred)
    ada_aktual = aktual_row is not None and not pd.isna(aktual_row)
    sst_aktual = rata_sst[wilayah] + aktual_row if ada_aktual else None
    err = abs(pred - aktual_row) if ada_aktual else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Anomali SST Diprediksi",
        f"{pred:+.3f} °C",
        delta=f"Aktual: {aktual_row:+.3f} °C" if ada_aktual else None,
        delta_color="off",
    )
    k2.metric(
        "SST Diprediksi",
        f"{sst_pred:.2f} °C",
        delta=f"SST aktual: {sst_aktual:.2f} °C" if ada_aktual else None,
        delta_color="off",
    )
    k3.metric(
        "Rata-rata SST Wilayah",
        f"{rata_sst[wilayah]:.2f} °C",
        delta=f"Galat absolut: {err:.4f}" if ada_aktual else None,
        delta_color="off",
    )
    k4.metric("Status", f"{emoji} {status}")

    with st.expander("🔧 Lihat fitur input model", expanded=True):
        tampil = fitur_row.T.copy()
        tampil.columns = ["nilai"]
        st.dataframe(tampil.style.format("{:.4f}"), width="stretch")

    st.subheader("Konteks 7 Hari Terakhir vs Prediksi")
    jendela = deret_sampai_t.iloc[-168:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=jendela.index, y=jendela.values, mode="lines", name="Anomali SST aktual", line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(
            x=[ts, target_waktu],
            y=[deret_sampai_t.iloc[-1], pred],
            mode="lines+markers",
            name=f"Prediksi +{JAM_HORIZON} jam ({jenis_model})",
            line=dict(color="#d62728", dash="dash"),
            marker=dict(size=10),
        )
    )
    if aktual_row is not None and not pd.isna(aktual_row):
        fig.add_trace(
            go.Scatter(
                x=[target_waktu], y=[aktual_row], mode="markers", name="Nilai aktual",
                marker=dict(color="green", size=12, symbol="diamond"),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        height=420,
        margin=dict(t=30, b=10),
        xaxis_title="Waktu",
        yaxis_title="Anomali SST (°C)",
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Perbandingan Semua Model pada Timestamp Ini")
    baris = []
    for jenis in ["Linear Regression", "XGBoost", "Naive (Persistence)"]:
        nilai = prediksi(fitur_row, jenis, wilayah)
        baris.append(
            {
                "Model": jenis,
                "Prediksi Anomali (°C)": round(nilai, 4),
                "Prediksi SST (°C)": round(rata_sst[wilayah] + nilai, 3),
                "Selisih vs LR (°C)": round(nilai - prediksi(fitur_row, "Linear Regression", wilayah), 4),
            }
        )
    st.dataframe(pd.DataFrame(baris), width="stretch", hide_index=True)

    if aktual_row is not None and not pd.isna(aktual_row):
        st.caption(
            f"Anomali SST terjadi akibat deviasi suhu permukaan laut terhadap rata-rata historis "
            f"wilayah ({rata_sst[wilayah]:.2f} °C). Prediksi {pred:+.3f} °C vs aktual {aktual_row:+.3f} °C."
        )


HALAMAN = {
    "🏠 Beranda": halaman_beranda,
    "📈 Tren Waktu": halaman_tren,
    "⚠️ Indeks Risiko": halaman_risiko,
    "🔗 Korelasi": halaman_korelasi,
    "🤖 Evaluasi Model": halaman_evaluasi,
    "🔮 Prediksi Interaktif": halaman_prediksi,
}

with st.sidebar:
    st.title("🌊 Pesisir Sulawesi Tenggara")
    st.markdown("**Prediksi Anomali SST & Kualitas Udara**")
    halaman_dipilih = st.radio("Navigasi", list(HALAMAN.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Data: Open-Meteo API · Model: Linear Regression, XGBoost, Naive")
    st.caption(f"Rentang data: {df['waktu'].min():%Y-%m-%d} → {df['waktu'].max():%Y-%m-%d}")

HALAMAN[halaman_dipilih]()
