import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

BASE = Path(__file__).parent.parent / "analysis"
DATA = BASE / "data_siap_model.csv"

df = pd.read_csv(DATA)
df['waktu'] = pd.to_datetime(df['waktu'])
df = df.sort_values(['wilayah', 'waktu']).reset_index(drop=True)

FEATURES = [
    "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
    "roll_mean_24", "roll_mean_168", "roll_std_24",
    "jam", "hari_dalam_minggu", "bulan",
]

out = BASE / "models"
out.mkdir(exist_ok=True)

for wilayah in df['wilayah'].unique():
    d = df[df['wilayah']==wilayah].copy()
    X = d[FEATURES]
    y = d['target']
    lr = LinearRegression().fit(X,y)
    xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, objective='reg:squarederror').fit(X,y)
    joblib.dump(lr, out / f"lr_{wilayah.replace(' ','_')}.pkl")
    joblib.dump(xgb, out / f"xgb_{wilayah.replace(' ','_')}.pkl")
    print('saved', wilayah)
