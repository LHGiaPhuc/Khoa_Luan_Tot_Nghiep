import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Vietnam_Climate_enhanced_features.xlsx"
SCALER_OUT = ROOT / "model" / "scalers_by_city.pkl"
META_OUT   = ROOT / "model" / "meta_info.csv"

WINDOW = 60

FEATURE_BASE_COLS = [
    "temperature_min_c",
    "temperature_max_c",
    "temperature_average_c",
    "HUMID",
    "PRESSURE",
    "wind_speed_km",
    "PRCP",
    "SW_down",
    "dew_point_c",
    "heat_index_c",
    "gust_proxy_km",
    "sin_day",
    "cos_day",

    "pressure_drop_1d",
    "pressure_drop_3d",
    "humidity_diff_1d",
    "temp_diff_1d",
    "prcp_3d_avg",
    "temp_lag_1",
]

REGION_COLS = ["Region_Central", "Region_North", "Region_South"]

FEATURE_COLS = FEATURE_BASE_COLS + REGION_COLS

df = pd.read_excel(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

def map_region(city):
    if city in ["Hanoi", "Hai Phong", "Quang Ninh", "Thanh Hoa", "Nghe An (Vinh)"]:
        return "North"
    elif city in [
        "Hue (Thua Thien Hue)",
        "Da Nang",
        "Quang Nam (Tam Ky)",
        "Binh Dinh (Quy Nhon)",
        "Nha Trang (Khanh Hoa)",
        "Buon Ma Thuot (Dak Lak)",
        "Da Lat (Lam Dong)",
    ]:
        return "Central"
    else:
        return "South"

df["Region"] = df["City"].apply(map_region)
df = pd.get_dummies(df, columns=["Region"])

scalers = {}
meta = []

for city, g in df.groupby("City"):
    g = g.sort_values("Date")

    if len(g) < WINDOW:
        print(f"Skipping {city}: too little data ({len(g)} rows)")
        continue

    scaler = MinMaxScaler()
    scaler.fit(g[FEATURE_COLS])  

    scalers[city] = scaler
    meta.append({"city": city, "region": map_region(city)})

joblib.dump(scalers, SCALER_OUT)
pd.DataFrame(meta).to_csv(META_OUT, index=False)

print("DONE — scalers_by_city.pkl and meta_info.csv rebuilt successfully!")
