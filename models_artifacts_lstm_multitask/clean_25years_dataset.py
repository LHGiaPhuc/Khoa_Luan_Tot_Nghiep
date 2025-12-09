# ------------------------------------------------------------
# clean_25years_dataset.py
# Tiền xử lý bộ 25 năm (2000–2025) cho 15 tỉnh/thành
# - Chuẩn hóa tên cột
# - Làm sạch + nội suy per-city (time interpolation)
# - Tạo đặc trưng: dew_point, heat_index, gust_proxy, sin/cos(day)
# - Cờ cảnh báo: rain_flag, storm_flag, heatwave_flag
# - Xuất file "Vietnam_Climate_cleaned_stable_model.xlsx" để train
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ================== PATHS ==================
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# Thử vài tên file thường gặp (chọn cái tồn tại)
CANDIDATES = [
    RAW_DIR / "Vietnam_2000_2025_NASA.xlsx",
    RAW_DIR / "Vietnam_Climate_2000_2025_NASA.xlsx",
    ROOT / "Vietnam_2000_2025_NASA.xlsx",
    ROOT / "Vietnam_Climate_2000_2025_NASA.xlsx",
]
IN_PATH = next((p for p in CANDIDATES if p.exists()), None)
if IN_PATH is None:
    raise SystemExit(
        "❌ Không tìm thấy file dữ liệu gốc. Hãy đặt file vào data/raw/ với tên "
        "'Vietnam_2000_2025_NASA.xlsx' (hoặc một trong các tên đã hỗ trợ)."
    )

OUT_PATH = OUT_DIR / "Vietnam_Climate_cleaned_stable_model.xlsx"

# ================== COLUMN MAP ==================
# Chuẩn về cột (khớp đúng với file bạn đã dùng để train/predict trước đây)
STD_COLS = [
    "Date", "City", "Lat", "Lon",
    "temperature_min_c", "temperature_max_c", "temperature_average_c",
    "HUMID", "PRESSURE", "wind_speed_km", "PRCP", "SW_down",
]

ALIAS_MAP = {
    "date": "Date",
    "city": "City",
    "lat": "Lat",
    "lon": "Lon",
    # nhiệt độ
    "temperature_min_": "temperature_min_c",
    "temperature_min_c": "temperature_min_c",
    "tmin": "temperature_min_c",
    "temperature_max_": "temperature_max_c",
    "temperature_max_c": "temperature_max_c",
    "tmax": "temperature_max_c",
    "temperature_average_": "temperature_average_c",
    "temperature_average_c": "temperature_average_c",
    "tavg": "temperature_average_c",
    # ẩm – áp – gió – mưa – bức xạ
    "humid": "HUMID",
    "humidity": "HUMID",
    "relative_humidity": "HUMID",
    "pressure": "PRESSURE",
    "press": "PRESSURE",
    "wind_speed": "wind_speed_km",
    "wind_speed_kmh": "wind_speed_km",
    "wind": "wind_speed_km",
    "prcp": "PRCP",
    "precipitation": "PRCP",
    "rain_mm": "PRCP",
    "sw_down": "SW_down",
    "shortwave_down": "SW_down",
}

NUMERIC_COLS = [
    "Lat", "Lon",
    "temperature_min_c", "temperature_max_c", "temperature_average_c",
    "HUMID", "PRESSURE", "wind_speed_km", "PRCP", "SW_down",
]

# ================== FEATURE ENGINEERING ==================
def dew_point_celsius(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus formula (°C)."""
    T = temp_c.astype(float)
    RH = rh.astype(float).clip(1, 100)  # tránh log(0)
    a, b = 17.625, 243.04
    alpha = np.log(RH/100.0) + (a * T) / (b + T)
    dp = (b * alpha) / (a - alpha)
    return dp

def heat_index_celsius(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """
    NOAA heat index, xấp xỉ cho °C. Tốt cho T >= 26°C.
    Với T thấp hơn, trả về T (không có hiệu ứng oi nóng).
    """
    T = temp_c.astype(float)
    RH = rh.astype(float).clip(1, 100)
    # chuyển tạm sang °F để dùng công thức gốc rồi quay lại °C
    Tf = T * 9/5 + 32
    HI_f = (-42.379 + 2.04901523*Tf + 10.14333127*RH
            - 0.22475541*Tf*RH - 6.83783e-3*Tf*Tf
            - 5.481717e-2*RH*RH + 1.22874e-3*Tf*Tf*RH
            + 8.5282e-4*Tf*RH*RH - 1.99e-6*Tf*Tf*RH*RH)
    HI_c = (HI_f - 32) * 5/9
    # nếu mát (<26°C) thì coi như không có hiệu ứng oi nóng
    HI_c = np.where(T < 26, T, HI_c)
    return pd.Series(HI_c, index=T.index)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["Date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.0)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.0)
    return df

# ================== CLEANING PIPELINE ==================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lower + strip
    lower_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=lower_map)
    cols_lower = {c: c.lower() for c in df.columns}

    final_map = {}
    for orig, low in cols_lower.items():
        final_map[orig] = ALIAS_MAP.get(low, orig)
    df = df.rename(columns=final_map)

    # đảm bảo City/Date chính tả đúng
    for c in list(df.columns):
        if c.lower() == "city" and c != "City":
            df = df.rename(columns={c: "City"})
        if c.lower() == "date" and c != "Date":
            df = df.rename(columns={c: "Date"})
    return df

def interpolate_city_block(g: pd.DataFrame) -> pd.DataFrame:
    """
    - Reindex theo dải ngày liên tục (min->max của city)
    - Nội suy time-based cho biến liên tục
    - Riêng PRCP: không nội suy tuyến tính dài -> fill nhỏ (limit=2) + còn lại = 0
    """
    g = g.sort_values("Date").set_index("Date")

    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full_idx)
    g.index.name = "Date"

    # giữ lại City/Lat/Lon
    g["City"] = g["City"].ffill().bfill()
    g["Lat"] = pd.to_numeric(g["Lat"], errors="coerce").ffill().bfill()
    g["Lon"] = pd.to_numeric(g["Lon"], errors="coerce").ffill().bfill()

    # numeric
    for c in NUMERIC_COLS:
        if c not in g.columns:
            continue
        g[c] = pd.to_numeric(g[c], errors="coerce")

    # nhóm nội suy "mượt": nhiệt độ, ẩm, áp, gió, bức xạ
    smooth_cols = [c for c in NUMERIC_COLS if c in g.columns and c != "PRCP"]
    g[smooth_cols] = g[smooth_cols].interpolate(
        method="time", limit=5, limit_direction="both"
    )

    # mưa: lấp gap nhỏ (2 ngày), còn lại xem như 0 (không mưa ghi 0)
    if "PRCP" in g.columns:
        g["PRCP"] = g["PRCP"].interpolate(method="time", limit=2, limit_direction="both")
        g["PRCP"] = g["PRCP"].fillna(0)

    return g.reset_index()

def quality_badge(pct_missing: float) -> str:
    if pct_missing <= 5:
        return "🟢"
    if pct_missing <= 20:
        return "🟡"
    return "🔴"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📥 Đọc dữ liệu: {IN_PATH}")

    # Đọc sheet đầu tiên
    df = pd.read_excel(IN_PATH, sheet_name=0)
    df = normalize_columns(df)

    # check cột tối thiểu
    required = {"Date", "City", "Lat", "Lon", "temperature_average_c", "PRCP", "wind_speed_km", "HUMID", "PRESSURE"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise SystemExit(f"❌ Thiếu cột bắt buộc: {missing}")

    # Chuẩn kiểu dữ liệu & sắp xếp
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City"]).drop_duplicates().sort_values(["City", "Date"])

    # Tiền xử lý theo City
    cities = df["City"].dropna().unique().tolist()
    cleaned = []
    print(f"🗺️ Số thành phố: {len(cities)}")

    for i, city in enumerate(sorted(cities), 1):
        g = df[df["City"] == city].copy()
        if g.empty:
            continue
        start, end = g["Date"].min().date(), g["Date"].max().date()
        before_len = len(g)

        g = interpolate_city_block(g)

        # tạo đặc trưng
        if {"temperature_average_c", "HUMID"}.issubset(g.columns):
            g["dew_point_c"] = dew_point_celsius(g["temperature_average_c"], g["HUMID"])
            g["heat_index_c"] = heat_index_celsius(g["temperature_average_c"], g["HUMID"])
        else:
            g["dew_point_c"] = np.nan
            g["heat_index_c"] = np.nan

        if "wind_speed_km" in g.columns:
            g["gust_proxy_km"] = g["wind_speed_km"] * 1.5
        else:
            g["gust_proxy_km"] = np.nan

        g = add_time_features(g)

        # cờ cảnh báo
        g["rain_flag"] = (g.get("PRCP", 0) > 5).astype(int)           # mưa có ý nghĩa
        g["storm_flag"] = (g.get("wind_speed_km", 0) > 40).astype(int) # gió mạnh / giông
        g["heatwave_flag"] = (g.get("temperature_max_c", g.get("temperature_average_c", 0)) >= 38).astype(int)

        after_len = len(g)
        # thống kê thiếu sau nội suy
        miss_pct = {}
        for c in NUMERIC_COLS:
            if c in g.columns:
                miss_pct[c] = g[c].isna().mean() * 100

        badge = quality_badge(np.nanmean(list(miss_pct.values())) if miss_pct else 100)
        print(f" {i:>2}. {city:<20} {start} → {end} | rows {before_len:>6} → {after_len:>6} | chất lượng {badge}")

        cleaned.append(g)

    out = pd.concat(cleaned, ignore_index=True)

    # Sắp xếp + chỉ giữ các cột chuẩn + cột mới
    base_cols = STD_COLS + ["dew_point_c", "heat_index_c", "gust_proxy_km", "sin_day", "cos_day",
                            "rain_flag", "storm_flag", "heatwave_flag"]
    keep_cols = [c for c in base_cols if c in out.columns]
    out = out.sort_values(["City", "Date"])[keep_cols]

    # In nhanh top chất lượng (sau xử lý gần như không còn NaN cho continuous cols)
    print("\n🔎 Kiểm tra thiếu dữ liệu (sau xử lý) theo thành phố (% NaN – càng thấp càng tốt):")
    report_rows = []
    for city, g in out.groupby("City"):
        row = {"City": city}
        for c in ["temperature_average_c", "PRCP", "wind_speed_km", "HUMID", "PRESSURE"]:
            if c in g.columns:
                row[c] = round(g[c].isna().mean() * 100, 3)
        report_rows.append(row)
    rep = pd.DataFrame(report_rows).fillna("")
    if not rep.empty:
        rep = rep.sort_values("City")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(rep.to_string(index=False))

    # Xuất file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, index=False)
    print(f"\n✅ Đã xuất: {OUT_PATH}")
    print(f"   Rows: {len(out):,} | Cities: {out['City'].nunique()} | Từ {out['Date'].min().date()} đến {out['Date'].max().date()}")
    print("   File này đã sẵn sàng để train mô hình 7 ngày + cảnh báo.")
    

if __name__ == "__main__":
    main()
