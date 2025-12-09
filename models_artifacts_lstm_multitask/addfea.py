import pandas as pd

df = pd.read_excel("D:/khoaluan/data/processed/Vietnam_Climate_cleaned_stable_model.xlsx")

# --- 1. Pressure Drop ---
df["pressure_drop_1d"] = df["PRESSURE"].diff(1)
df["pressure_drop_3d"] = df["PRESSURE"].diff(3)

# --- 2. Humidity Trend ---
df["humidity_diff_1d"] = df["HUMID"].diff(1)

# --- 3. Temperature Gradient ---
df["temp_diff_1d"] = df["temperature_average_c"].diff(1)

# --- 4. Rolling PRCP ---
df["prcp_3d_avg"] = df["PRCP"].rolling(window=3).mean()

# --- 5. Lag Feature ---
df["temp_lag_1"] = df["temperature_average_c"].shift(1)

# Loại bỏ NaN do diff/rolling
df = df.dropna().reset_index(drop=True)

# Lưu lại file mới
df.to_excel("D:/khoaluan/data/processed/Vietnam_Climate_enhanced_features.xlsx", index=False)

print("DONE – đã tạo các feature cần thiết!")
