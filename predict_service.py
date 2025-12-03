from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Reshape,
    Softmax,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add,
)

from weather_events import detect_events, build_summary

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models_artifacts_transformer_multitask"

MODEL_PATH = MODEL_DIR / "transformer_multitask_heat.h5"
SCALER_PATH = MODEL_DIR / "scalers_by_city.pkl"
META_INFO_PATH = MODEL_DIR / "meta_info.csv"

CLIMATE_DATA_PATH = ROOT / "Vietnam_Climate_enhanced_features.xlsx"

WINDOW = 60
HORIZON = 7


META_INFO = pd.read_csv(META_INFO_PATH)

SCALERS_BY_CITY: Dict[str, Any] = joblib.load(SCALER_PATH)

_any_scaler = next(iter(SCALERS_BY_CITY.values()))
if hasattr(_any_scaler, "feature_names_in_"):
    NUM_FEATURES = len(_any_scaler.feature_names_in_)
else:
    NUM_FEATURES = _any_scaler.n_features_in_  


def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pe = angle_rads[np.newaxis, ...]
    return tf.cast(pe, dtype=tf.float32)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):

    x = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(head_size)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_transformer(
    timesteps: int,
    num_features: int,
    horizon: int = 7,
    num_rain_classes: int = 5,
    num_wind_classes: int = 5,
) -> Model:
    d_model = 128

    inp = Input(shape=(timesteps, num_features), name="input_layer")

    x = Dense(d_model)(inp)

    pe = positional_encoding(timesteps, d_model)
    x = x + pe

    x = transformer_encoder(
        x, head_size=d_model, num_heads=4, ff_dim=256, dropout=0.2
    )
    x = transformer_encoder(
        x, head_size=d_model, num_heads=4, ff_dim=256, dropout=0.2
    )

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)

    # Outputs
    temp_out = Dense(horizon, name="temp_out")(x) 
    wind_out = Dense(horizon, name="wind_out")(x)

    rain_logits = Dense(horizon * num_rain_classes)(x)
    rain_logits = Reshape((horizon, num_rain_classes))(rain_logits)
    rain_level_out = Softmax(axis=-1, name="rain_level_out")(rain_logits)

    windlvl_logits = Dense(horizon * num_wind_classes)(x)
    windlvl_logits = Reshape((horizon, num_wind_classes))(windlvl_logits)
    wind_level_out = Softmax(axis=-1, name="wind_level_out")(windlvl_logits)

    model = Model(
        inp,
        [temp_out, wind_out, rain_level_out, wind_level_out],
        name="transformer_multitask",
    )
    return model

MODEL = build_transformer(
    timesteps=WINDOW,
    num_features=NUM_FEATURES,
    horizon=HORIZON,
    num_rain_classes=5,
    num_wind_classes=5,
)

MODEL.load_weights(MODEL_PATH)

CITY_CODE_TO_NAME: Dict[str, str] = {
    "hanoi": "Hanoi",
    "haiphong": "Hai Phong",
    "quangninh": "Quang Ninh",
    "thanhhoa": "Thanh Hoa",

    "vinh": "Nghe An (Vinh)",
    "hue": "Hue (Thua Thien Hue)",

    "danang": "Da Nang",
    "quynhon": "Binh Dinh (Quy Nhon)",
    "nhatrang": "Nha Trang (Khanh Hoa)",
    "quangnam": "Quang Nam (Tam Ky)",

    "dalat": "Da Lat (Lam Dong)",
    "buonmethuot": "Buon Ma Thuot (Dak Lak)",

    "hcmc": "Ho Chi Minh City",
    "hochiminh": "Ho Chi Minh City",
    "saigon": "Ho Chi Minh City",
    "cantho": "Can Tho",
    "camau": "Ca Mau",
}

def _get_city_meta(city_code: str) -> Dict[str, Any]:
    city_name = CITY_CODE_TO_NAME.get(city_code.lower(), city_code)
    row = META_INFO.loc[META_INFO["city"] == city_name]
    if row.empty:
        raise ValueError(f"City '{city_name}' not found in meta_info.csv")
    return {
        "city_name": city_name,
        "region": row.iloc[0]["region"],
    }

def _build_future_dates_from_end(end_date: str, horizon: int) -> List[pd.Timestamp]:
    end_dt = pd.to_datetime(end_date)
    return [end_dt + pd.Timedelta(days=i + 1) for i in range(horizon)]

def _vn_day_name(idx: int) -> str:
    labels = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    return labels[idx] if 0 <= idx < len(labels) else ""

def _find_col(df: pd.DataFrame, logical_name: str) -> str:
    logical = logical_name.lower()
    for c in df.columns:
        if c.lower() == logical:
            return c
    raise ValueError(f"Column '{logical_name}' not found in climate data")

def _build_input_window(city_name: str, end_date: str) -> np.ndarray:
    if not CLIMATE_DATA_PATH.exists():
        raise FileNotFoundError(f"Climate data not found: {CLIMATE_DATA_PATH}")

    df = pd.read_excel(CLIMATE_DATA_PATH)

    date_col = _find_col(df, "Date")
    city_col = _find_col(df, "City")

    df[date_col] = pd.to_datetime(df[date_col])
    end_dt = pd.to_datetime(end_date)

    df_city = (
        df[(df[city_col] == city_name) & (df[date_col] <= end_dt)]
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if df_city.empty:
        raise ValueError(f"No climate data for {city_name} up to {end_dt.date()}")

    if len(df_city) >= WINDOW:
        window_df = df_city.tail(WINDOW)
    else:
        window_df = df_city.copy()

    scaler = SCALERS_BY_CITY.get(city_name)
    if scaler is None:
        raise ValueError(f"No scaler for city {city_name}")

    feature_cols = list(scaler.feature_names_in_)

    missing_cols = [c for c in feature_cols if c not in window_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns for {city_name}: {missing_cols}")

    X_raw = window_df[feature_cols].to_numpy()
    X_scaled = scaler.transform(X_raw)

    if X_scaled.shape[0] < WINDOW:
        pad_len = WINDOW - X_scaled.shape[0]
        pad_block = np.repeat(X_scaled[0:1, :], pad_len, axis=0)
        X_scaled = np.vstack([pad_block, X_scaled])

    return X_scaled.reshape(1, WINDOW, -1)

def _run_model_and_decode(X_input: np.ndarray, future_dates: List[pd.Timestamp]):
    preds = MODEL.predict(X_input, verbose=0)

    temp, wind, rain_lvl, wind_lvl = preds

    temp = temp[0]
    wind = wind[0]
    rain_lvl = rain_lvl[0]
    wind_lvl = wind_lvl[0]

    rain_class = np.argmax(rain_lvl, axis=-1)
    wind_class = np.argmax(wind_lvl, axis=-1)

    out = []
    for i in range(HORIZON):
        d = future_dates[i]
        hi = float(temp[i])

        out.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "day_name": _vn_day_name(d.weekday()),
                "heat_index": hi,           
                "temp_avg": hi,            
                "wind_speed": float(wind[i]),
                "rain_level": int(rain_class[i]),
                "wind_level": int(wind_class[i]),
            }
        )
    return out

def run_prediction(
    city_code: str,
    time_option: Optional[str] = None,
    timeframe_value: Optional[int] = None,
    end_date: Optional[str] = None,
):
    if end_date is None or end_date == "":
        end_date = datetime.today().strftime("%Y-%m-%d")

    info = _get_city_meta(city_code)
    city_name = info["city_name"]
    region = info["region"]

    future_dates = _build_future_dates_from_end(end_date, HORIZON)
    X_input = _build_input_window(city_name, end_date)
    forecast = _run_model_and_decode(X_input, future_dates)

    rain_levels = [d["rain_level"] for d in forecast]
    wind_levels = [d["wind_level"] for d in forecast]
    print(f"[DEBUG] {city_name} - rain_levels 7 days forward: {rain_levels}")
    print(f"[DEBUG] {city_name} - wind_levels 7 days forward: {wind_levels}")

    events = detect_events(forecast)
    summary = build_summary(city_name, end_date, forecast, events)

    return {
        "city": city_name,
        "region": region,
        "selected_date": end_date,
        "horizon_days": HORIZON,
        "forecast": forecast,
        "summary": summary,
        "events": events,
    }
