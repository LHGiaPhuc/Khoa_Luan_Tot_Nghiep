from typing import List, Dict, Any
import numpy as np

CITY_TEMP_BIAS_BY_NAME: Dict[str, float] = {
    "Hanoi": 2,
    "Hai Phong": 2,
    "Quang Ninh": 2,
    "Thanh Hoa": 2,

    "Nghe An (Vinh)": 2,
    "Hue (Thua Thien Hue)": 2,

    "Da Nang": 3,
    "Binh Dinh (Quy Nhon)": 3,
    "Nha Trang (Khanh Hoa)": 4,
    "Quang Nam (Tam Ky)": 3,

    "Da Lat (Lam Dong)": 1,
    "Buon Ma Thuot (Dak Lak)": 2,

    "Ho Chi Minh City": 4,
    "Can Tho": 3,
    "Ca Mau": 3,
}

def _get_bias(city_name: str) -> float:
    return CITY_TEMP_BIAS_BY_NAME.get(city_name, 3.0)

def _longest_run(indices: List[int]) -> (int, int, int):
    if not indices:
        return 0, -1, -1

    max_len = 1
    cur_len = 1
    start = indices[0]
    best_start = start
    best_end = start

    for prev, cur in zip(indices, indices[1:]):
        if cur == prev + 1:
            cur_len += 1
        else:
            if cur_len > max_len:
                max_len = cur_len
                best_start = start
                best_end = prev
            cur_len = 1
            start = cur

    if cur_len > max_len:
        max_len = cur_len
        best_start = start
        best_end = start + cur_len - 1

    return max_len, best_start, best_end


def detect_events(forecast: List[Dict[str, Any]]) -> Dict[str, Any]:

    temps = [d["temp_avg"] for d in forecast]
    rains = [d["rain_level"] for d in forecast]
    winds = [d["wind_level"] for d in forecast]

    HEATWAVE_TEMP = 35.0         
    HEATWAVE_MIN_DAYS = 3

    HOT_DRY_TEMP = 33.0          
    HOT_DRY_MAX_RAIN = 1         

    COMFORT_MIN_TEMP = 22.0      
    COMFORT_MAX_TEMP = 28.0
    COMFORT_MAX_RAIN = 1

    LONG_RAIN_MIN_LEVEL = 2
    LONG_RAIN_MIN_LEN = 3

    HEAVY_RAIN_LEVEL = 3
    STRONG_WIND_LEVEL = 2       

    hot_indices = [i for i, t in enumerate(temps) if t >= HEATWAVE_TEMP]
    hw_len, hw_start, hw_end = _longest_run(hot_indices)
    has_heatwave = hw_len >= HEATWAVE_MIN_DAYS

    avg_temp = float(np.mean(temps))
    max_rain = max(rains) if rains else 0
    has_hot_dry = (avg_temp >= HOT_DRY_TEMP) and (max_rain <= HOT_DRY_MAX_RAIN)

    has_comfort = (
        COMFORT_MIN_TEMP <= avg_temp <= COMFORT_MAX_TEMP
        and max_rain <= COMFORT_MAX_RAIN
    )

    wet_indices = [i for i, r in enumerate(rains) if r >= LONG_RAIN_MIN_LEVEL]
    lr_len, lr_start, lr_end = _longest_run(wet_indices)
    has_long_rain = lr_len >= LONG_RAIN_MIN_LEN

    heavy_rain_days = [i for i, r in enumerate(rains) if r >= HEAVY_RAIN_LEVEL]
    has_heavy_rain = len(heavy_rain_days) > 0

    has_showers = (max_rain == 1) and not has_heavy_rain and not has_long_rain


    strong_wind_days = [i for i, w in enumerate(winds) if w >= STRONG_WIND_LEVEL]
    has_strong_wind = len(strong_wind_days) > 0

    thunder_days = [
        i for i in range(len(forecast))
        if rains[i] >= 2 and winds[i] >= 2
    ]
    has_thunder = len(thunder_days) > 0

    storm_days = [
        i for i in range(len(forecast))
        if rains[i] >= HEAVY_RAIN_LEVEL and winds[i] >= STRONG_WIND_LEVEL
    ]
    has_storm_risk = len(storm_days) > 0

    urban_flood_risk = has_long_rain or (len(heavy_rain_days) >= 2)

    return {
        "heatwave": {
            "has_event": has_heatwave,
            "start_idx": hw_start,
            "end_idx": hw_end,
        },
        "hot_dry": {
            "has_event": has_hot_dry,
        },
        "comfortable": {
            "has_event": has_comfort,
        },
        "long_rain": {
            "has_event": has_long_rain,
            "start_idx": lr_start,
            "end_idx": lr_end,
        },
        "heavy_rain": {
            "has_event": has_heavy_rain,
            "days": heavy_rain_days,
        },
        "showers": {
            "has_event": has_showers,
        },
        "urban_flood_risk": {
            "has_event": urban_flood_risk,
        },
        "strong_wind": {
            "has_event": has_strong_wind,
            "days": strong_wind_days,
        },
        "thunderstorm": {
            "has_event": has_thunder,
            "days": thunder_days,
        },
        "storm_risk": {
            "has_event": has_storm_risk,
            "days": storm_days,
        },
    }

def build_summary(
    city_name: str,
    end_date: str,
    forecast: List[Dict[str, Any]],
    events: Dict[str, Any],
) -> str:
    avg_temp = float(np.mean([d["temp_avg"] for d in forecast]))

    bias = _get_bias(city_name)
    heat_vals = [
        d.get("heat_index", d["temp_avg"]) + bias
        for d in forecast
    ]
    avg_feel = float(np.mean(heat_vals))

    avg_wind = float(np.mean([d["wind_speed"] for d in forecast]))

    parts: List[str] = []
    parts.append(
        f"Dự đoán cho 7 ngày sau {end_date}, {city_name} có nhiệt độ trung bình khoảng {avg_temp:.1f}°C, "
        f"nhiệt độ cảm nhận khoảng {avg_feel:.1f}°C, "
        f"sức gió trung bình {avg_wind:.1f} km/h."
    )

    if events["storm_risk"]["has_event"]:
        idx = events["storm_risk"]["days"][0]
        d = forecast[idx]["date"]
        parts.append(
            f" Có khả năng xảy ra mưa to kèm gió mạnh (giông bão cục bộ) vào khoảng ngày {d}, "
            f"khuyến cáo nên hạn chế di chuyển ngoài trời và chú ý an toàn."
        )
    elif events["urban_flood_risk"]["has_event"]:
        if events["long_rain"]["has_event"]:
            s = events["long_rain"]["start_idx"]
            e = events["long_rain"]["end_idx"]
            d1 = forecast[s]["date"]
            d2 = forecast[e]["date"]
            parts.append(
                f" Có một đợt mưa vừa đến to kéo dài từ khoảng ngày {d1} đến {d2}, "
                f"nguy cơ gây ngập úng tại các khu vực trũng thấp."
            )
        else:
            parts.append(
                " Lượng mưa dự báo khá lớn trong nhiều ngày, có thể gây ngập úng ở đô thị."
            )
    elif events["long_rain"]["has_event"]:
        s = events["long_rain"]["start_idx"]
        e = events["long_rain"]["end_idx"]
        d1 = forecast[s]["date"]
        d2 = forecast[e]["date"]
        parts.append(
            f" Dự báo có một đợt mưa kéo dài (mưa vừa trở lên) từ khoảng ngày {d1} đến {d2}, "
            f"khuyến cáo hạn chế các hoạt động ngoài trời."
        )
    elif events["heavy_rain"]["has_event"]:
        first_heavy = events["heavy_rain"]["days"][0]
        d = forecast[first_heavy]["date"]
        parts.append(
            f" Một vài thời điểm xuất hiện mưa to đến rất to, đáng chú ý vào khoảng ngày {d}."
        )
    elif events["showers"]["has_event"]:
        parts.append(
            " Thời tiết có khả năng xuất hiện mưa rào, mưa vừa rải rác nhưng không kéo dài."
        )

    if events["thunderstorm"]["has_event"] and not events["storm_risk"]["has_event"]:
        idx = events["thunderstorm"]["days"][0]
        d = forecast[idx]["date"]
        parts.append(
            f" Có khả năng xảy ra giông kèm gió giật, đặc biệt vào khoảng ngày {d}, "
            f"cần chú ý sấm sét và đảm bảo an toàn."
        )
    elif events["strong_wind"]["has_event"] and not events["storm_risk"]["has_event"]:
        first_windy = events["strong_wind"]["days"][0]
        d = forecast[first_windy]["date"]
        parts.append(
            f" Gió có lúc mạnh, đặc biệt vào khoảng ngày {d}, cẩn nâng cao cảnh giác khi lưu thông trên đường và đi tàu thuyền nhỏ."
        )

    if events["heatwave"]["has_event"]:
        s = events["heatwave"]["start_idx"]
        e = events["heatwave"]["end_idx"]
        d1 = forecast[s]["date"]
        d2 = forecast[e]["date"]
        parts.append(
            f" Có khả năng xuất hiện đợt nắng nóng gay gắt"
            f"từ khoảng ngày {d1} đến {d2}, khuyến cáo hạn chế ở ngoài trời vào buổi trưa."
        )
    elif events["hot_dry"]["has_event"]:
        parts.append(
            " Thời tiết nhìn chung nắng nhiều, ít mưa, không khí khá khô và oi trong nhiều ngày."
        )
    elif events["comfortable"]["has_event"]:
        parts.append(
            " Nhiệt độ ở mức dễ chịu, ít mưa, thời tiết tương đối thuận lợi cho các hoạt động ngoài trời."
        )

    if (
        not events["storm_risk"]["has_event"]
        and not events["urban_flood_risk"]["has_event"]
        and not events["long_rain"]["has_event"]
        and not events["heavy_rain"]["has_event"]
        and not events["showers"]["has_event"]
        and not events["thunderstorm"]["has_event"]
        and not events["strong_wind"]["has_event"]
        and not events["heatwave"]["has_event"]
        and not events["hot_dry"]["has_event"]
        and not events["comfortable"]["has_event"]
    ):
        parts.append(
            " Thời tiết nhìn chung ổn định, không có dấu hiệu rõ rệt của các hiện tượng cực đoan trong 7 ngày tới."
        )

    return " ".join(parts)
