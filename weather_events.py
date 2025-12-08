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
    """
    Return (max_len, start_idx, end_idx) for the longest consecutive run
    in a sorted list of indices.
    """
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
    """
    Detect key weather events from a 7-day forecast.
    Each forecast item should contain:
      - temp_avg
      - rain_level (0..4)
      - wind_level (0..4)
    """

    temps = [d["temp_avg"] for d in forecast]
    rains = [d["rain_level"] for d in forecast]
    winds = [d["wind_level"] for d in forecast]

    # Thresholds (can be tuned later)
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

    # Heatwave detection
    hot_indices = [i for i, t in enumerate(temps) if t >= HEATWAVE_TEMP]
    hw_len, hw_start, hw_end = _longest_run(hot_indices)
    has_heatwave = hw_len >= HEATWAVE_MIN_DAYS

    # Hot and dry period
    avg_temp = float(np.mean(temps))
    max_rain = max(rains) if rains else 0
    has_hot_dry = (avg_temp >= HOT_DRY_TEMP) and (max_rain <= HOT_DRY_MAX_RAIN)

    # Comfortable weather
    has_comfort = (
        COMFORT_MIN_TEMP <= avg_temp <= COMFORT_MAX_TEMP
        and max_rain <= COMFORT_MAX_RAIN
    )

    # Long rainy spell (moderate or heavier)
    wet_indices = [i for i, r in enumerate(rains) if r >= LONG_RAIN_MIN_LEVEL]
    lr_len, lr_start, lr_end = _longest_run(wet_indices)
    has_long_rain = lr_len >= LONG_RAIN_MIN_LEN

    # Heavy rain days
    heavy_rain_days = [i for i, r in enumerate(rains) if r >= HEAVY_RAIN_LEVEL]
    has_heavy_rain = len(heavy_rain_days) > 0

    # Showers only (light rain, no long rain or heavy rain)
    has_showers = (max_rain == 1) and not has_heavy_rain and not has_long_rain

    # Strong wind days
    strong_wind_days = [i for i, w in enumerate(winds) if w >= STRONG_WIND_LEVEL]
    has_strong_wind = len(strong_wind_days) > 0

    # Thunderstorms: rain >= 2 and wind >= 2
    thunder_days = [
        i for i in range(len(forecast))
        if rains[i] >= 2 and winds[i] >= 2
    ]
    has_thunder = len(thunder_days) > 0

    # Storm-like: heavy rain and strong wind at the same time
    storm_days = [
        i for i in range(len(forecast))
        if rains[i] >= HEAVY_RAIN_LEVEL and winds[i] >= STRONG_WIND_LEVEL
    ]
    has_storm_risk = len(storm_days) > 0

    # Urban flood risk: long rainy spell or at least two heavy-rain days
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
    """
    Build a one-paragraph natural-language summary.
    Logic:
      - First sentence: average temperature, feel-like temperature and wind.
      - Then add impact based mainly on feel-like temperature.
      - After that, refine with rain/wind events (showers, heavy rain, storm...), 
        nhưng chỉ ở mức phù hợp với ngữ cảnh (ví dụ: mưa rào nhưng trời vẫn nóng).
    """

    # Basic averages
    avg_temp = float(np.mean([d["temp_avg"] for d in forecast]))

    bias = _get_bias(city_name)
    heat_vals = [
        d.get("heat_index", d["temp_avg"]) + bias
        for d in forecast
    ]
    avg_feel = float(np.mean(heat_vals))

    avg_wind = float(np.mean([d["wind_speed"] for d in forecast]))

    # Base sentence
    text = (
        f"Dự đoán cho 7 ngày sau {end_date}, {city_name} có nhiệt độ trung bình khoảng {avg_temp:.1f}°C, "
        f"nhiệt độ cảm nhận thực tế khoảng {avg_feel:.1f}°C, sức gió trung bình khoảng {avg_wind:.1f} km/h."
    )

    # Impact mainly from temperature
    if avg_feel >= 35:
        text += (
            " Nhiệt độ cảm nhận ở mức rất cao, thời tiết oi bức rõ rệt, dễ gây mệt mỏi và khó chịu, "
            "tăng nguy cơ say nắng, mất nước nếu người dân phải làm việc hoặc di chuyển ngoài trời trong thời gian dài."
        )
    elif avg_feel >= 30:
        text += (
            " Nhiệt độ cảm nhận khá cao, thời tiết nóng và oi, có thể gây khó chịu, "
            "làm giảm sự thoải mái khi đi lại, sinh hoạt ngoài trời và khiến nhu cầu sử dụng điều hòa, quạt tăng lên."
        )
    elif avg_feel >= 24:
        text += (
            " Nhiệt độ cảm nhận ở mức tương đối dễ chịu, nhìn chung phù hợp cho các hoạt động sinh hoạt, "
            "làm việc và vui chơi ngoài trời nếu không có mưa lớn hoặc gió mạnh đi kèm."
        )
    else:
        text += (
            " Nhiệt độ cảm nhận khá mát đến hơi lạnh, người dân nên chuẩn bị trang phục đủ ấm "
            "khi ra ngoài vào buổi sáng sớm hoặc tối muộn."
        )

    # Now refine with rain/wind events (on top of temperature)
    # Priority: storm / flood > long/heavy rain > showers

    if events["storm_risk"]["has_event"]:
        idx = events["storm_risk"]["days"][0]
        d = forecast[idx]["date"]
        text += (
            f" Ngoài ra, mô hình cho thấy khả năng xuất hiện mưa to kèm gió mạnh vào khoảng ngày {d}, "
            "dễ gây ngập lụt nhanh trên đường phố, ảnh hưởng mạnh đến việc di chuyển, buôn bán và các hoạt động ngoài trời."
        )
    elif events["urban_flood_risk"]["has_event"]:
        if events["long_rain"]["has_event"]:
            s = events["long_rain"]["start_idx"]
            e = events["long_rain"]["end_idx"]
            d1 = forecast[s]["date"]
            d2 = forecast[e]["date"]
            text += (
                f" Một đợt mưa vừa đến mưa to kéo dài từ khoảng ngày {d1} đến {d2} có thể gây ngập úng cục bộ, "
                "làm tăng nguy cơ kẹt xe, gián đoạn hoạt động kinh doanh nhỏ và khiến sinh hoạt hằng ngày trở nên bất tiện."
            )
        else:
            text += (
                " Lượng mưa tích lũy trong nhiều ngày tương đối lớn, làm tăng khả năng ngập úng ở những khu vực trũng thấp, "
                "người dân nên chủ động theo dõi tình hình để điều chỉnh kế hoạch đi lại."
            )
    elif events["long_rain"]["has_event"]:
        s = events["long_rain"]["start_idx"]
        e = events["long_rain"]["end_idx"]
        d1 = forecast[s]["date"]
        d2 = forecast[e]["date"]
        text += (
            f" Dự báo có một đợt mưa kéo dài từ khoảng ngày {d1} đến {d2}, "
            "có thể làm đường sá trơn trượt, giảm tầm nhìn khi di chuyển và khiến các hoạt động vui chơi, buôn bán ngoài trời bị hạn chế."
        )
    elif events["heavy_rain"]["has_event"]:
        d = forecast[events["heavy_rain"]["days"][0]]["date"]
        text += (
            f" Một số thời điểm có thể xuất hiện mưa to vào khoảng ngày {d}, "
            "gây nguy cơ ngập cục bộ tại các điểm trũng và làm chậm trễ lịch trình đi lại, giao nhận hàng hóa."
        )
    elif events["showers"]["has_event"]:
        # This branch will hit trường hợp bạn chụp: mưa level 1, nhưng nhiệt độ vẫn ~32°C
        text += (
            " Thỉnh thoảng có thể xuất hiện mưa rào, mưa vừa rải rác nhưng không kéo dài; "
            "những cơn mưa này chủ yếu làm dịu bớt cảm giác nóng trong thời gian ngắn, "
            "song cũng có thể gây gián đoạn tạm thời cho các hoạt động ngoài trời như đi dạo, thể dục hoặc mua sắm."
        )

    # Thunderstorm and strong wind as extra details
    if events["thunderstorm"]["has_event"] and not events["storm_risk"]["has_event"]:
        d = forecast[events["thunderstorm"]["days"][0]]["date"]
        text += (
            f" Một số thời điểm, đặc biệt khoảng ngày {d}, có thể xuất hiện giông sét, "
            "người dân nên hạn chế đứng dưới cây cao hoặc ở khu vực trống trải để đảm bảo an toàn."
        )

    if events["strong_wind"]["has_event"] and not events["storm_risk"]["has_event"]:
        d = forecast[events["strong_wind"]["days"][0]]["date"]
        text += (
            f" Gió đôi lúc thổi khá mạnh vào khoảng ngày {d}, "
            "có thể gây khó khăn cho việc di chuyển bằng xe máy hoặc tàu thuyền nhỏ, "
            "người dân nên chú ý giữ thăng bằng và cố định các vật dụng dễ bay."
        )

    # If there is absolutely nothing remarkable besides the base temperature
    if (
        not events["storm_risk"]["has_event"]
        and not events["urban_flood_risk"]["has_event"]
        and not events["long_rain"]["has_event"]
        and not events["heavy_rain"]["has_event"]
        and not events["showers"]["has_event"]
        and not events["thunderstorm"]["has_event"]
        and not events["strong_wind"]["has_event"]
    ):
        text += (
            " Không có dấu hiệu rõ rệt của các hiện tượng thời tiết cực đoan, "
            "tác động bất lợi đến sinh hoạt hằng ngày và sức khỏe người dân được đánh giá ở mức thấp."
        )

    return text
