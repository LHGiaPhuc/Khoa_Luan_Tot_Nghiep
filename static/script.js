const CITIES = [
  { code: "hanoi", label: "Hà Nội" },
  { code: "haiphong", label: "Hải Phòng" },
  { code: "quangninh", label: "Quảng Ninh" },
  { code: "thanhhoa", label: "Thanh Hóa" },

  { code: "vinh", label: "Vinh (Nghệ An)" },
  { code: "hue", label: "Huế (Thừa Thiên Huế)" },

  { code: "danang", label: "Đà Nẵng" },
  { code: "quynhon", label: "Quy Nhơn (Bình Định)" },
  { code: "nhatrang", label: "Nha Trang (Khánh Hòa)" },
  { code: "quangnam", label: "Quảng Nam (Tam Kỳ)" },

  { code: "dalat", label: "Đà Lạt (Lâm Đồng)" },
  { code: "buonmethuot", label: "Buôn Ma Thuột (Đắk Lắk)" },

  { code: "hcmc", label: "TP. Hồ Chí Minh" },
  { code: "cantho", label: "Cần Thơ" },
  { code: "camau", label: "Cà Mau" },
];

const CITY_TEMP_BIAS = {
  hanoi: 2,
  haiphong: 2,
  quangninh: 2,
  thanhhoa: 2,

  vinh: 2,
  hue: 2,

  danang: 3,
  quynhon: 3,
  nhatrang: 4,
  quangnam: 3,

  dalat: 1,
  buonmethuot: 2,

  hcmc: 4,
  cantho: 3,
  camau: 3,
};

function getBias(cityCode) {
  return CITY_TEMP_BIAS[cityCode] ?? 3;
}

const citySelector = document.getElementById("city-selector");
const datePicker = document.getElementById("date-picker");
const dailyList = document.getElementById("daily-list");
const summaryBox = document.getElementById("weather-summary");

const tempChartCanvas = document.getElementById("temp-chart");
let tempChart = null;

function initCityDropdown() {
  CITIES.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.code;
    opt.textContent = c.label;
    if (c.code === "hcmc") opt.selected = true;
    citySelector.appendChild(opt);
  });
}

function initDatePicker() {
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  datePicker.value = `${yyyy}-${mm}-${dd}`;
}

function mainWeatherIcon(d) {
  const currentCity = citySelector.value;
  const bias = getBias(currentCity);

  const t = d.heat_index + bias;
  const rain = d.rain_level ?? 0;
  const wind = d.wind_level ?? 0;

  if (rain >= 3 && wind >= 2) return "⛈️";

  if (rain >= 2) {
    return "🌧️";
  }

  if (t >= 37) return "🥵";
  if (t >= 33) return "🌞";
  if (t >= 26) return "🌤️";
  if (t >= 20) return "☁️";
  return "🌫️";
}

function windText(level) {
  switch (level) {
    case 0:
      return "Gió yếu";
    case 1:
      return "Gió nhẹ";
    case 2:
      return "Gió vừa";
    case 3:
      return "Gió mạnh";
    default:
      return "Gió rất mạnh";
  }
}

function updateBackground() {
  const hour = new Date().getHours();
  const body = document.body;

  body.classList.remove("morning-bg", "noon-bg", "sunset-bg", "night-bg");

  if (hour >= 5 && hour < 11) body.classList.add("morning-bg");
  else if (hour >= 11 && hour < 16) body.classList.add("noon-bg");
  else if (hour >= 16 && hour < 19) body.classList.add("sunset-bg");
  else body.classList.add("night-bg");
}

async function fetchForecast() {
  const city = citySelector.value;
  const endDateRaw = datePicker.value;

  const payload = {
    city: city,
    end_date: endDateRaw === "" ? null : endDateRaw,
  };

  summaryBox.textContent = "Đang tải dự báo...";
  dailyList.innerHTML = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error("Lỗi API:", err);
      summaryBox.textContent = `Bad Request (${response.status})`;
      return;
    }

    const data = await response.json();
    renderForecast(data);
  } catch (err) {
    console.error("Lỗi kết nối:", err);
    summaryBox.textContent = "Không thể kết nối server.";
  }
}

function renderTempChart(forecast) {
  if (!tempChartCanvas || typeof Chart === "undefined") return;

  const ctx = tempChartCanvas.getContext("2d");
  const currentCity = citySelector.value;
  const bias = getBias(currentCity);

  const labels = forecast.map((d) => d.day_name);

  const temps = forecast.map((d) => {
    const base =
      typeof d.heat_index === "number" && !Number.isNaN(d.heat_index)
        ? d.heat_index
        : d.temp_avg;
    return base + bias;
  });

  const lineGradient = ctx.createLinearGradient(0, 0, tempChartCanvas.width, 0);
  lineGradient.addColorStop(0, "#4da0ff");
  lineGradient.addColorStop(0.5, "#ffd85b");
  lineGradient.addColorStop(1, "#ff5959");

  const fillGradient = ctx.createLinearGradient(
    0,
    0,
    0,
    tempChartCanvas.height
  );
  fillGradient.addColorStop(0, "rgba(255, 217, 102, 0.35)");
  fillGradient.addColorStop(1, "rgba(0,0,0,0)");

  if (tempChart) {
    tempChart.data.labels = labels;
    tempChart.data.datasets[0].data = temps;
    tempChart.data.datasets[0].borderColor = lineGradient;
    tempChart.data.datasets[0].backgroundColor = fillGradient;
    tempChart.update();
    return;
  }

  tempChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Nhiệt độ cảm nhận (°C)",
          data: temps,
          borderColor: lineGradient,
          backgroundColor: fillGradient,
          borderWidth: 3,
          tension: 0.35,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: "#ffffff",
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              const i = ctx.dataIndex;
              const d = forecast[i];
              const bias = getBias(citySelector.value);

              return [
                `🌡 Nhiệt độ: ${(d.heat_index + bias).toFixed(1)}°C`,
                `🌧 Mức mưa: ${d.rain_level}`,
                `💨 Gió: ${windText(d.wind_level)}`,
                `⛅ Trạng thái: ${mainWeatherIcon(d)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#ffffff" },
          grid: { color: "rgba(255,255,255,0.08)" },
        },
        y: {
          ticks: { color: "#ffffff" },
          grid: { color: "rgba(255,255,255,0.15)" },
        },
      },
    },
  });
}

function renderForecast(data) {
  const { forecast, summary } = data;

  dailyList.innerHTML = "";

  const currentCity = citySelector.value;
  const bias = getBias(currentCity);

  forecast.forEach((d) => {
    const item = document.createElement("div");
    item.className = "daily-item";

    item.innerHTML = `
      <div class="day-name">${d.day_name}</div>
      <div class="weather-icon">${mainWeatherIcon(d)}</div>
      <div class="temps">
        <span class="temp-high">${(d.heat_index + bias).toFixed(1)}°C</span>
      </div>
    `;

    dailyList.appendChild(item);
  });

  renderTempChart(forecast);

  summaryBox.textContent = summary;
}

initCityDropdown();
initDatePicker();
updateBackground();
fetchForecast();

citySelector.addEventListener("change", fetchForecast);
datePicker.addEventListener("change", fetchForecast);
