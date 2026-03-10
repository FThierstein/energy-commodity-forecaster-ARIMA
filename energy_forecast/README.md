# 🛢️ Energy Commodity Price Forecaster

Python project for forecasting energy commodity prices using **ARIMA/SARIMA** time series models.

> Part of a multi-model forecasting portfolio. Future modules will cover LSTM, XGBoost, Prophet, and hybrid approaches.

---

## 📁 Project Structure

```
energy_forecast/
├── main.py                      ← Main pipeline (entry point)
├── requirements.txt             ← Python dependencies
├── data/
│   ├── fetcher.py               ← Data download and caching via yfinance
│   └── commodities.csv          ← Auto-generated price cache
├── models/
│   ├── arima_model.py           ← ArimaForecaster class + stationarity tests
│   ├── plots.py                 ← All visualisation functions (USD-labelled)
│   └── report_generator.py      ← Auto-generated PDF report per commodity
└── results/                     ← Auto-generated charts, CSVs and PDF reports
    ├── all_series.png
    ├── correlation.png
    ├── <commodity>_acf_pacf.png
    ├── <commodity>_stationarity.png
    ├── <commodity>_diagnostics.png
    ├── <commodity>_validation.png
    ├── <commodity>_forecast.png
    ├── <commodity>_forecast.csv
    ├── <commodity>_report.pdf
    └── summary.csv
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/FThierstein/energy-commodity-forecaster-ARIMA.git
cd energy-commodity-forecaster-ARIMA

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Basic run (30-day forecast)
```bash
python main.py
```

### Available options
```bash
python main.py --forecast-days 60   # forecast 60 trading days ahead
python main.py --test-size 90       # use 90 observations in the test set
python main.py --seasonal           # use SARIMA (weekly seasonal component)

# Combine options
python main.py --forecast-days 30 --test-size 60 --seasonal
```

---

## 📊 Commodities Covered

| Name         | Ticker       | Description                          | Unit          |
|--------------|--------------|--------------------------------------|---------------|
| WTI_Oil      | CL=F / USO   | WTI Crude Oil (West Texas)           | USD / barrel  |
| Brent_Oil    | BZ=F / BNO   | Brent Crude Oil (European benchmark) | USD / barrel  |
| Natural_Gas  | NG=F / UNG   | Natural Gas (Henry Hub)              | USD / MMBtu   |
| Coal         | KOL / ARCH   | Coal (VanEck Coal ETF)               | USD / share   |
| Electricity  | ICLN / FSLR  | Electricity (iShares Clean Energy)   | USD / share   |

> All prices are denominated in **US Dollars (USD)**. Futures tickers are tried first; ETF tickers are used as fallback.

---

## 🔬 Pipeline Overview

### 1. Data Download
- Source: **Yahoo Finance** via `yfinance`
- History from 2015 to present (~10 years)
- Automatic cache in `data/commodities.csv` (avoids repeated downloads)

### 2. Exploratory Analysis
- Individual time series charts (USD, with units)
- Pearson correlation matrix of daily returns

### 3. Per Commodity
1. **ACF / PACF** — visual identification of autocorrelation structure
2. **Stationarity tests**
   - Augmented Dickey-Fuller (ADF): H₀ = unit root → *p < 0.05* ⟹ stationary
   - KPSS: H₀ = stationary → *p > 0.05* ⟹ stationary
3. **Auto-ARIMA** — `pmdarima.auto_arima` selects (p,d,q) by AIC minimisation
4. **SARIMAX training** — via `statsmodels`
5. **Residual diagnostics** — 4-panel chart
6. **Walk-forward validation** — 1-step expanding window
   - Metrics: MAE (USD), RMSE (USD), MAPE (%)
7. **Forecast** — N trading days ahead with 95% confidence interval
8. **PDF report** — auto-generated per commodity via `reportlab`

### 4. Summary
- `results/summary.csv` with metrics and final forecast value per commodity

---

## 📈 Example Output (summary.csv)

| Commodity   | Model        | MAE (USD) | RMSE (USD) | MAPE (%) | Forecast_30d (USD) |
|-------------|--------------|-----------|------------|----------|--------------------|
| WTI_Oil     | ARIMA(1,1,1) | 1.23      | 1.87       | 1.54     | 78.32              |
| Natural_Gas | ARIMA(2,1,2) | 0.08      | 0.11       | 2.31     | 2.64               |
| ...         | ...          | ...       | ...        | ...      | ...                |

---

## 🧩 Suggested Extensions

| Idea                        | How to implement                                         |
|-----------------------------|----------------------------------------------------------|
| Exogenous features          | Use SARIMAX with USD index, temperature, inventory data  |
| Compare with LSTM           | Add `models/lstm_model.py`                               |
| Interactive dashboard       | Integrate with `streamlit` or `dash`                     |
| Price alerts                | Add email / Telegram notifications                       |
| Automated deployment        | Schedule with `cron` or GitHub Actions                   |

---

## 📝 Notes

- Data consists of **futures / ETF closing prices** and does **not** constitute financial advice.
- ARIMA performs best over **short horizons** (≤ 30 days). For longer horizons, consider hybrid models.
- First run may take 2–5 minutes (data download + auto_arima order selection).
