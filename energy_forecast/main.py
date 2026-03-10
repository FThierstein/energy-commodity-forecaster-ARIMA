"""
main.py
-------
Main pipeline for energy commodity price forecasting.

Workflow
--------
1. Download historical price data (yfinance)
2. Exploratory data analysis (time series, correlation, ACF/PACF)
3. For each commodity:
      a. Stationarity tests (ADF + KPSS)
      b. Auto-ARIMA order selection
      c. Model training
      d. Residual diagnostics
      e. Walk-forward validation
      f. Forecast for N future trading days
      g. Generate individual PDF report
4. Final summary CSV

Usage
-----
  python main.py [--forecast-days 30] [--test-size 60] [--seasonal]
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Always set working directory to the project folder (fixes VS Code cwd issue)
PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print(f"  Working directory: {PROJECT_DIR}")

from data.fetcher        import fetch, save, load
from models.arima_model  import ArimaForecaster, adf_test, kpss_test, plot_stationarity
from models.plots        import (
    plot_all_series, plot_correlation,
    plot_acf_pacf, plot_forecast, plot_validation,
)
from models.report_generator import generate_report


# ---------------------------------------------------------------------------

def run(forecast_days: int = 30, test_size: int = 60, seasonal: bool = False) -> None:

    os.makedirs("results", exist_ok=True)

    # ── 1. Data download ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Data Download")
    print("=" * 60)

    cache = Path("data/commodities.csv")
    if cache.exists():
        print("  Cache found — loading from disk...")
        df = load()
    else:
        prices = fetch(start="2015-01-01")
        save(prices)
        df = pd.DataFrame(prices)

    # Guard: abort if no data was downloaded
    if df.empty or len(df) == 0:
        print("\n  ERROR: No data was downloaded.")
        print("  Check your internet connection and try again.")
        print("  If the problem persists, Yahoo Finance tickers may be temporarily unavailable.\n")
        return

    print(f"  Period  : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Loaded  : {list(df.columns)}")
    print(f"  Shape   : {df.shape}")

    # ── 2. Exploratory analysis ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — Exploratory Data Analysis")
    print("=" * 60)

    plot_all_series(df)
    plot_correlation(df)

    # ── 3. Model each commodity ───────────────────────────────────────────────
    summary_rows = []
    report_paths = []

    for name in df.columns:
        series = df[name].dropna()

        print(f"\n{'─' * 60}")
        print(f"  COMMODITY : {name}  ({len(series)} observations)")
        print(f"{'─' * 60}")

        # ACF / PACF
        plot_acf_pacf(series)

        # Stationarity tests
        adf = adf_test(series)
        kps = kpss_test(series)
        print(f"  ADF   p={adf['p_value']:.4f}  stationary={adf['stationary']}")
        print(f"  KPSS  p={kps['p_value']:.4f}  stationary={kps['stationary']}")
        plot_stationarity(series)

        # ARIMA model
        forecaster = ArimaForecaster(seasonal=seasonal, seasonal_period=5)
        forecaster.auto_select_order(series)
        forecaster.fit(series)
        forecaster.diagnostics(name=name)

        # Walk-forward validation
        val = forecaster.walk_forward_validate(series, test_size=test_size)
        plot_validation(val, name=name)

        # Forecast
        fc = forecaster.forecast(steps=forecast_days)
        plot_forecast(series, fc, name=name)
        fc.to_csv(f"results/{name}_forecast.csv")

        # PDF report
        pdf_path = generate_report(
            name=name,
            series=series,
            adf=adf,
            kps=kps,
            forecaster=forecaster,
            val=val,
            forecast=fc,
        )
        report_paths.append(pdf_path)

        summary_rows.append({
            "Commodity":              name,
            "Model":                  f"ARIMA{forecaster.order_}",
            "MAE (USD)":              round(val["MAE"],  4),
            "RMSE (USD)":             round(val["RMSE"], 4),
            "MAPE (%)":               round(val["MAPE"], 2),
            f"Forecast_{forecast_days}d (USD)": round(fc["forecast"].iloc[-1], 4),
        })

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Summary")
    print("=" * 60)

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv("results/summary.csv", index=False)

    print("\n  OK Summary saved  -> results/summary.csv")
    print("  OK PDF reports    -> " + ", ".join(p.name for p in report_paths))
    print("  OK All charts     -> results/")
    print("\n  Pipeline complete!\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Commodity ARIMA Forecaster")
    parser.add_argument("--forecast-days", type=int,  default=30,
                        help="Forecast horizon in trading days (default: 30)")
    parser.add_argument("--test-size",     type=int,  default=60,
                        help="Number of observations in the test set (default: 60)")
    parser.add_argument("--seasonal",      action="store_true",
                        help="Use SARIMA instead of ARIMA (adds weekly seasonal component)")
    args = parser.parse_args()

    run(
        forecast_days=args.forecast_days,
        test_size=args.test_size,
        seasonal=args.seasonal,
    )
