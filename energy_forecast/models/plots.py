"""
plots.py
--------
Visualization functions for energy commodity analysis and forecasting.
All price axes are labelled with the appropriate currency and unit (USD).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Color palette per commodity
PALETTE = {
    "WTI_Oil":     "#e63946",
    "Brent_Oil":   "#f4a261",
    "Natural_Gas": "#2a9d8f",
    "Coal":        "#264653",
    "Electricity": "#457b9d",
}

# Y-axis unit labels (currency + unit)
UNITS = {
    "WTI_Oil":     "Price (USD / barrel)",
    "Brent_Oil":   "Price (USD / barrel)",
    "Natural_Gas": "Price (USD / MMBtu)",
    "Coal":        "Price (USD / share)",
    "Electricity": "Price (USD / share)",
}

# Full display names
DISPLAY_NAMES = {
    "WTI_Oil":     "WTI Crude Oil",
    "Brent_Oil":   "Brent Crude Oil",
    "Natural_Gas": "Natural Gas (Henry Hub)",
    "Coal":        "Coal (KOL ETF)",
    "Electricity": "Electricity (ICLN ETF)",
}


# ---------------------------------------------------------------------------

def plot_all_series(df: pd.DataFrame) -> None:
    """Plot all commodity time series, each in its own subplot."""
    fig, axes = plt.subplots(len(df.columns), 1,
                             figsize=(14, 3 * len(df.columns)), sharex=True)
    if len(df.columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        color = PALETTE.get(col, "steelblue")
        unit  = UNITS.get(col, "Price (USD)")
        ax.plot(df.index, df[col], color=color, lw=1.2)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(DISPLAY_NAMES.get(col, col), fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.fill_between(df.index, df[col], alpha=0.08, color=color)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle("Historical Prices — Energy Commodities (USD)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  OK All-series chart saved.")


def plot_correlation(df: pd.DataFrame) -> None:
    """Pearson correlation matrix of daily returns."""
    corr = df.pct_change().dropna().corr()
    labels = [DISPLAY_NAMES.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Correlation of Daily Returns (USD-denominated)", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation.png", dpi=150)
    plt.close()
    print("  OK Correlation matrix saved.")


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> None:
    """Side-by-side ACF and PACF plots."""
    unit = UNITS.get(series.name, "USD")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf( series.dropna(), lags=lags, ax=ax1,
              title=f"ACF — {DISPLAY_NAMES.get(series.name, series.name)}")
    plot_pacf(series.dropna(), lags=lags, ax=ax2,
              title=f"PACF — {DISPLAY_NAMES.get(series.name, series.name)}", method="ywm")
    for ax in (ax1, ax2):
        ax.set_xlabel("Lag (trading days)", fontsize=9)
    fig.text(0.01, 0.5, "Autocorrelation", va="center", rotation="vertical", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{series.name}_acf_pacf.png", dpi=150)
    plt.close()
    print(f"  OK ACF/PACF chart for {series.name} saved.")


def plot_forecast(
    history:  pd.Series,
    forecast: pd.DataFrame,
    name:     str = "",
    n_hist:   int = 120,
) -> None:
    """
    Plot the last `n_hist` trading days of history + future forecast with CI.

    forecast must have columns: forecast, lower_ci, upper_ci  (all in USD)
    """
    hist_tail    = history.dropna().iloc[-n_hist:]
    color        = PALETTE.get(name, "steelblue")
    unit         = UNITS.get(name, "Price (USD)")
    display_name = DISPLAY_NAMES.get(name, name)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_tail.index, hist_tail.values,
            color=color, lw=1.5, label="Historical price")
    ax.plot(forecast.index, forecast["forecast"],
            color="black", lw=2, ls="--", label="Forecast")
    ax.fill_between(
        forecast.index,
        forecast["lower_ci"],
        forecast["upper_ci"],
        alpha=0.25, color="grey", label="95% Confidence Interval"
    )
    ax.axvline(history.index[-1], color="red", ls=":", lw=1, label="Today")

    # Annotate last forecast value
    last_val = forecast["forecast"].iloc[-1]
    ax.annotate(f"${last_val:.2f}",
                xy=(forecast.index[-1], last_val),
                xytext=(10, 0), textcoords="offset points",
                fontsize=9, color="black")

    ax.set_title(f"ARIMA Price Forecast — {display_name}", fontsize=13)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_forecast.png", dpi=150)
    plt.close()
    print(f"  OK Forecast chart for {name} saved.")


def plot_validation(val: dict, name: str = "") -> None:
    """Plot walk-forward predictions vs actual values."""
    actual       = val["actual"]
    preds        = val["predictions"]
    color        = PALETTE.get(name, "steelblue")
    unit         = UNITS.get(name, "Price (USD)")
    display_name = DISPLAY_NAMES.get(name, name)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(actual.index, actual.values, color=color, lw=1.5, label="Actual price")
    ax.plot(preds.index,  preds.values,  color="black", lw=1.5, ls="--", label="Predicted price")
    ax.set_title(
        f"Walk-Forward Validation — {display_name}\n"
        f"MAE = {val['MAE']:.3f} | RMSE = {val['RMSE']:.3f} | MAPE = {val['MAPE']:.2f}%"
        f"  (all errors in USD)",
        fontsize=11,
    )
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_validation.png", dpi=150)
    plt.close()
    print(f"  OK Validation chart for {name} saved.")
