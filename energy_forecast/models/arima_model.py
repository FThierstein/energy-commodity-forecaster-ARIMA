"""
arima_model.py
--------------
Fits an ARIMA or SARIMA model to a commodity price time series
and generates forecasts with confidence intervals.

Features
--------
* auto_arima            - automatically selects best (p,d,q)(P,D,Q,s) parameters
* fit                   - trains the model on the provided data
* forecast              - produces N-step ahead forecasts
* diagnostics           - plots residuals and statistical tests
* walk_forward_validate - validates model with expanding window (walk-forward)

All prices are in USD. Units depend on the commodity:
  WTI / Brent Oil : USD per barrel
  Natural Gas     : USD per MMBtu
  Coal / Electricity: USD per share (ETF proxy)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

warnings.filterwarnings("ignore")

# Units label per commodity (used in chart annotations)
UNITS = {
    "WTI_Oil":     "USD/barrel",
    "Brent_Oil":   "USD/barrel",
    "Natural_Gas": "USD/MMBtu",
    "Coal":        "USD/share",
    "Electricity": "USD/share",
}


# ---------------------------------------------------------------------------
# Stationarity helpers
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: unit root (non-stationary). p < 0.05 => stationary.
    """
    result = adfuller(series.dropna())
    return {"statistic": result[0], "p_value": result[1], "stationary": result[1] < 0.05}


def kpss_test(series: pd.Series) -> dict:
    """
    KPSS test.
    H0: stationary. p > 0.05 => stationary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    return {"statistic": stat, "p_value": p, "stationary": p > 0.05}


def plot_stationarity(series: pd.Series, window: int = 30) -> None:
    """Plot the series with rolling mean and standard deviation."""
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    unit = UNITS.get(series.name, "USD")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series,       label="Original series", color="steelblue", lw=1)
    ax.plot(rolling_mean, label=f"{window}-day rolling mean", color="orange",   lw=2)
    ax.plot(rolling_std,  label=f"{window}-day rolling std",  color="red",      lw=1.5, ls="--")
    ax.set_title(f"Stationarity Check — {series.name}", fontsize=13)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{series.name}_stationarity.png", dpi=150)
    plt.close()
    print(f"  OK Stationarity chart saved.")


# ---------------------------------------------------------------------------
# Main forecaster class
# ---------------------------------------------------------------------------

class ArimaForecaster:
    """
    Wraps pmdarima.auto_arima + statsmodels SARIMAX.

    Parameters
    ----------
    seasonal        : bool – include seasonal component?
    seasonal_period : int  – periodicity (5=weekly, 12=monthly, 252=annual trading days)
    """

    def __init__(self, seasonal: bool = False, seasonal_period: int = 5):
        self.seasonal        = seasonal
        self.m               = seasonal_period
        self.order_          = None   # (p, d, q)
        self.seasonal_order_ = None   # (P, D, Q, s)
        self.model_          = None   # fitted SARIMAX result
        self.series_         = None

    # -- Automatic order selection -------------------------------------------

    def auto_select_order(self, series: pd.Series, max_p: int = 5, max_q: int = 5) -> None:
        """Use pmdarima.auto_arima to find the best AIC order."""
        print(f"  Auto-ARIMA for {series.name}...  (may take a few seconds)")
        model = pm.auto_arima(
            series.dropna(),
            start_p=1, start_q=1,
            max_p=max_p, max_q=max_q,
            d=None,                      # auto-detect integration order
            seasonal=self.seasonal,
            m=self.m if self.seasonal else 1,
            stepwise=True,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
        )
        self.order_          = model.order
        self.seasonal_order_ = model.seasonal_order if self.seasonal else (0, 0, 0, 0)
        print(f"    Best model: ARIMA{self.order_}  seasonal{self.seasonal_order_}  "
              f"AIC={model.aic():.1f}")

    # -- Training ------------------------------------------------------------

    def fit(self, series: pd.Series) -> "ArimaForecaster":
        """Train the SARIMAX model with the already-selected orders."""
        if self.order_ is None:
            self.auto_select_order(series)
        self.series_ = series.dropna().asfreq("B").ffill()   # business-day frequency
        m = SARIMAX(
            self.series_,
            order=self.order_,
            seasonal_order=self.seasonal_order_,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_ = m.fit(disp=False)
        print(f"  OK Model trained on {len(self.series_)} observations.")
        return self

    # -- Forecasting ---------------------------------------------------------

    def forecast(self, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generate forecast for `steps` business days ahead.

        Returns DataFrame with columns: forecast, lower_ci, upper_ci
        All values in USD (same unit as the input series).
        """
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .forecast().")
        pred    = self.model_.get_forecast(steps=steps)
        summary = pred.summary_frame(alpha=alpha)
        result  = pd.DataFrame({
            "forecast": summary["mean"],
            "lower_ci": summary["mean_ci_lower"],
            "upper_ci": summary["mean_ci_upper"],
        })
        return result

    # -- Diagnostics ---------------------------------------------------------

    def diagnostics(self, name: str = "") -> None:
        """Plot 4-panel residual diagnostics chart."""
        if self.model_ is None:
            raise RuntimeError("Call .fit() first.")
        fig = self.model_.plot_diagnostics(figsize=(14, 8))
        fig.suptitle(f"Residual Diagnostics — {name or self.series_.name}", fontsize=13)
        plt.tight_layout()
        tag = name or (self.series_.name if self.series_.name else "model")
        plt.savefig(f"results/{tag}_diagnostics.png", dpi=150)
        plt.close()
        print(f"  OK Diagnostics chart saved.")

    # -- Walk-forward validation ---------------------------------------------

    def walk_forward_validate(
        self,
        series: pd.Series,
        test_size: int = 60,
        step: int = 1,
    ) -> dict:
        """
        Validate the model using an expanding window (walk-forward).

        Parameters
        ----------
        series    : full price series (USD)
        test_size : number of observations in the test set
        step      : forecast horizon per iteration (1 = 1 business day)

        Returns
        -------
        dict with MAE (USD), RMSE (USD), MAPE (%), and prediction series
        """
        series = series.dropna()
        train  = series[:-test_size]
        test   = series[-test_size:]
        preds  = []

        print(f"  Walk-forward validation: {test_size} steps...")
        history = list(train)
        for i, obs in enumerate(test):
            m = SARIMAX(
                history,
                order=self.order_,
                seasonal_order=self.seasonal_order_,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            yhat = m.forecast(steps=step)[0]
            preds.append(yhat)
            history.append(obs)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{test_size}")

        preds  = np.array(preds)
        actual = test.values
        mae    = mean_absolute_error(actual, preds)
        rmse   = np.sqrt(mean_squared_error(actual, preds))
        mape   = np.mean(np.abs((actual - preds) / actual)) * 100

        unit = UNITS.get(series.name, "USD")
        print(f"  OK  MAE={mae:.4f} {unit}  RMSE={rmse:.4f} {unit}  MAPE={mape:.2f}%")
        return {
            "MAE":  mae,  "RMSE": rmse,  "MAPE": mape,
            "predictions": pd.Series(preds, index=test.index, name="predicted"),
            "actual":      test,
        }
