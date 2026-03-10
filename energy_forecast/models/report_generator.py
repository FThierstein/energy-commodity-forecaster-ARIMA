"""
report_generator.py
-------------------
Generates a professional PDF report for each commodity using ReportLab.

Each report includes:
  - Cover page  : commodity name, date, ticker, currency/unit
  - Section 1   : Price history summary statistics
  - Section 2   : Stationarity test results (ADF + KPSS)
  - Section 3   : Selected ARIMA model order
  - Section 4   : Walk-forward validation metrics (MAE, RMSE, MAPE)
  - Section 5   : Forecast table (next N trading days, USD)
  - Charts      : all PNG charts embedded inline
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

RESULTS_DIR = Path("results")

# Units and display names (mirrors plots.py)
UNITS = {
    "WTI_Oil":     "USD / barrel",
    "Brent_Oil":   "USD / barrel",
    "Natural_Gas": "USD / MMBtu",
    "Coal":        "USD / share",
    "Electricity": "USD / share",
}

DISPLAY_NAMES = {
    "WTI_Oil":     "WTI Crude Oil",
    "Brent_Oil":   "Brent Crude Oil",
    "Natural_Gas": "Natural Gas (Henry Hub)",
    "Coal":        "Coal (KOL ETF)",
    "Electricity": "Electricity (ICLN ETF)",
}

TICKERS = {
    "WTI_Oil":     "CL=F / USO",
    "Brent_Oil":   "BZ=F / BNO",
    "Natural_Gas": "NG=F / UNG",
    "Coal":        "KOL / ARCH",
    "Electricity": "ICLN / FSLR",
}

ACCENT = colors.HexColor("#1a4a7a")
LIGHT   = colors.HexColor("#e8f0fa")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "CoverTitle", parent=base["Title"],
            fontSize=28, textColor=ACCENT, spaceAfter=6, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=13, textColor=colors.grey, spaceAfter=4, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=14, textColor=ACCENT, spaceBefore=14, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=11, textColor=ACCENT, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"],
            fontSize=8, textColor=colors.grey, spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=8, textColor=colors.grey, alignment=TA_RIGHT,
        ),
    }
    return styles


def _table_style(header_color=ACCENT):
    return TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])


def _embed_image(path: Path, width: float = 15 * cm) -> Image | None:
    """Return a ReportLab Image if the file exists, else None."""
    if path.exists():
        img = Image(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth  = width
        img.drawHeight = width * ratio
        return img
    return None


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_report(
    name:      str,
    series:    pd.Series,
    adf:       dict,
    kps:       dict,
    forecaster,             # ArimaForecaster instance (already fitted)
    val:       dict,
    forecast:  pd.DataFrame,
) -> Path:
    """
    Build a PDF report for one commodity and save to results/.

    Parameters
    ----------
    name       : commodity key  (e.g. "WTI_Oil")
    series     : full price series (USD)
    adf        : result dict from adf_test()
    kps        : result dict from kpss_test()
    forecaster : fitted ArimaForecaster
    val        : result dict from walk_forward_validate()
    forecast   : DataFrame from forecaster.forecast()

    Returns
    -------
    Path to the generated PDF file
    """
    out_path    = RESULTS_DIR / f"{name}_report.pdf"
    display     = DISPLAY_NAMES.get(name, name)
    unit        = UNITS.get(name, "USD")
    ticker      = TICKERS.get(name, "N/A")
    today       = datetime.today().strftime("%B %d, %Y")
    S           = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Energy Commodity", S["subtitle"]))
    story.append(Paragraph(f"Price Forecast Report", S["subtitle"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(display, S["title"]))
    story.append(HRFlowable(width="80%", thickness=2, color=ACCENT, spaceAfter=10))
    story.append(Paragraph(f"Ticker: {ticker}", S["subtitle"]))
    story.append(Paragraph(f"Currency &amp; Unit: {unit}", S["subtitle"]))
    story.append(Paragraph(f"Report generated: {today}", S["subtitle"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "This report was produced by the Energy Commodity ARIMA Forecasting pipeline. "
        "Prices are sourced from Yahoo Finance. All monetary values are in US Dollars (USD). "
        "This document is for informational purposes only and does not constitute financial advice.",
        S["small"]
    ))
    story.append(PageBreak())

    # ── 1. Price History ─────────────────────────────────────────────────────
    story.append(Paragraph("1. Price History", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))

    s = series.dropna()
    stats = {
        "Start date":          s.index[0].strftime("%Y-%m-%d"),
        "End date":            s.index[-1].strftime("%Y-%m-%d"),
        "Trading days":        f"{len(s):,}",
        "Mean price":          f"${s.mean():.2f}",
        "Median price":        f"${s.median():.2f}",
        "Minimum price":       f"${s.min():.2f}",
        "Maximum price":       f"${s.max():.2f}",
        "Std deviation":       f"${s.std():.2f}",
        "Annualised volatility": f"{s.pct_change().std() * np.sqrt(252) * 100:.1f}%",
        "Currency / Unit":     unit,
    }
    tbl_data = [["Metric", "Value"]] + [[k, v] for k, v in stats.items()]
    tbl = Table(tbl_data, colWidths=[9*cm, 7*cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    img = _embed_image(RESULTS_DIR / "all_series.png")
    if img:
        story.append(img)
        story.append(Paragraph("Figure 1 — Historical prices of all energy commodities (USD).", S["caption"]))

    img2 = _embed_image(RESULTS_DIR / f"{name}_stationarity.png")
    if img2:
        story.append(img2)
        story.append(Paragraph(
            f"Figure 2 — {display}: rolling mean and standard deviation ({unit}).", S["caption"]))

    story.append(PageBreak())

    # ── 2. Stationarity Tests ─────────────────────────────────────────────────
    story.append(Paragraph("2. Stationarity Analysis", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))
    story.append(Paragraph(
        "Two complementary tests are applied. ADF (H0: unit root) requires p &lt; 0.05 for "
        "stationarity; KPSS (H0: stationary) requires p &gt; 0.05 for stationarity. "
        "If the series is non-stationary, ARIMA handles differencing automatically via the "
        "<i>d</i> parameter.", S["body"]))

    tbl_data = [
        ["Test", "Statistic", "p-value", "Result"],
        ["ADF",  f"{adf['statistic']:.4f}", f"{adf['p_value']:.4f}",
         "Stationary" if adf["stationary"] else "Non-stationary"],
        ["KPSS", f"{kps['statistic']:.4f}", f"{kps['p_value']:.4f}",
         "Stationary" if kps["stationary"] else "Non-stationary"],
    ]
    tbl = Table(tbl_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    img = _embed_image(RESULTS_DIR / f"{name}_acf_pacf.png")
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 3 — Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions. "
            "These guide ARIMA order selection.", S["caption"]))

    story.append(PageBreak())

    # ── 3. Model ──────────────────────────────────────────────────────────────
    story.append(Paragraph("3. ARIMA Model", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))

    order = forecaster.order_
    s_order = forecaster.seasonal_order_
    story.append(Paragraph(
        f"The best model selected by AIC minimisation via <i>auto_arima</i> is "
        f"<b>ARIMA({order[0]},{order[1]},{order[2]})</b> with seasonal order "
        f"({s_order[0]},{s_order[1]},{s_order[2]},{s_order[3]}).",
        S["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    param_data = [
        ["Parameter", "Value", "Description"],
        ["p", str(order[0]),   "Autoregressive order"],
        ["d", str(order[1]),   "Degree of differencing"],
        ["q", str(order[2]),   "Moving average order"],
        ["P", str(s_order[0]), "Seasonal AR order"],
        ["D", str(s_order[1]), "Seasonal differencing"],
        ["Q", str(s_order[2]), "Seasonal MA order"],
        ["s", str(s_order[3]), "Seasonal period (trading days)"],
    ]
    tbl = Table(param_data, colWidths=[3*cm, 3*cm, 10*cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    img = _embed_image(RESULTS_DIR / f"{name}_diagnostics.png")
    if img:
        story.append(img)
        story.append(Paragraph(
            "Figure 4 — Residual diagnostics. Well-specified models show white-noise residuals "
            "(no autocorrelation, approximately normal distribution).", S["caption"]))

    story.append(PageBreak())

    # ── 4. Validation ─────────────────────────────────────────────────────────
    story.append(Paragraph("4. Walk-Forward Validation", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))
    story.append(Paragraph(
        "The model is validated using an expanding-window walk-forward approach: "
        "at each step the model is re-trained on all available history and asked to "
        "predict one trading day ahead. This mimics real-world usage.", S["body"]))

    tbl_data = [
        ["Metric", "Value", "Interpretation"],
        ["MAE  (Mean Absolute Error)",
         f"${val['MAE']:.4f}",
         f"Average absolute error in {unit}"],
        ["RMSE (Root Mean Squared Error)",
         f"${val['RMSE']:.4f}",
         f"Penalises large errors more; in {unit}"],
        ["MAPE (Mean Absolute % Error)",
         f"{val['MAPE']:.2f}%",
         "Scale-free percentage error"],
    ]
    tbl = Table(tbl_data, colWidths=[6*cm, 3.5*cm, 6.5*cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    img = _embed_image(RESULTS_DIR / f"{name}_validation.png")
    if img:
        story.append(img)
        story.append(Paragraph(
            f"Figure 5 — Walk-forward validation: actual vs predicted prices ({unit}).",
            S["caption"]))

    story.append(PageBreak())

    # ── 5. Forecast ───────────────────────────────────────────────────────────
    story.append(Paragraph("5. Price Forecast", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6))
    story.append(Paragraph(
        f"Forecast horizon: <b>{len(forecast)} trading days</b>. "
        f"All values in <b>{unit}</b>. "
        f"Confidence interval: <b>95%</b>.",
        S["body"]
    ))

    img = _embed_image(RESULTS_DIR / f"{name}_forecast.png")
    if img:
        story.append(img)
        story.append(Paragraph(
            f"Figure 6 — {display}: historical prices and {len(forecast)}-day forecast "
            f"with 95% confidence interval ({unit}).", S["caption"]))
    story.append(Spacer(1, 0.3*cm))

    # Forecast table — show every 5th row to keep it manageable
    step  = max(1, len(forecast) // 20)
    rows  = [["Date", f"Forecast ({unit})", f"Lower 95% CI", f"Upper 95% CI"]]
    for dt, row in forecast.iloc[::step].iterrows():
        rows.append([
            dt.strftime("%Y-%m-%d"),
            f"${row['forecast']:.2f}",
            f"${row['lower_ci']:.2f}",
            f"${row['upper_ci']:.2f}",
        ])
    tbl = Table(rows, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
    tbl.setStyle(_table_style())
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Disclaimer: Forecasts are statistical extrapolations based on historical price patterns. "
        "They do not constitute investment advice. Energy prices are highly sensitive to "
        "geopolitical events, weather, and supply/demand shocks not captured by ARIMA models.",
        S["small"]
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"  OK PDF report saved: {out_path}")
    return out_path
