# -*- coding: utf-8 -*-
# =============================================================================
# 03_datenvisualisierung.py
# Luftdruck Würzburg – Alle Abbildungen der ARIMA-Analyse
# =============================================================================
# Eingabe : data/luftdruck_bereinigt.csv  (Ausgabe von 02_datenbereinigung.py)
#           data/luftdruck_train.csv
#           data/luftdruck_test.csv
# Ausgabe : plots/01_rohdaten.png
#           plots/02_transformation.png
#           plots/03_acf_pacf.png
#           plots/04_residuenanalyse.png
#           plots/05_prognose.png
#           plots/06_train_test_split.png
#           plots/07_test_evaluation.png
#           plots/08_cv_metriken.png
# =============================================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

EINGABE       = os.path.join("data", "processed", "luftdruck_bereinigt.csv")
EINGABE_TRAIN = os.path.join("data", "processed", "luftdruck_train.csv")
EINGABE_TEST  = os.path.join("data", "processed", "luftdruck_test.csv")
PLOTORDNER    = "Plots"
os.makedirs(PLOTORDNER, exist_ok=True)

print("=" * 60)
print("SCHRITT 3: DATENVISUALISIERUNG")
print("=" * 60)

# --- Daten laden -------------------------------------------------------------
ts = pd.read_csv(EINGABE, index_col=0, parse_dates=True).squeeze()
ts.index.freq = "D"
ts_monthly = ts.resample("MS").mean()
print(f"\n  Daten geladen: {len(ts):,} Tageswerte")

# ---- Hilfsfunktion: Dateiname im Plotordner ---------------------------------
def pfad(dateiname):
    return os.path.join(PLOTORDNER, dateiname)

# =============================================================================
# ABB. 1 – ROHDATEN (Tageswerte + monatliche Übersicht)
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

axes[0].plot(ts.index, ts.values, lw=0.4, color="steelblue", alpha=0.8)
axes[0].set_title(
    "Tagesdurchschnittlicher Luftdruck – Würzburg (Station 5705)", fontsize=13
)
axes[0].set_ylabel("Luftdruck [hPa]")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].xaxis.set_major_locator(mdates.YearLocator(10))

axes[1].plot(ts_monthly.index, ts_monthly.values, lw=1.0, color="darkblue")
axes[1].set_title("Monatliche Mittelwerte (Übersicht)", fontsize=13)
axes[1].set_ylabel("Luftdruck [hPa]")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].xaxis.set_major_locator(mdates.YearLocator(10))

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(pfad("01_rohdaten.png"), bbox_inches="tight")
plt.close()
print("\n  [1] Gespeichert: plots/01_rohdaten.png")

# =============================================================================
# ABB. 2 – TRANSFORMATION (Niveau vs. erste Differenz)
# =============================================================================

# Integrationsordnung bestimmen (ADF + KPSS auf Niveau)
p_adf = adfuller(ts, autolag="AIC")[1]
p_kpss = kpss(ts, regression="c", nlags="auto")[1]
d = 0 if (p_adf < 0.05 and p_kpss >= 0.05) else 1
ts_stat = ts.copy() if d == 0 else ts.diff(d).dropna()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(ts.index, ts.values, color="steelblue", lw=0.8)
axes[0].set_title("Originale Zeitreihe (Tageswerte)", fontsize=12)
axes[0].set_ylabel("Luftdruck [hPa]")
axes[0].axhline(
    ts.mean(), color="red", ls="--", lw=1.2,
    label=f"Mittelwert = {ts.mean():.1f} hPa"
)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(ts_stat.index, ts_stat.values, color="darkgreen", lw=0.6)
titel = f"Erste Differenz (d=1) – stationär" if d > 0 else "Niveau (d=0) – stationär"
axes[1].set_title(f"Transformierte Reihe: {titel}", fontsize=12)
axes[1].set_ylabel("Δ Luftdruck [hPa]" if d > 0 else "Luftdruck [hPa]")
axes[1].axhline(0, color="red", ls="--", lw=1.2)
axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(pfad("02_transformation.png"), bbox_inches="tight")
plt.close()
print(f"  [2] Gespeichert: plots/02_transformation.png  (d={d})")

# =============================================================================
# ABB. 3 – ACF UND PACF der transformierten Reihe
# =============================================================================
max_lag = 30
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(ts_stat, lags=max_lag, ax=axes[0], alpha=0.05, color="steelblue")
axes[0].set_title(
    f"Autokorrelationsfunktion (ACF) – max. {max_lag} Lags", fontsize=12
)
axes[0].set_xlabel("Lag (Tage)")
axes[0].grid(True, alpha=0.3)

plot_pacf(ts_stat, lags=max_lag, ax=axes[1], alpha=0.05, color="darkred", method="ywm")
axes[1].set_title(
    f"Partielle Autokorrelationsfunktion (PACF) – max. {max_lag} Lags", fontsize=12
)
axes[1].set_xlabel("Lag (Tage)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(pfad("03_acf_pacf.png"), bbox_inches="tight")
plt.close()
print("  [3] Gespeichert: plots/03_acf_pacf.png")

# =============================================================================
# ABB. 4 – RESIDUENANALYSE des finalen Modells
# (Modell hier neu geschätzt; Parameter aus 04_deskriptive_analyse.py)
# =============================================================================
print("\n  Schaetze ARIMA(3,1,1) fuer Residuenplot ...")
modell = ARIMA(ts, order=(3, d, 1))
fit = modell.fit()
residuen = fit.resid

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(residuen.index, residuen.values, lw=0.5, color="steelblue")
ax1.axhline(0, color="red", ls="--", lw=1)
ax1.set_title("Residuen – ARIMA(3,1,1)", fontsize=12)
ax1.set_ylabel("Residuum [hPa]")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.tick_params(axis="x", rotation=45)

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(
    residuen, bins=60, color="steelblue", edgecolor="white",
    density=True, alpha=0.7
)
x_norm = np.linspace(residuen.min(), residuen.max(), 200)
ax2.plot(
    x_norm,
    stats.norm.pdf(x_norm, residuen.mean(), residuen.std()),
    "r-", lw=2, label="Normalverteilung"
)
ax2.set_title("Histogramm der Residuen", fontsize=11)
ax2.set_xlabel("Residuum [hPa]")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
stats.probplot(residuen, dist="norm", plot=ax3)
ax3.set_title("QQ-Plot der Residuen", fontsize=11)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
plot_acf(residuen, lags=24, ax=ax4, alpha=0.05, color="steelblue")
ax4.set_title("ACF der Residuen", fontsize=11)
ax4.set_xlabel("Lag (Tage)")
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
plot_pacf(residuen, lags=24, ax=ax5, alpha=0.05, color="darkred", method="ywm")
ax5.set_title("PACF der Residuen", fontsize=11)
ax5.set_xlabel("Lag (Tage)")
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(pfad("04_residuenanalyse.png"), bbox_inches="tight")
plt.close()
print("  [4] Gespeichert: plots/04_residuenanalyse.png")

# =============================================================================
# ABB. 5 – 10-TAGE-PROGNOSE
# =============================================================================
n_prognose  = 10
n_historisch = 180   # letzte 6 Monate im Plot

prognose    = fit.get_forecast(steps=n_prognose)
prog_mittel = prognose.predicted_mean
prog_ki     = prognose.conf_int(alpha=0.05)

ts_plot = ts.iloc[-n_historisch:]
fitted  = fit.fittedvalues.iloc[-n_historisch:]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    ts_plot.index, ts_plot.values,
    color="steelblue", lw=1.0,
    label="Historische Werte (letzte 6 Monate)"
)
ax.plot(
    fitted.index, fitted.values,
    color="orange", lw=0.8, ls="--", alpha=0.8,
    label="Angepasste Werte"
)
ax.plot(
    prog_mittel.index, prog_mittel.values,
    color="red", lw=2, marker="o", markersize=5,
    label="Prognose (10 Tage)"
)
ax.fill_between(
    prog_ki.index,
    prog_ki.iloc[:, 0], prog_ki.iloc[:, 1],
    color="red", alpha=0.15,
    label="95%-Konfidenzintervall"
)
ax.axvline(ts.index[-1], color="gray", ls=":", lw=1.5, label="Prognosebeginn")

ax.set_title(
    "ARIMA(3,1,1) – 10-Tage-Prognose\n"
    "Luftdruck Würzburg (Tageswerte)",
    fontsize=13
)
ax.set_ylabel("Luftdruck [hPa]")
ax.set_xlabel("Datum")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(pfad("05_prognose.png"), bbox_inches="tight")
plt.close()
print("  [5] Gespeichert: plots/05_prognose.png")

# =============================================================================
# ABB. 6 – TRAIN/TEST-SPLIT VISUALISIERUNG
# =============================================================================
ts_train = pd.read_csv(EINGABE_TRAIN, index_col=0, parse_dates=True).squeeze()
ts_test  = pd.read_csv(EINGABE_TEST,  index_col=0, parse_dates=True).squeeze()
ts_train.index.freq = "D"
ts_test.index.freq  = "D"

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_train.index, ts_train.values, color="steelblue", lw=0.5,
        label=f"Trainingsdaten ({len(ts_train):,} Tage, 70%)")
ax.plot(ts_test.index,  ts_test.values,  color="darkorange", lw=0.5,
        label=f"Testdaten ({len(ts_test):,} Tage, 30%)")
ax.axvline(ts_test.index[0], color="black", ls="--", lw=1.5,
           label=f"Split: {ts_test.index[0].date()}")
ax.set_title("Train/Test-Split (70/30) – Luftdruck Würzburg", fontsize=13)
ax.set_ylabel("Luftdruck [hPa]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(10))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pfad("06_train_test_split.png"), bbox_inches="tight")
plt.close()
print("  [6] Gespeichert: plots/06_train_test_split.png")

# =============================================================================
# ABB. 7 – TEST-SET-EVALUATION (Prognose vs. Ist-Werte)
# =============================================================================
print("\n  Schaetze ARIMA(3,1,1) auf Trainingsdaten fuer Test-Evaluation ...")
fit_train = ARIMA(ts_train, order=(3, 1, 1)).fit()
prog_test = fit_train.get_forecast(steps=len(ts_test)).predicted_mean
prog_ki_t = fit_train.get_forecast(steps=len(ts_test)).conf_int(alpha=0.05)
prog_test.index = ts_test.index
prog_ki_t.index = ts_test.index

# Metriken berechnen
mse  = mean_squared_error(ts_test, prog_test)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(ts_test, prog_test)
mape = np.mean(np.abs((ts_test.values - prog_test.values) / ts_test.values)) * 100

# Plot: letzten 6 Monate Training + gesamter Test
n_hist = 180
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts_train.iloc[-n_hist:].index, ts_train.iloc[-n_hist:].values,
        color="steelblue", lw=0.8, label="Trainingsdaten (letzte 6 Monate)")
ax.plot(ts_test.index, ts_test.values,
        color="darkorange", lw=0.8, label="Tatsaechliche Testwerte")
ax.plot(prog_test.index, prog_test.values,
        color="red", lw=1.0, ls="--", label="Prognose (Multi-Step)")
ax.fill_between(prog_ki_t.index,
                prog_ki_t.iloc[:, 0], prog_ki_t.iloc[:, 1],
                color="red", alpha=0.10, label="95%-Konfidenzintervall")
ax.axvline(ts_test.index[0], color="black", ls=":", lw=1.2)
ax.set_title(
    f"ARIMA(3,1,1) – Test-Set-Evaluation\n"
    f"MSE={mse:.2f}  RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%",
    fontsize=12
)
ax.set_ylabel("Luftdruck [hPa]")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(pfad("07_test_evaluation.png"), bbox_inches="tight")
plt.close()
print("  [7] Gespeichert: plots/07_test_evaluation.png")

# =============================================================================
# ABB. 8 – CROSS-VALIDATION-METRIKEN (5-Fold TimeSeriesSplit)
# =============================================================================
print("\n  Fuehre 5-Fold Time-Series-Cross-Validation durch ...")
CV_FENSTER  = 730   # letzte 2 Jahre der Trainingsdaten als CV-Pool
CV_TESTSIZE = 30    # Tage pro Fold
N_SPLITS    = 5

ts_cv  = ts_train.iloc[-CV_FENSTER:]
tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=CV_TESTSIZE)
fold_metriken = []

for fold, (idx_tr, idx_te) in enumerate(tscv.split(ts_cv)):
    cv_tr = ts_cv.iloc[idx_tr]
    cv_te = ts_cv.iloc[idx_te]
    fc    = ARIMA(cv_tr, order=(3, 1, 1)).fit().get_forecast(steps=len(cv_te)).predicted_mean
    fc.index = cv_te.index
    fold_metriken.append({
        "Fold": fold + 1,
        "MSE" : mean_squared_error(cv_te, fc),
        "RMSE": np.sqrt(mean_squared_error(cv_te, fc)),
        "MAE" : mean_absolute_error(cv_te, fc),
        "MAPE": np.mean(np.abs((cv_te.values - fc.values) / cv_te.values)) * 100,
    })

cv_df = pd.DataFrame(fold_metriken).set_index("Fold")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metriken_farben = {"MSE": "steelblue", "RMSE": "darkorange",
                   "MAE": "seagreen",  "MAPE": "tomato"}

for ax, (metrik, farbe) in zip(axes.flat, metriken_farben.items()):
    werte = cv_df[metrik]
    ax.bar(werte.index, werte.values, color=farbe, alpha=0.8, edgecolor="white")
    ax.axhline(werte.mean(), color="black", ls="--", lw=1.5,
               label=f"Mittelwert: {werte.mean():.3f}")
    einheit = "%" if metrik == "MAPE" else "hPa²" if metrik == "MSE" else "hPa"
    ax.set_title(f"{metrik} pro Fold [{einheit}]", fontsize=11)
    ax.set_xlabel("Fold")
    ax.set_ylabel(einheit)
    ax.set_xticks(werte.index)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    f"5-Fold Time-Series-Cross-Validation – ARIMA(3,1,1)\n"
    f"CV-Fenster: letzte {CV_FENSTER} Tage Training, {CV_TESTSIZE} Tage/Fold",
    fontsize=12
)
plt.tight_layout()
plt.savefig(pfad("08_cv_metriken.png"), bbox_inches="tight")
plt.close()
print("  [8] Gespeichert: plots/08_cv_metriken.png")

print("\n  Alle Abbildungen erfolgreich gespeichert.")
print("=" * 60)
