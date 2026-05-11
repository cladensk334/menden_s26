# -*- coding: utf-8 -*-
# =============================================================================
# 02_datenbereinigung.py
# Luftdruck Würzburg – Bereinigung, Lückenfüllung, Ausreißerprüfung
# =============================================================================
# Eingabe : data/luftdruck_roh.csv       (Ausgabe von 01_datenimport.py)
# Ausgabe : data/luftdruck_bereinigt.csv (Tageswerte, bereinigt)
# =============================================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd

EINGABE  = os.path.join("data", "luftdruck_roh.csv")
AUSGABE  = os.path.join("data", "luftdruck_bereinigt.csv")

print("=" * 60)
print("SCHRITT 2: DATENBEREINIGUNG")
print("=" * 60)

# --- Rohdaten laden ----------------------------------------------------------
ts = pd.read_csv(EINGABE, index_col=0, parse_dates=True).squeeze()
ts.index.freq = "D"
print(f"\n  Geladene Reihe       : {len(ts):,} Tageswerte")
print(f"  Fehlende Werte (NaN) : {ts.isna().sum()}")

# --- 1. Lückenfüllung durch lineare Zeitinterpolation -----------------------
n_vor = ts.isna().sum()
ts = ts.interpolate(method="time")
n_nach = ts.isna().sum()
print(f"\n  [1] Lineare Interpolation fehlender Tage")
print(f"      Vor Interpolation : {n_vor} NaN")
print(f"      Nach Interpolation: {n_nach} NaN")

# --- 2. Ausreißer prüfen (IQR-Methode, Warnung ohne Entfernung) --------------
q1, q3 = ts.quantile(0.25), ts.quantile(0.75)
iqr = q3 - q1
grenze_u = q1 - 3.0 * iqr
grenze_o = q3 + 3.0 * iqr
ausreisser = ts[(ts < grenze_u) | (ts > grenze_o)]

print(f"\n  [2] Ausreißerprüfung (3×IQR-Methode)")
print(f"      IQR-Grenzen      : [{grenze_u:.2f}, {grenze_o:.2f}] hPa")
print(f"      Ausreißer gefunden: {len(ausreisser)}")
if len(ausreisser) > 0:
    print(f"      Extremwerte:")
    for d, v in ausreisser.sort_values().iloc[[0, -1]].items():
        print(f"        {d.date()}: {v:.2f} hPa")
    # Ausreißer per gleitendem Median ersetzen (7-Tage-Fenster)
    ts_median = ts.rolling(window=7, center=True, min_periods=1).median()
    maske = (ts < grenze_u) | (ts > grenze_o)
    ts[maske] = ts_median[maske]
    print(f"      => Ersetzt durch 7-Tage-gleitenden Median")

# --- 3. Statistik der bereinigten Reihe -------------------------------------
print(f"\n  [3] Bereinigte Zeitreihe – Deskriptive Statistik")
print(f"      Zeitraum         : {ts.index[0].date()} bis {ts.index[-1].date()}")
print(f"      Beobachtungen    : {len(ts):,}")
print(f"      Mittelwert       : {ts.mean():.2f} hPa")
print(f"      Standardabweich. : {ts.std():.2f} hPa")
print(f"      Minimum          : {ts.min():.2f} hPa  ({ts.idxmin().date()})")
print(f"      Maximum          : {ts.max():.2f} hPa  ({ts.idxmax().date()})")
print(f"      Median           : {ts.median():.2f} hPa")
print(f"      Schiefe          : {ts.skew():.4f}")
print(f"      Kurtosis         : {ts.kurt():.4f}")

# --- 4. Monatliche Mittelwerte als Zusatzspalte speichern -------------------
ts_monthly = ts.resample("MS").mean()
print(f"\n  [4] Monatliche Mittelwerte: {len(ts_monthly)} Werte berechnet")

# --- Bereinigten Datensatz speichern ----------------------------------------
ts.name = "PP_TER"
df_out = ts.to_frame()
df_out["PP_TER_monatlich"] = ts.resample("D").asfreq().map(
    ts_monthly.reindex(ts.index, method="ffill")
)
ts.to_csv(AUSGABE, header=True)
print(f"\n  Bereinigte Daten gespeichert: {AUSGABE}")
print("=" * 60)
