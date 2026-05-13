# -*- coding: utf-8 -*-
# =============================================================================
# 01_datenimport.py
# Luftdruck Würzburg (Station 5705) – Rohdaten einlesen und speichern
# =============================================================================
# Eingabe : Luftdruck Würzburg.txt  (Semikolon-getrennt, DWD-Format)
# Ausgabe : data/luftdruck_roh.csv  (Tageswerte, unbereinigt)
# =============================================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import pandas as pd

DATEIPFAD = os.path.join("data", "raw", "Luftdruck Würzburg.txt")
AUSGABE   = os.path.join("data", "processed", "luftdruck_roh.csv")

# --- Ausgabeordner anlegen ---------------------------------------------------
os.makedirs(os.path.join("data", "processed"), exist_ok=True)

print("=" * 60)
print("SCHRITT 1: DATENIMPORT")
print("=" * 60)

# --- Rohdaten einlesen -------------------------------------------------------
print(f"\n  Lese Datei: {DATEIPFAD}")

df = pd.read_csv(
    DATEIPFAD,
    sep=";",
    skipinitialspace=True,
    na_values=["-999", "-999.0", ""],
)
df.columns = df.columns.str.strip()

print(f"  Zeilen gelesen       : {len(df):,}")
print(f"  Spalten              : {list(df.columns)}")

# --- Relevante Spalten auswählen ---------------------------------------------
df = df[["MESS_DATUM", "PP_TER"]].copy()
df["MESS_DATUM"] = df["MESS_DATUM"].astype(str).str.strip()

# --- Datum parsen (Format YYYYMMDDH / YYYYMMDDHH) ----------------------------
df["Datum"] = pd.to_datetime(df["MESS_DATUM"].str[:8], format="%Y%m%d")
df["PP_TER"] = pd.to_numeric(df["PP_TER"], errors="coerce")

# Fehlwerte aus der Rohquelle zählen (noch nicht bereinigen)
n_fehlend_roh = df["PP_TER"].isna().sum()
print(f"  Fehlende Druckwerte  : {n_fehlend_roh}")

# --- Tagesdurchschnitte (3 Messungen/Tag -> 1 Wert) --------------------------
df_valid = df.dropna(subset=["PP_TER"])
ts_daily = df_valid.groupby("Datum")["PP_TER"].mean()

# Alle Tage des Zeitraums sicherstellen (vollständiger Index)
ts_daily = ts_daily.asfreq("D")
n_luecken = ts_daily.isna().sum()

print(f"\n  Zeitraum             : {ts_daily.index[0].date()} bis {ts_daily.index[-1].date()}")
print(f"  Tage gesamt          : {len(ts_daily):,}")
print(f"  Tage ohne Messung    : {n_luecken}")
print(f"  Mittelwert           : {ts_daily.mean():.2f} hPa")
print(f"  Standardabweichung   : {ts_daily.std():.2f} hPa")
print(f"  Minimum              : {ts_daily.min():.2f} hPa")
print(f"  Maximum              : {ts_daily.max():.2f} hPa")

# --- Zwischenspeichern (unbereinigt, Lücken als NaN) ------------------------
ts_daily.to_csv(AUSGABE, header=True)
print(f"\n  Rohdaten gespeichert : {AUSGABE}")

# --- Hinweis: Train/Test-Split (70/30) ---------------------------------------
n_gesamt = len(ts_daily)
n_train  = int(n_gesamt * 0.70)
n_test   = n_gesamt - n_train
split_datum = ts_daily.index[n_train].date()
print(f"\n  Geplanter Train/Test-Split (70/30):")
print(f"    Trainingsdaten : {n_train:,} Tage  (bis {ts_daily.index[n_train - 1].date()})")
print(f"    Testdaten      : {n_test:,} Tage  (ab  {split_datum})")
print(f"    Split wird in 02_datenbereinigung_Luftdruck.py durchgefuehrt.")
print("=" * 60)
