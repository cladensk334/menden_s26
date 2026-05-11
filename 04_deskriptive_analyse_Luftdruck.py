# -*- coding: utf-8 -*-
# =============================================================================
# 04_deskriptive_analyse.py
# Luftdruck Würzburg – ARIMA-Analyse nach Box-Jenkins-Methode
# =============================================================================
# Eingabe : data/luftdruck_bereinigt.csv  (Ausgabe von 02_datenbereinigung.py)
# Ausgabe : Konsolenausgabe (Testergebnisse, Modellselektion, Prognose)
#           data/prognose.csv             (10-Tage-Prognose als CSV)
# =============================================================================
# Reihenfolge der Analyse:
#   1. Integrationsordnung bestimmen (ADF- und KPSS-Test)
#   2. Transformation zur Stationarität (Differenzierung)
#   3. ACF / PACF interpretieren
#   4. Modellselektion über AIC / BIC (ARIMA-Gitter)
#   5. Residuendiagnose (Ljung-Box, Jarque-Bera, ADF)
#   6. t-Statistiken der Koeffizienten
#   7. 10-Tage-Prognose mit 95%-Konfidenzintervall
# =============================================================================
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

EINGABE  = os.path.join("data", "luftdruck_bereinigt.csv")
AUSGABE  = os.path.join("data", "prognose.csv")

# --- Daten laden -------------------------------------------------------------
ts = pd.read_csv(EINGABE, index_col=0, parse_dates=True).squeeze()
ts.index.freq = "D"

print("=" * 65)
print("DESKRIPTIVE ANALYSE: LUFTDRUCK WÜRZBURG – ARIMA (Box-Jenkins)")
print("=" * 65)
print(f"\n  Zeitreihe: {ts.index[0].date()} bis {ts.index[-1].date()}")
print(f"  Beobachtungen: {len(ts):,} Tageswerte")

# =============================================================================
# SCHRITT 1: INTEGRATIONSORDNUNG
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 1: INTEGRATIONSORDNUNG – STATIONARITAETSTESTS")
print("=" * 65)


def adf_test(serie, bezeichnung=""):
    """Augmented Dickey-Fuller Test (H0: Einheitswurzel vorhanden)."""
    result = adfuller(serie.dropna(), autolag="AIC")
    print(f"\n  ADF-Test: {bezeichnung}")
    print(f"    Teststatistik  : {result[0]:.4f}")
    print(f"    p-Wert         : {result[1]:.4f}")
    print(f"    Lags verwendet : {result[2]}")
    print(
        f"    Krit. Werte    : 1%={result[4]['1%']:.3f}, "
        f"5%={result[4]['5%']:.3f}, 10%={result[4]['10%']:.3f}"
    )
    entscheid = "STATIONAER" if result[1] < 0.05 else "NICHT STATIONAER"
    print(f"    => Entscheidung (5%-Niveau): {entscheid}")
    return result[1]


def kpss_test(serie, bezeichnung=""):
    """KPSS-Test (H0: Reihe ist stationär)."""
    result = kpss(serie.dropna(), regression="c", nlags="auto")
    print(f"\n  KPSS-Test: {bezeichnung}")
    print(f"    Teststatistik  : {result[0]:.4f}")
    print(f"    p-Wert         : {result[1]:.4f}")
    print(f"    Krit. Werte    : {result[3]}")
    entscheid = "NICHT STATIONAER" if result[1] < 0.05 else "STATIONAER"
    print(f"    => Entscheidung (5%-Niveau): {entscheid}")
    return result[1]


print("\n--- Niveau der Zeitreihe ---")
p_adf_n = adf_test(ts, "Taeglicher Luftdruck (Niveau)")
p_kpss_n = kpss_test(ts, "Taeglicher Luftdruck (Niveau)")

ts_diff1 = ts.diff().dropna()
print("\n--- Erste Differenz ---")
p_adf_d1 = adf_test(ts_diff1, "Erste Differenz (d=1)")
p_kpss_d1 = kpss_test(ts_diff1, "Erste Differenz (d=1)")

if p_adf_n < 0.05 and p_kpss_n >= 0.05:
    d = 0
    print("\n  => Zeitreihe ist im NIVEAU stationaer => d = 0")
elif p_adf_d1 < 0.05 or p_kpss_d1 >= 0.05:
    d = 1
    print("\n  => Zeitreihe wird nach ERSTER DIFFERENZIERUNG stationaer => d = 1")
else:
    d = 1
    print("\n  => d = 1 (konservative Wahl bei gemischten Ergebnissen)")

print(f"\n  Gewaehlte Integrationsordnung: d = {d}")

# =============================================================================
# SCHRITT 2: TRANSFORMATION
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 2: TRANSFORMATION ZUR STATIONARITAET")
print("=" * 65)

ts_stat = ts.copy() if d == 0 else ts.diff(d).dropna()
print(f"\n  {'Keine Differenzierung noetig.' if d == 0 else str(d)+'-fache Differenzierung angewendet.'}")
print(f"  Bereinigte Reihe: {len(ts_stat):,} Beobachtungen")
print(f"  Mittelwert      : {ts_stat.mean():.4f} hPa")
print(f"  Standardabweich.: {ts_stat.std():.4f} hPa")

# =============================================================================
# SCHRITT 3: ACF / PACF INTERPRETATION
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 3: ACF UND PACF DER TRANSFORMIERTEN REIHE")
print("=" * 65)

max_lag = 30
acf_werte  = acf(ts_stat, nlags=max_lag, alpha=0.05)
pacf_werte = pacf(ts_stat, nlags=max_lag, alpha=0.05, method="ywm")
acf_konf   = acf_werte[1]
pacf_konf  = pacf_werte[1]

sig_acf = [
    i for i in range(1, max_lag + 1)
    if abs(acf_werte[0][i]) > (acf_konf[i, 1] - acf_werte[0][i])
]
sig_pacf = [
    i for i in range(1, max_lag + 1)
    if abs(pacf_werte[0][i]) > (pacf_konf[i, 1] - pacf_werte[0][i])
]

print(f"\n  Signifikante ACF-Lags  (5%): {sig_acf[:10]}")
print(f"  Signifikante PACF-Lags (5%): {sig_pacf[:10]}")
print(
    "\n  Interpretation:\n"
    "  - ACF klingt langsam ab, PACF bricht nach Lag p ab  => AR(p)\n"
    "  - PACF klingt langsam ab, ACF bricht nach Lag q ab  => MA(q)\n"
    "  - Beide klingen ab                                   => ARMA(p,q)\n"
)

# ACF-Tabelle (erste 10 Lags)
print(f"  {'Lag':>4} {'ACF':>8} {'PACF':>8}")
print("  " + "-" * 22)
for i in range(1, 11):
    print(f"  {i:>4} {acf_werte[0][i]:>8.4f} {pacf_werte[0][i]:>8.4f}")

# =============================================================================
# SCHRITT 4: MODELLSELEKTION (AIC / BIC GITTER)
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 4: MODELLSELEKTION – ARIMA-GITTER")
print("=" * 65)

p_werte = range(0, 4)
q_werte = range(0, 4)
ergebnisse = []
print(f"\n  Schaetze ARIMA(p,{d},q) fuer p, q in {{0, 1, 2, 3}} ...")

for p, q in itertools.product(p_werte, q_werte):
    if p == 0 and q == 0:
        continue
    try:
        fit_tmp = ARIMA(ts, order=(p, d, q)).fit()
        ergebnisse.append({
            "p": p, "q": q,
            "AIC"   : round(fit_tmp.aic, 2),
            "BIC"   : round(fit_tmp.bic, 2),
            "LogLik": round(fit_tmp.llf, 2),
        })
    except Exception:
        pass

ergebnis_df = pd.DataFrame(ergebnisse).sort_values("AIC")
print("\n  Modellvergleich (sortiert nach AIC):")
print(ergebnis_df.to_string(index=False))

aic_best = ergebnis_df.iloc[0]
bic_best = ergebnis_df.sort_values("BIC").iloc[0]
p_final  = int(aic_best["p"])
q_final  = int(aic_best["q"])
d_final  = d

print(f"\n  Bestes Modell nach AIC: ARIMA({int(aic_best['p'])},{d},{int(aic_best['q'])})  AIC={aic_best['AIC']}")
print(f"  Bestes Modell nach BIC: ARIMA({int(bic_best['p'])},{d},{int(bic_best['q'])})  BIC={bic_best['BIC']}")
print(f"\n  => Gewaaehltes Modell (nach AIC): ARIMA({p_final},{d_final},{q_final})")

# =============================================================================
# SCHRITT 5 & 6: MODELLSCHAETZUNG, t-STATISTIKEN
# =============================================================================
print("\n" + "=" * 65)
print(f"SCHRITT 5 & 6: MODELLSCHAETZUNG – ARIMA({p_final},{d_final},{q_final})")
print("=" * 65)

modell_final = ARIMA(ts, order=(p_final, d_final, q_final))
fit_final    = modell_final.fit()
print(fit_final.summary())

# t-Statistiken
params  = fit_final.params
stderr  = fit_final.bse
tstat   = fit_final.tvalues
pvalues = fit_final.pvalues

print("\n  --- t-Statistiken der Koeffizienten ---\n")
print(
    f"  {'Koeffizient':<18} {'Schaetzer':>10} {'Std.Fehler':>12} "
    f"{'t-Statistik':>12} {'p-Wert':>10} {'Signif.':>8}"
)
print("  " + "-" * 75)
for name in params.index:
    sig = (
        "***" if pvalues[name] < 0.01
        else "**" if pvalues[name] < 0.05
        else "*"  if pvalues[name] < 0.10
        else ""
    )
    print(
        f"  {name:<18} {params[name]:>10.4f} {stderr[name]:>12.4f} "
        f"{tstat[name]:>12.4f} {pvalues[name]:>10.4f} {sig:>8}"
    )

print("\n  Signifikanzniveaus: *** p<0.01  ** p<0.05  * p<0.10")
print(
    "\n  Interpretation:\n"
    "  - t-Statistik = Schaetzer / Standardfehler\n"
    "  - |t| > 1.96 => auf 5%-Niveau signifikant (asymptotisch N(0,1))\n"
    "  - p < 0.01   => hochsignifikant (***)\n"
)

# =============================================================================
# SCHRITT 5 (FORTS.): RESIDUENDIAGNOSE
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 5: RESIDUENDIAGNOSE")
print("=" * 65)

residuen = fit_final.resid

# Ljung-Box
lb = acorr_ljungbox(residuen, lags=[10, 20, 30], return_df=True)
print("\n  Ljung-Box-Test (H0: keine Autokorrelation in den Residuen):")
for lag in [10, 20, 30]:
    row = lb.loc[lag]
    ent = "nicht abgelehnt (ok)" if row["lb_pvalue"] > 0.05 else "ABGELEHNT (!)"
    print(
        f"    Lag {lag:2d}: Q = {row['lb_stat']:8.3f},  "
        f"p = {row['lb_pvalue']:.4f}  => H0 {ent}"
    )

# Jarque-Bera
jb_stat, jb_p = stats.jarque_bera(residuen)
jb_ent = "nicht abgelehnt (ok)" if jb_p > 0.05 else "ABGELEHNT (!)"
print(f"\n  Jarque-Bera-Test (H0: Normalverteilung der Residuen):")
print(f"    Statistik = {jb_stat:.3f},  p = {jb_p:.4f}  => H0 {jb_ent}")
if jb_p < 0.05:
    print(
        "    Hinweis: Bei langen Meteorologiereihen haeufig aufgrund\n"
        "    von Extremwetterlagen. Konsistenz der Schaetzer bleibt erhalten."
    )

# ADF auf Residuen
adf_res = adfuller(residuen, autolag="AIC")
adf_ent = "Stationaer (ok)" if adf_res[1] < 0.05 else "Nicht stationaer (!)"
print(f"\n  ADF-Test auf Residuen (H0: Einheitswurzel):")
print(f"    Statistik = {adf_res[0]:.4f},  p = {adf_res[1]:.4f}  => {adf_ent}")

# Residuenstatistik
print(f"\n  Residuen – Deskriptive Statistik:")
print(f"    Mittelwert  : {residuen.mean():.6f}")
print(f"    Std.abw.    : {residuen.std():.4f}")
print(f"    Schiefe     : {residuen.skew():.4f}")
print(f"    Kurtosis    : {residuen.kurt():.4f}")

# =============================================================================
# SCHRITT 7: 10-TAGE-PROGNOSE
# =============================================================================
print("\n" + "=" * 65)
print("SCHRITT 7: 10-TAGE-PROGNOSE MIT 95%-KONFIDENZINTERVALL")
print("=" * 65)

n_prognose  = 10
prognose    = fit_final.get_forecast(steps=n_prognose)
prog_mittel = prognose.predicted_mean
prog_ki     = prognose.conf_int(alpha=0.05)

print(f"\n  Prognose ab {prog_mittel.index[0].strftime('%d.%m.%Y')}:\n")
print(
    f"  {'Datum':<14} {'Prognose [hPa]':>16} "
    f"{'95%-KI untere':>15} {'95%-KI obere':>14}"
)
print("  " + "-" * 62)
for datum, prog, ki_u, ki_o in zip(
    prog_mittel.index,
    prog_mittel.values,
    prog_ki.iloc[:, 0].values,
    prog_ki.iloc[:, 1].values,
):
    print(
        f"  {datum.strftime('%d.%m.%Y'):<14} {prog:>16.2f} "
        f"{ki_u:>15.2f} {ki_o:>14.2f}"
    )

# Prognose als CSV speichern
prog_df = pd.DataFrame({
    "Datum"       : prog_mittel.index.strftime("%d.%m.%Y"),
    "Prognose"    : prog_mittel.values.round(2),
    "KI_95_unten" : prog_ki.iloc[:, 0].values.round(2),
    "KI_95_oben"  : prog_ki.iloc[:, 1].values.round(2),
})
prog_df.to_csv(AUSGABE, index=False)
print(f"\n  Prognose gespeichert: {AUSGABE}")

# =============================================================================
# ZUSAMMENFASSUNG
# =============================================================================
print("\n" + "=" * 65)
print("ZUSAMMENFASSUNG")
print("=" * 65)
print(f"""
  Datensatz           : Luftdruck Wuerzburg, Station 5705
  Zeitraum            : {ts.index[0].strftime('%d.%m.%Y')} – {ts.index[-1].strftime('%d.%m.%Y')}
  Beobachtungen       : {len(ts):,} Tageswerte

  Integrationsordnung : d = {d_final}
  Gewaaehltes Modell  : ARIMA({p_final},{d_final},{q_final})
  AIC                 : {fit_final.aic:.2f}
  BIC                 : {fit_final.bic:.2f}
  Log-Likelihood      : {fit_final.llf:.2f}

  Prognose Wert 1     : {prog_mittel.values[0]:.2f} hPa  (95%-KI: [{prog_ki.iloc[0,0]:.2f}, {prog_ki.iloc[0,1]:.2f}])
  Prognose Wert 10    : {prog_mittel.values[-1]:.2f} hPa  (95%-KI: [{prog_ki.iloc[-1,0]:.2f}, {prog_ki.iloc[-1,1]:.2f}])
""")
print("=" * 65)
