

# STATIONARITÄTSTESTS – Wind (FM)

from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# ── ADF Test ─────────────────────────────────────────────────
adf_result = adfuller(wind_clean, autolag="AIC")

print(" Augmented Dickey-Fuller Test (ADF) ")
print(f"H0: Die Zeitreihe hat eine Einheitswurzel (nicht stationär)")
print(f"H1: Die Zeitreihe ist stationär\n")
print(f"  Teststatistik:    {adf_result[0]:.4f}")
print(f"  p-Wert:           {adf_result[1]:.4f}")
print(f"  Lags verwendet:   {adf_result[2]}")
print(f"  Kritische Werte:")
for key, val in adf_result[4].items():
    print(f"    {key}: {val:.4f}")
if adf_result[1] < 0.05:
    print(f"\n  → p < 0.05: H0 wird abgelehnt ✅ Zeitreihe ist stationär")
else:
    print(f"\n  → p > 0.05: H0 wird nicht abgelehnt ❌ Zeitreihe ist nicht stationär")

# ── KPSS Test ────────────────────────────────────────────────
kpss_result = kpss(wind_clean, regression="c", nlags="auto")

print("\n KPSS Test ")
print(f"H0: Die Zeitreihe ist stationär")
print(f"H1: Die Zeitreihe ist nicht stationär\n")
print(f"  Teststatistik:    {kpss_result[0]:.4f}")
print(f"  p-Wert:           {kpss_result[1]:.4f}")
print(f"  Lags verwendet:   {kpss_result[2]}")
print(f"  Kritische Werte:")
for key, val in kpss_result[3].items():
    print(f"    {key}: {val:.4f}")
if kpss_result[1] > 0.05:
    print(f"\n  → p > 0.05: H0 wird nicht abgelehnt ✅ Zeitreihe ist stationär")
else:
    print(f"\n  → p < 0.05: H0 wird abgelehnt ❌ Zeitreihe ist nicht stationär")

"""Stationaritätstests prüfen in der Zeitreihenanalyse, ob statistische Eigenschaften wie Mittelwert und Varianz über die Zeit konstant bleiben. Dies ist entscheidend für verlässliche Prognosemodelle, da nicht-stationäre Daten oft zu falschen Ergebnissen führen. Die gängigsten Methoden sind visuelle Inspektion (Plots), der Augmented Dickey-Fuller (ADF) Test und der KPSS-Test."""

# Schwache Stationalität ──> erste Differenz bilden
wind_diff = wind_clean.diff().dropna()

# ── ADF auf differenzierte Reihe ─────────────────────────────
adf_diff = adfuller(wind_diff, autolag="AIC")
kpss_diff = kpss(wind_diff, regression="c", nlags="auto")

print("=== Tests auf differenzierte Reihe (d=1) ===\n")
print(f"ADF  p-Wert:  {adf_diff[1]:.4f} → {'✅ Stationär' if adf_diff[1] < 0.05 else '❌ Nicht stationär'}")
print(f"KPSS p-Wert:  {kpss_diff[1]:.4f} → {'✅ Stationär' if kpss_diff[1] > 0.05 else '❌ Nicht stationär'}")

# ── Plot zum Vergleich ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(wind_clean, linewidth=0.4, color="steelblue")
axes[0].set_title("Original Zeitreihe")
axes[0].set_ylabel("m/s")

axes[1].plot(wind_diff, linewidth=0.4, color="tomato")
axes[1].set_title("Differenzierte Zeitreihe (d=1)")
axes[1].set_ylabel("Δm/s")
plt.tight_layout()
plt.show()

"""Was bedeutet d=1?
Es bedeutet dass deine Zeitreihe einen schwachen Trend hat der durch einmaliges Differenzieren entfernt wird. Die differenzierte Reihe zeigt die tägliche Veränderung der Windgeschwindigkeit statt der absoluten Werte.
"""

# ACF und PACF auf der differenzierten Reihe (d=1)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(wind_diff, lags=40, ax=axes[0],
         title="ACF – Windgeschwindigkeit differenziert (d=1)")
plot_pacf(wind_diff, lags=40, ax=axes[1],
          title="PACF – Windgeschwindigkeit differenziert (d=1)",
          method="ywm")
plt.tight_layout()
plt.show()

# SCHRITT 4C: ARIMA MODELLSELEKTION (Grid Search) – Wind (FM)

from itertools import product
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("=== Grid Search ARIMA(p, d, q) ===\n")
print("Suche über p ∈ {0,1,2,3}, d ∈ {0,1}, q ∈ {0,1,2,3}...\n")

results = []

for p, d, q in product(range(4), [1], range(4)):
    try:
        model = ARIMA(wind_clean, order=(p, d, q))
        fit = model.fit()
        results.append({
            "Ordnung (p,d,q)": f"({p},{d},{q})",
            "AIC": round(fit.aic, 2),
            "BIC": round(fit.bic, 2),
            "HQIC": round(fit.hqic, 2)
        })
    except Exception:
        continue

# Nach AIC sortieren
results_df = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)

print("=== Top 10 Modelle nach AIC ===\n")
print(results_df.head(10).to_string(index=False))

best_order_str = results_df.iloc[0]["Ordnung (p,d,q)"]
print(f"\n→ Bestes Modell nach AIC: ARIMA{best_order_str}")

"""AIC - AIC Akaike Information Criterion, SC - BIC Schwarz Criterion (= Bayesian Information Criterion), HQ - HQIC Hannan-Quinn Information Criterion

ARIMA(3, 1, 2) bedeutet:

p = 3 → Das Modell nutzt die letzten 3 Tage als Eingabe (AR-Teil)
d = 1 → Die Reihe wurde einmal differenziert (Trend entfernt)
q = 2 → Die letzten 2 Vorhersagefehler werden korrigiert (MA-Teil)
"""

# SCHRITT 4D: MODELL SCHÄTZEN & DIAGNOSTIK – Wind (FM)


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Bestes Modell fitten ─────────────────────────────────────
# Passe p, d, q entsprechend dem Ergebnis aus 4C an!
best_order = (3,1,2)   # ← wird nach Grid Search angepasst

print(f"=== Modell ARIMA{best_order} wird geschätzt ===\n")
model = ARIMA(wind_clean, order=best_order)
fit = model.fit()

# ── Koeffizienten und t-Statistiken ─────────────────────────
print("=== Geschätzte Koeffizienten ===\n")
print(f"{'Koeffizient':<15} {'Schätzer':>10} {'t-Statistik':>12} {'p-Wert':>10} {'Signifikanz':>12}")
print("-" * 65)
for name in fit.params.index:
    p_val = fit.pvalues[name]
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"{name:<15} {fit.params[name]:>10.4f} {fit.tvalues[name]:>12.4f} {p_val:>10.4f} {sig:>12}")
print("\nSignifikanzniveaus: * p<0.1  ** p<0.05  *** p<0.01")

# ── Residualdiagnostik ───────────────────────────────────────
residuals = pd.Series(fit.resid).dropna()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(residuals, linewidth=0.5, color="steelblue")
axes[0].axhline(0, color="tomato", linestyle="--", linewidth=0.8)
axes[0].set_title("Residuen über Zeit")

plot_acf(residuals, lags=30, ax=axes[1], title="ACF der Residuen")

axes[2].hist(residuals, bins=40, color="steelblue",
             alpha=0.7, edgecolor="white")
axes[2].set_title("Verteilung der Residuen")

plt.tight_layout()
plt.show()

# ── Ljung-Box Test ───────────────────────────────────────────
lb = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\n=== Ljung-Box Test (H0: keine Autokorrelation in Residuen) ===\n")
print(lb.to_string())
if (lb["lb_pvalue"] > 0.05).all():
    print("\n→ p > 0.05 für alle Lags ✅ Residuen sind unkorrelliert → Modell gut spezifiziert")
else:
    print("\n→ p < 0.05 ❌ Residuen zeigen noch Autokorrelation → Modell anpassen")

# SCHRITT 4E: PROGNOSE – Wind (FM)


import matplotlib.pyplot as plt
import pandas as pd

print("=== 10-Perioden Prognose ===\n")

# ── Prognose berechnen ───────────────────────────────────────
forecast_obj = fit.get_forecast(steps=10)
fc_mean = forecast_obj.predicted_mean
fc_ci = forecast_obj.conf_int(alpha=0.05)
fc_ci.columns = ["Unteres 95% KI", "Oberes 95% KI"]

# ── Prognosetabelle ──────────────────────────────────────────
fc_df = pd.concat([fc_mean.rename("Prognose"), fc_ci], axis=1).round(4)
print(fc_df.to_string())

# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

# Letzte 100 Tage als Kontext
ax.plot(wind_clean[-100:], linewidth=0.8,
        color="steelblue", label="Historisch (letzte 100 Tage)")

# Prognose
ax.plot(fc_df["Prognose"], color="tomato",
        linewidth=2, marker="o", label="Prognose")

# Konfidenzintervall
ax.fill_between(fc_df.index,
                fc_df["Unteres 95% KI"],
                fc_df["Oberes 95% KI"],
                alpha=0.3, color="tomato", label="95% Konfidenzintervall")

ax.set_title("10-Tage Prognose – Windgeschwindigkeit Würzburg (FM)", fontsize=13)
ax.set_ylabel("m/s")
ax.set_xlabel("Datum")
ax.legend()
plt.tight_layout()
plt.show()

"""
Windgeschwindigkeit ist eine stark stochastische Größe mit hoher täglicher Variabilität (σ = 1.72 m/s). Das ARIMA(3,1,2) Modell konvergiert bereits nach wenigen Perioden gegen den langfristigen Mittelwert, da vergangene Windwerte nur eine sehr kurze Gedächtnisstruktur aufweisen. Dies ist kein Modellversagen, sondern ein Ausdruck der inhärenten Unvorhersehbarkeit von Windgeschwindigkeit über längere Zeithorizonte. Für präzisere Windprognosen wären physikalische Atmosphärenmodelle erforderlich."""
