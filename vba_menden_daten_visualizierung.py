
# SCHRITT 3: DATENVISUALISIERUNG – Wind Geschwindigkeit (FM)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

# ── Plot 1: Zeitreihe komplett ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(wind_clean, linewidth=0.4, color="steelblue", alpha=0.8)
ax1.set_title("Windgeschwindigkeit Würzburg 1966–2026 (Tagesmittel)", fontsize=13)
ax1.set_ylabel("m/s")
ax1.set_xlabel("Jahr")

# ── Plot 2: Jahresmittelwerte ────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
jahres_mittel = wind_clean.resample("YE").mean()
ax2.bar(jahres_mittel.index.year, jahres_mittel.values,
        color="steelblue", alpha=0.7, width=0.8)
ax2.axhline(wind_clean.mean(), color="tomato", linestyle="--",
            linewidth=1.2, label=f"Gesamtmittel: {wind_clean.mean():.2f} m/s")
ax2.set_title("Jahresmittelwerte der Windgeschwindigkeit", fontsize=13)
ax2.set_ylabel("m/s")
ax2.set_xlabel("Jahr")
ax2.legend()

# ── Plot 3: Monatliche Saisonalität ─────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
wind_df = wind_clean.to_frame(name="FM")
wind_df["Monat"] = wind_df.index.month
monat_mittel = wind_df.groupby("Monat")["FM"].mean()
monate_labels = ["Jan","Feb","Mär","Apr","Mai","Jun",
                 "Jul","Aug","Sep","Okt","Nov","Dez"]
ax3.bar(monate_labels, monat_mittel.values, color="steelblue", alpha=0.7)
ax3.set_title("Durchschnittliche Windgeschwindigkeit pro Monat", fontsize=12)
ax3.set_ylabel("m/s")
ax3.set_xlabel("Monat")

# ── Plot 4: Histogramm ───────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(wind_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax4.axvline(wind_clean.mean(), color="tomato", linestyle="--",
            linewidth=1.2, label=f"Mittelwert: {wind_clean.mean():.2f} m/s")
ax4.axvline(wind_clean.median(), color="orange", linestyle="--",
            linewidth=1.2, label=f"Median: {wind_clean.median():.2f} m/s")
ax4.set_title("Verteilung der Windgeschwindigkeit", fontsize=12)
ax4.set_xlabel("m/s")
ax4.set_ylabel("Häufigkeit")
ax4.legend()

# ── Plot 5: Boxplot pro Jahrzehnt ────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
wind_df["Jahrzehnt"] = (wind_df.index.year // 10) * 10
jahrzehnte = sorted(wind_df["Jahrzehnt"].unique())
data_per_decade = [wind_df[wind_df["Jahrzehnt"] == j]["FM"].values
                   for j in jahrzehnte]
ax5.boxplot(data_per_decade, labels=[str(j)+"s" for j in jahrzehnte],
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.7))
ax5.set_title("Windgeschwindigkeit pro Jahrzehnt (Boxplot)", fontsize=12)
ax5.set_ylabel("m/s")
ax5.set_xlabel("Jahrzehnt")

# ── Plot 6: Rollender Mittelwert ─────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
rolling_365 = wind_clean.rolling(window=365, center=True).mean()
ax6.plot(wind_clean, linewidth=0.3, color="steelblue", alpha=0.4, label="Tagesmittel")
ax6.plot(rolling_365, linewidth=1.5, color="tomato", label="365-Tage Mittelwert")
ax6.set_title("Rollender 365-Tage Mittelwert", fontsize=12)
ax6.set_ylabel("m/s")
ax6.set_xlabel("Jahr")
ax6.legend()

plt.suptitle("Datenvisualisierung – Windgeschwindigkeit Würzburg (FM)",
             fontsize=15, fontweight="bold", y=1.01)
plt.show()
