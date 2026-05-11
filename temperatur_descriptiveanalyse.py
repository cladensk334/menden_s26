#stationärtests
from statsmodels.tsa.stattools import adfuller, kpss

# ADF-Test
result_adf = adfuller(daily_clean, regression='ct', autolag='AIC')

print("ADF-Test (Niveau):")
print(f"  Teststatistik : {result_adf[0]:.4f}")
print(f"  p-Wert        : {result_adf[1]:.4f}")
print(f"  Lags genutzt  : {result_adf[2]}")
print("  Kritische Werte:")
for key, val in result_adf[4].items():
    print(f"    {key}: {val:.4f}")

# KPSS-Test
result_kpss = kpss(daily_clean, regression='ct', nlags='auto')

print("\nKPSS-Test (Niveau):")
print(f"  Teststatistik : {result_kpss[0]:.4f}")
print(f"  p-Wert        : {result_kpss[1]:.4f}")

#ADF/KPSS
d1 = daily_clean.diff().dropna()

#Tests, aber differenzierte Reihe
result_adf_d1 = adfuller(d1, regression='c', autolag='AIC')
print(f"ADF (1. Diff.): p = {result_adf_d1[1]:.4f}")

result_kpss_d1 = kpss(d1, regression='c', nlags='auto')
print(f"KPSS (1. Diff.): p = {result_kpss_d1[1]:.4f}")


#modelselektion 
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

rows = []

for p in range(0, 6):      # AR-Ordnung
    for q in range(0, 6):  # MA-Ordnung
        try:
            m = ARIMA(daily_clean, order=(p, 1, q)).fit()
            rows.append({
                'p': p, 'q': q,
                'AIC': m.aic,
                'BIC': m.bic,
                'LogL': m.llf
            })
        except Exception:
            pass  # Manche Kombinationen konvergieren nicht

rdf = pd.DataFrame(rows)

print("Top 5 nach AIC:")
print(rdf.sort_values('AIC').head(5).to_string(index=False))

print("\nTop 5 nach BIC:")
print(rdf.sort_values('BIC').head(5).to_string(index=False))

#model schätzen 
model = ARIMA(daily_clean, order=(1, 1, 2)).fit()
print(model.summary())

#residualdiagnositk 
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_arch

resid = model.resid.dropna()

# Ljung-Box: Test auf Autokorrelation in den Residuen
# H0: keine Autokorrelation bis Lag k
lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
print(lb)

# Jarque-Bera: Test auf Normalverteilung
# H0: Residuen sind normalverteilt
jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
print(f"JB: Stat={jb_stat:.2f}, p={jb_p:.4f}")
print(f"Schiefe={jb_skew:.3f}, Kurtosis={jb_kurt:.3f}")

# Heteroskedastizitätstest
# H0: keine Heteroskedastizität
arch_stat, arch_p, _, _ = het_arch(resid)
print(f"ARCH-Test: Stat={arch_stat:.4f}, p={arch_p:.4f}")