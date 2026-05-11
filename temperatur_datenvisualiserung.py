#PACF/ACF 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(d1,  lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF der 1. Differenz')

plot_pacf(d1, lags=40, ax=axes[1], method='ywm', alpha=0.05)
axes[1].set_title('PACF der 1. Differenz')

plt.tight_layout()
plt.show()

