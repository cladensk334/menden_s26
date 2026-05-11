import pandas as pd
import numpy as np

# Semikolon als Trennzeichen, Leerzeichen um Spaltennamen entfernen
df = pd.read_csv(
    '/Users/clara/Desktop/uni_dreck/Menden_S26/temperatur_raw_05705_akt/produkt_tu_termin_20241031_20260503_05705.txt',
    sep=';',
    skipinitialspace=True
)
df.columns = df.columns.str.strip()

#vorschau
print(df.head())
print(df.dtypes)
print(df.shape)