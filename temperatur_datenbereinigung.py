#fehlerwerte auslesen
# -999 -> als NaN
df['TT_TER'] = df['TT_TER'].replace(-999.0, np.nan)

# Anazhl Fehlwerte 
print(df['TT_TER'].isna().sum())

#aggregation auf tagesebene
df['date'] = pd.to_datetime(
    df['MESS_DATUM'].astype(str).str[:8],
    format='%Y%m%d'
)
daily = df.groupby('date')['TT_TER'].mean().rename('temp_mean')
daily = daily.asfreq('D')

print(daily.head(10))
print(f"Fehlende Tage: {daily.isna().sum()}")

daily_clean = daily.interpolate(method='linear')

print(daily_clean.describe())