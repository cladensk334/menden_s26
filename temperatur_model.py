forecast = model.get_forecast(steps=10)

# Punktprognose
fc_mean = forecast.predicted_mean

# Konfidenzintervall (alpha=0.05 -> 95%-Kontingenzintervall)
fc_ci = forecast.conf_int(alpha=0.05)

result = pd.concat([fc_mean.rename('Prognose'), fc_ci], axis=1)
result.columns = ['Prognose', 'KI_unten', 'KI_oben']
print(result.round(2))