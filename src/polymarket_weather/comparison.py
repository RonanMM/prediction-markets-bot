import pandas as pd, json
from processing import load_weather_daily

# Market implied per date
df = pd.read_csv('data/polymarket/london_snapshots.csv')
df = df[df['closed'] == False].drop_duplicates('condition_id')
df['probs'] = df['outcome_probs_json'].apply(json.loads)
df['p_yes'] = df['probs'].apply(lambda p: p.get('Yes', p.get('yes', 0)))
df['target_temp'] = df['question'].str.extract(r'be (?:between )?(\d+)°C').astype(float)
df = df.dropna(subset=['target_temp'])

implied = df.groupby('end_date_iso').apply(
    lambda g: (g['target_temp'] * g['p_yes']).sum() / g['p_yes'].sum()
    if g['p_yes'].sum() > 0.01 else None
).rename('implied_temp').reset_index()

# Forecast per date
wx = load_weather_daily('London')[['date_local','temp_max_c']].drop_duplicates('date_local')
wx.columns = ['end_date_iso', 'forecast_max_c']

merged = implied.merge(wx, on='end_date_iso', how='left')
merged['gap_c'] = merged['implied_temp'] - merged['forecast_max_c']
print(merged.to_string(index=False))