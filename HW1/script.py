import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from helper_functions import bass_model, bass_model_sim
import os


img_folder = 'img'
os.makedirs(img_folder, exist_ok=True) 

## 3: Generating full shipments data from revenue data
revenue_df = pd.read_excel('./data/revenue_irobot_worldwide_2012_2023.xlsx')
shipments_df = pd.read_excel('./data/irobot_shipments_worldwide_2014_2018.xlsx')

merged = pd.merge(
    revenue_df[revenue_df['year'].between(2014, 2018)],
    shipments_df[shipments_df['year'].between(2014, 2018)],
    on='year'
)
fixed_avg_price = (merged['revenue'] / merged['shipments']).mean()
missing_years = [2012, 2013, 2019, 2020, 2021, 2022, 2023]
rev_missing = revenue_df[revenue_df['year'].isin(missing_years)].copy()
rev_missing['shipments'] = rev_missing['revenue'] / fixed_avg_price
shipments_df = pd.concat([shipments_df, rev_missing[['year', 'shipments']]], ignore_index=True)
shipments_df = shipments_df.sort_values('year').reset_index(drop=True)
shipments_df['cum_shipments'] = shipments_df['shipments'].cumsum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Annual sales/shipments
ax1.plot(shipments_df['year'], shipments_df['shipments'], marker='o', linewidth=2.5, 
         markersize=8, label='Actual Yearly Sales')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Annual Sales (millions)')
ax1.set_title(' Annual Sales (2012-2023)')
ax1.grid(axis='y')
ax1.set_xticks(shipments_df['year'])
ax1.tick_params(axis='x', rotation=45)

# Cumulative sales/shipments
ax2.plot(shipments_df['year'], shipments_df['cum_shipments'], marker='o', linewidth=2.5, 
         markersize=8, label='Actual Cumulative Sales')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Cumulative Sales (millions)')
ax2.set_title(' Cumulative Sales (2012-2023)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(shipments_df['year'])
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'iRobot_annual_and_cumulative_data.png'), dpi=300, bbox_inches='tight')
plt.close()

## 4: Estimating Bass model parameters

# --- Fit Bass model ---
t = np.arange(len(shipments_df))
y = shipments_df['shipments'].values
p0, q0, M0 = 0.03, 0.38, y.sum()*2  # initial guesses
params, _ = curve_fit(bass_model, t, y, p0=[p0, q0, M0])
p_est, q_est, M_est = params
print(f"Estimated p: {p_est:.4f}, q: {q_est:.4f}, M: {M_est:.0f}")

S_fit, N_fit = bass_model_sim(p_est, q_est, M_est, len(shipments_df))

# --- Plot f(t) and F(t) ---
plt.figure(figsize=(12,6))

# yearly shipments f(t)
plt.subplot(1,2,1)
plt.plot(shipments_df['year'], shipments_df['shipments'], 'o-', label='Actual f(t)')
plt.plot(shipments_df['year'], S_fit, 's--', label='Bass f(t) fit')
plt.xlabel('Year')
plt.ylabel('Shipments (f(t))')
plt.title('Yearly Shipments')
plt.legend()
plt.grid(True)

# cumulative adoption F(t)
plt.subplot(1,2,2)
plt.plot(shipments_df['year'], shipments_df['shipments'].cumsum(), 'o-', label='Actual F(t)')
plt.plot(shipments_df['year'], N_fit, 's--', label='Bass F(t) fit')
plt.xlabel('Year')
plt.ylabel('Cumulative Shipments (F(t))')
plt.title('Cumulative Adoption')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'shipments_bass_fit.png'), dpi=300, bbox_inches='tight')
plt.close()

## 5: Predicting future adoption
# simulate adoption for 30 years from 2012
n_years = 30
S_pred, N_pred = bass_model_sim(p_est, q_est, M_est, n_years)

start_year = 2012
years_pred = list(range(start_year, start_year + n_years))

diffusion_df = pd.DataFrame({
    'year': years_pred,
    'predicted_new_adopters': S_pred,   # f(t)
    'predicted_cumulative_adopters': N_pred  # F(t)
})

print(diffusion_df.head(10))

plt.figure(figsize=(12,6))

# yearly new adopters f(t)
plt.subplot(1,2,1)
plt.plot(diffusion_df['year'], diffusion_df['predicted_new_adopters'], 'o-', label='f(t) – new adopters')
plt.xlabel('Year')
plt.ylabel('Estimated New Adopters')
plt.title('Predicted Yearly Adoption f(t)')
plt.legend()
plt.grid(True)

# cumulative adopters F(t)
plt.subplot(1,2,2)
plt.plot(diffusion_df['year'], diffusion_df['predicted_cumulative_adopters'], 'o-', label='F(t) – cumulative adopters')
plt.xlabel('Year')
plt.ylabel('Cumulative Adopters')
plt.title('Predicted Cumulative Adoption F(t)')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(os.path.join(img_folder, 'predicted_adoption_forecast.png'), dpi=300, bbox_inches='tight')
plt.close()

## 6: Global or country-specific

ownership_df = pd.read_excel('./data/ownership_rate_of_robots_by_country_2025.xlsx')
print(ownership_df)
ownership_df_sorted = ownership_df.sort_values('rate', ascending=True)

plt.figure(figsize=(10,6))

# Horizontal bar plot
plt.barh(ownership_df_sorted['country'], ownership_df_sorted['rate'])

# Add numbers at the end of bars
for index, value in enumerate(ownership_df_sorted['rate']):
    plt.text(value + 0.5, index, str(value), va='center')

# Remove x-axis labels and ticks
plt.xticks([])

plt.title('Robot Vacuum Ownership by Country (2025)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(os.path.join(img_folder, 'robot_ownership_by_country.png'), dpi=300, bbox_inches='tight')

plt.close()
## 7: Create adopters DataFrame ---
adopters_df = pd.DataFrame({
    'year': shipments_df['year'],
    'estimated_new_adopters': S_fit,   # f(t)
    'cumulative_adopters': N_fit       # F(t)
})

# print(adopters_df)

# : Optional: Forecast adoption beyond 2023 ---
future_years = 5  # number of years to forecast
total_periods = len(shipments_df) + future_years

S_forecast, N_forecast = bass_model_sim(p_est, q_est, M_est, total_periods)
years_forecast = list(shipments_df['year']) + list(range(shipments_df['year'].iloc[-1]+1, shipments_df['year'].iloc[-1]+future_years+1))

forecast_df = pd.DataFrame({
    'year': years_forecast,
    'estimated_new_adopters': S_forecast,
    'cumulative_adopters': N_forecast
})

# print(forecast_df)

plt.figure(figsize=(12,6))

# yearly new adopters f(t)
plt.subplot(1,2,1)
plt.plot(forecast_df['year'], forecast_df['estimated_new_adopters'], 'o-', label='Bass f(t) forecast')
plt.xlabel('Year')
plt.ylabel('Estimated New Adopters')
plt.title('Yearly Estimated Adopters f(t) Forecast')
plt.legend()
plt.grid(True)

# cumulative adoption F(t)
plt.subplot(1,2,2)
plt.plot(forecast_df['year'], forecast_df['cumulative_adopters'], 'o-', label='Bass F(t) forecast')
plt.xlabel('Year')
plt.ylabel('Cumulative Adopters')
plt.title('Cumulative Adoption F(t) Forecast')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(os.path.join(img_folder, 'forecast_adoption.png'), dpi=300, bbox_inches='tight')
plt.close()
