import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Paths and data
BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.json"
FORECAST_PATH = BASE / "forecast.xlsx"

# Load config and forecast
import json
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

forecast_df = pd.read_excel(FORECAST_PATH)
forecast_df.columns = [c.strip().lower() for c in forecast_df.columns]
forecast_df = forecast_df.sort_values('date').reset_index(drop=True)
forecast_df['date'] = pd.to_datetime(forecast_df['date'])

# Streamlit UI
st.title("E-commerce Fulfillment Optimizer")
st.sidebar.header("Parameters")
productivity = st.sidebar.slider("Productivity (units/hour)", 1.0, 100.0, float(cfg['productivity_units_per_hour']), step=0.1)
hc_max = st.sidebar.slider("Headcount per shift", 1, int(cfg['headcount_max_per_shift']), int(cfg['headcount_max_per_shift']))
reg_hours_max = st.sidebar.slider("Max hours per shift", 1.0, 12.0, float(cfg['max_hours_per_shift']), step=0.1)
ot_hours_max = st.sidebar.slider("Max OT hours per person", 0.0, 8.0, float(cfg['max_ot_hours_per_person']), step=0.1)
backlog_threshold_days = st.sidebar.slider("Backlog threshold (days)", 0.1, 10.0, float(cfg['backlog_threshold_days']), step=0.1)
min_backlog_days = st.sidebar.slider("Min backlog days", 0.0, 5.0, float(cfg.get('min_backlog_days', 0.8)), step=0.1)
max_days_over_threshold = st.sidebar.slider("Max days allowed over backlog", 0, 10, int(cfg.get('max_days_over_threshold',0)))
weekend_premium = float(cfg.get('weekend_premium', 1.5))
weekend_activation_cost = float(cfg.get('weekend_activation_cost', 0.0))
regular_rate = float(cfg.get('regular_rate', 20))
ot_mult = float(cfg.get('ot_multiplier', 1.5))
first_day_backlog = float(cfg.get('first_day_backlog', 0))

# LP Model setup
dates = list(forecast_df['date'])
n_days = len(dates)
forecast = forecast_df['forecast_demand'].values

prob = LpProblem("Ecom_Fulfillment_Optimization", LpMinimize)

# Decision variables per day
reg_hours = [LpVariable(f"reg_hours_{i}", 0, hc_max*reg_hours_max) for i in range(n_days)]
ot_hours = [LpVariable(f"ot_hours_{i}", 0, hc_max*ot_hours_max) for i in range(n_days)]
weekend_active = [LpVariable(f"weekend_active_{i}", 0, 1) for i in range(n_days)]
slack = [LpVariable(f"slack_{i}", 0, None) for i in range(n_days)]
backlog = [LpVariable(f"backlog_{i}", 0, None) for i in range(n_days)]

# Constraints
for i in range(n_days):
    is_weekend = dates[i].weekday() >= 5
    prob += reg_hours[i] <= hc_max*reg_hours_max*(1 if not is_weekend else weekend_active[i])
    prob += ot_hours[i] <= hc_max*ot_hours_max*(1 if not is_weekend else weekend_active[i])
    prev_backlog = first_day_backlog if i==0 else backlog[i-1]
    prob += backlog[i] == prev_backlog + forecast[i] - (reg_hours[i]*productivity + ot_hours[i]*productivity)
    next_day_demand = forecast[i+1] if i+1 < n_days else forecast[i]
    prob += backlog[i]/max(1,next_day_demand) - slack[i] <= backlog_threshold_days
    if dates[i].weekday() < 4:
        prob += backlog[i] >= min_backlog_days*next_day_demand
    else:
        prob += backlog[i] >= 0
prob += lpSum(slack) <= max_days_over_threshold*backlog_threshold_days

# Objective: total cost
prob += lpSum([reg_hours[i]*regular_rate + ot_hours[i]*regular_rate*ot_mult + weekend_active[i]*weekend_activation_cost*weekend_premium for i in range(n_days)])

# Solve
prob.solve()

# Output results
results = []
for i in range(n_days):
    reg_h = value(reg_hours[i])
    ot_h = value(ot_hours[i])
    wa = value(weekend_active[i])
    backlog_units = value(backlog[i])
    processed_units = reg_h*productivity + ot_h*productivity
    results.append({
        'date': dates[i],
        'forecast_demand': forecast[i],
        'processed_units': processed_units,
        'backlog_end_units': backlog_units,
        'regular_hours': reg_h,
        'ot_hours': ot_h,
        'weekend_activated': bool(round(wa)),
        'total_cost': reg_h*regular_rate + ot_h*regular_rate*ot_mult + wa*weekend_activation_cost*weekend_premium
    })

df_out = pd.DataFrame(results)

st.subheader("Daily Plan")
st.dataframe(df_out)

st.download_button("Download CSV", df_out.to_csv(index=False), file_name="plan.csv")
st.download_button("Download Excel", df_out.to_excel(index=False, engine='openpyxl'), file_name="plan.xlsx")

# Charts
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(df_out['date'], df_out['processed_units'], label='Processed Units')
ax1.set_ylabel('Processed Units')
ax2 = ax1.twinx()
ax2.plot(df_out['date'], df_out['backlog_end_units']/np.maximum(1,df_out['forecast_demand']), color='red', marker='o', label='Backlog (Days)')
ax2.axhline(y=backlog_threshold_days, color='orange', linestyle='--', label='SLA Threshold')
ax2.set_ylabel('Backlog (Days)')
fig.autofmt_xdate()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
st.pyplot(fig)
