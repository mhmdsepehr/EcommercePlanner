# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE = Path(__file__).parent
CONFIG_PATH = BASE / "config.json"
FORECAST_PATH = BASE / "forecast.xlsx"

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5  # Sat/Sun

def expand_shift_calendar(ranges):
    out = {}
    for blk in ranges:
        start = pd.to_datetime(blk["start"]).date()
        end = pd.to_datetime(blk["end"]).date()
        shifts = int(blk.get("shifts", 2))
        cur = start
        while cur <= end:
            if pd.Timestamp(cur).weekday() < 5:
                out[str(cur)] = shifts
            cur += timedelta(days=1)
    return out

def next_day_demand(df: pd.DataFrame, i: int) -> float:
    if i + 1 < len(df):
        nd = float(df.loc[i+1, "forecast_demand"])
        if nd > 0:
            return nd
    td = float(df.loc[i, "forecast_demand"])
    if td > 0:
        return td
    mean = float(df["forecast_demand"].mean())
    return max(1.0, mean)

# ---------------------------------------------------
# Load inputs
# ---------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

df_forecast = pd.read_excel(FORECAST_PATH)
df_forecast.columns = [c.strip().lower() for c in df_forecast.columns]
if "date" not in df_forecast.columns or "forecast_demand" not in df_forecast.columns:
    raise ValueError("forecast.xlsx must have columns: 'date', 'forecast_demand'.")

df_forecast["date"] = pd.to_datetime(df_forecast["date"])
df_forecast = df_forecast.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------
# Streamlit Sidebar - Interactive Controls
# ---------------------------------------------------
st.sidebar.title("Planner Adjustments")

demand_scale = st.sidebar.slider(
    "Demand Adjustment (%)",
    min_value=50,
    max_value=150,
    value=100,
    step=1
)

productivity = st.sidebar.slider(
    "Productivity (units/hour)",
    min_value=1.0,
    max_value=100.0,
    value=float(cfg["productivity_units_per_hour"]),
    step=0.1
)

backlog_threshold = st.sidebar.slider(
    "Backlog Threshold (days)",
    min_value=0.5,
    max_value=10.0,
    value=float(cfg["backlog_threshold_days"]),
    step=0.1
)

extra_no_work = st.sidebar.multiselect(
    "Extra No-Work Dates",
    options=df_forecast["date"].dt.date
)

# ---------------------------------------------------
# Parameters (adjusted)
# ---------------------------------------------------
prod = float(productivity)
hc_max = int(cfg["headcount_max_per_shift"])
reg_hours = float(cfg["max_hours_per_shift"])
max_ot = float(cfg["max_ot_hours_per_person"])
holidays = set(pd.to_datetime(cfg.get("holidays", [])).date).union(extra_no_work)
first_backlog_units = float(cfg["first_day_backlog"])
backlog_threshold_days = float(backlog_threshold)
max_days_over_threshold = int(cfg.get("max_days_over_threshold", 0))
min_backlog_days = float(cfg["min_backlog_days"])
regular_rate = float(cfg["regular_rate"])
ot_mult = float(cfg["ot_multiplier"])
weekend_premium = float(cfg["weekend_premium"])
weekend_activation_cost = float(cfg.get("weekend_activation_cost", 0.0))
max_output_per_shift = float(cfg.get("max_output_per_shift", float("inf")))
shift_overrides = expand_shift_calendar(cfg.get("shift_calendar", []))

# Apply demand scaling
df_forecast["forecast_demand"] = df_forecast["forecast_demand"] * (demand_scale / 100)

# ---------------------------------------------------
# Core Functions
# ---------------------------------------------------
def capacity_units(shifts: int, hours_per_person: float) -> float:
    per_shift_reg_cap = min(hc_max * prod * hours_per_person, max_output_per_shift)
    return shifts * per_shift_reg_cap

def ot_capacity_units(shifts: int, ot_hours: float) -> float:
    if ot_hours <= 0:
        return 0.0
    per_shift_ot_cap = hc_max * prod * ot_hours
    if np.isfinite(max_output_per_shift):
        remaining = max(0.0, max_output_per_shift - (hc_max * prod * reg_hours))
        per_shift_ot_cap = min(per_shift_ot_cap, remaining)
    return shifts * per_shift_ot_cap

def cost_for_units(units_reg: float, units_ot: float, is_wkend: bool) -> float:
    cpu_reg = regular_rate / prod
    cpu_ot = (regular_rate * ot_mult) / prod
    c = units_reg * cpu_reg + units_ot * cpu_ot
    if is_wkend:
        c *= weekend_premium
        if (units_reg + units_ot) > 0:
            c += weekend_activation_cost
    return c

def choose_plan_for_day(i, day, backlog_start_units, demand_today, days_over_counter):
    is_wkend = is_weekend(day)
    is_holiday = (day.date() in holidays)

    max_weekday_shifts = int(shift_overrides.get(str(day.date()), 1)) if not is_wkend else 0
    shifts_allowed = [0] if is_holiday else ([0, 1] if is_wkend else list(range(1, max_weekday_shifts + 1)))

    nd = next_day_demand(df_forecast, i)
    floor_units = min_backlog_days * nd if day.weekday() in range(0,4) else 0.0
    total_available_units = backlog_start_units + demand_today
    candidates = []

    for s in shifts_allowed:
        reg_units_cap = capacity_units(s, reg_hours) if s > 0 else 0.0
        req_units_min = max(0.0, total_available_units - floor_units)
        target_at_threshold_units = max(0.0, total_available_units - backlog_threshold_days * nd)
        must_hit_threshold = backlog_start_units / max(1.0, nd) > backlog_threshold_days and days_over_counter >= max_days_over_threshold
        if must_hit_threshold:
            req_units_min = max(req_units_min, target_at_threshold_units)

        reg_units = min(reg_units_cap, min(req_units_min, total_available_units))
        need_after_reg = max(0.0, req_units_min - reg_units)
        ot_units_cap = ot_capacity_units(s, max_ot) if s > 0 else 0.0
        ot_units = min(ot_units_cap, min(need_after_reg, total_available_units - reg_units))
        processed = reg_units + ot_units

        backlog_end_units = max(0.0, total_available_units - processed)
        backlog_end_days = backlog_end_units / max(1.0, nd)
        feasible = not (must_hit_threshold and backlog_end_days > backlog_threshold_days + 1e-9)

        cost = cost_for_units(reg_units, ot_units, is_wkend)
        reg_hpp = reg_units / (prod * hc_max * s) if s > 0 else 0.0
        ot_hpp = ot_units / (prod * hc_max * s) if s > 0 else 0.0

        candidates.append(dict(
            shifts=s,
            is_weekend=is_wkend,
            reg_units=reg_units,
            ot_units=ot_units,
            processed=processed,
            reg_hpp=reg_hpp,
            ot_hpp=ot_hpp,
            backlog_end_units=backlog_end_units,
            backlog_end_days=backlog_end_days,
            feasible=feasible,
            total_cost=cost
        ))

    best = min(candidates, key=lambda c: (0 if c["feasible"] else 1, c["total_cost"], c["backlog_end_days"]))
    headcount = hc_max * best["shifts"]
    return {"chosen": best, "headcount": headcount}

# ---------------------------------------------------
# Generate Plan
# ---------------------------------------------------
results = []
backlog_units = first_backlog_units
days_over = 0

for i, row in df_forecast.iterrows():
    day = row["date"]
    demand = float(row["forecast_demand"])
    pick = choose_plan_for_day(i, day, backlog_units, demand, days_over)
    ch = pick["chosen"]

    total_avail = backlog_units + demand
    processed = min(ch["processed"], total_avail)
    backlog_units = max(0.0, total_avail - processed)
    nd = next_day_demand(df_forecast, i)
    backlog_days = backlog_units / max(1.0, nd)

    days_over = days_over + 1 if backlog_days > backlog_threshold_days + 1e-9 else 0

    is_wkend = is_weekend(day)
    reg_cost = (regular_rate / prod) * ch["reg_units"]
    ot_cost = ((regular_rate * ot_mult) / prod) * ch["ot_units"]
    wknd_cost = weekend_activation_cost if is_wkend and ch["shifts"] > 0 else 0.0
    if is_wkend and ch["shifts"] > 0:
        reg_cost *= weekend_premium
        ot_cost *= weekend_premium
    total_cost = reg_cost + ot_cost + wknd_cost

    results.append({
        "date": day,
        "forecast_demand": demand,
        "processed_units": processed,
        "backlog_end_units": backlog_units,
        "backlog_end_days": round(backlog_days, 3),
        "shifts": ch["shifts"],
        "headcount": int(hc_max * ch["shifts"]),
        "hours_per_person_per_shift": round(ch["reg_hpp"], 2),
        "ot_hours_per_person": round(ch["ot_hpp"], 2),
        "weekend_activated": bool(is_wkend and ch["shifts"] > 0),
        "violated_threshold": bool(backlog_days > backlog_threshold_days + 1e-9),
        "regular_cost": round(reg_cost, 2),
        "ot_cost": round(ot_cost, 2),
        "weekend_activation_cost": round(wknd_cost, 2),
        "daily_cost_total": round(total_cost, 2)
    })

df_plan = pd.DataFrame(results)

# ---------------------------------------------------
# Streamlit Outputs
# ---------------------------------------------------
st.title("📦 E-Commerce Fulfillment Planner")
st.subheader("Daily Plan Table")
st.dataframe(df_plan)

# KPIs
total_cost = df_plan["daily_cost_total"].sum()
total_hours = sum((row['hours_per_person_per_shift'] + row['ot_hours_per_person']) * row['headcount'] for idx, row in df_plan.iterrows())
st.subheader("KPIs")
st.metric("Total Cost ($)", round(total_cost,2))
st.metric("Total Hours", round(total_hours,2))

# Plot chart
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.bar(df_plan["date"], df_plan["processed_units"], label="Processed Units")
ax1.set_ylabel("Processed Units")
ax2 = ax1.twinx()
ax2.plot(df_plan["date"], df_plan["backlog_end_days"], marker="o", color="red", label="Backlog (Days)")
ax2.axhline(y=backlog_threshold_days, linestyle="--", color="orange", label=f"SLA Threshold ({backlog_threshold_days}d)")
ax2.set_ylabel("Backlog (Days)")
ax1.set_xlabel("Date")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.autofmt_xdate()
st.pyplot(fig)

# Download buttons
st.subheader("Download Plan")
csv_data = df_plan.to_csv(index=False)
st.download_button("Download CSV", csv_data, file_name="adjusted_plan.csv", mime="text/csv")
excel_path = BASE / "adjusted_plan.xlsx"
df_plan.to_excel(excel_path, index=False)
st.download_button("Download Excel", open(excel_path, "rb").read(),
                   file_name="adjusted_plan.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

