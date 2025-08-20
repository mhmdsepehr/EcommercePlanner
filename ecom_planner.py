import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths (adjust if needed)
# ---------------------------------------------------
BASE = Path(r"C:\Users\msepehr\OneDrive - Michael Kors (USA), Inc\Desktop\Python\Ecom Planner")
CONFIG_PATH = BASE / "config.json"
FORECAST_PATH = BASE / "forecast.xlsx"
OUTPUT_XLSX = BASE / "plan_output.xlsx"

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def is_weekend(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5  # 5=Sat, 6=Sun

def expand_shift_calendar(ranges):
    """
    Expand ranges like:
      [{"start":"YYYY-MM-DD","end":"YYYY-MM-DD","shifts":2}, ...]
    into a dict: { "YYYY-MM-DD": 2, ... } (weekdays only).
    """
    out = {}
    for blk in ranges:
        start = pd.to_datetime(blk["start"]).date()
        end = pd.to_datetime(blk["end"]).date()
        shifts = int(blk.get("shifts", 2))
        cur = start
        while cur <= end:
            if pd.Timestamp(cur).weekday() < 5:  # only weekdays
                out[str(cur)] = shifts
            cur += timedelta(days=1)
    return out

def next_day_demand(df: pd.DataFrame, i: int) -> float:
    """Forecast for the next day; fall back to today's, then mean>0."""
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

df = pd.read_excel(FORECAST_PATH)
df.columns = [c.strip().lower() for c in df.columns]
if "date" not in df.columns or "forecast_demand" not in df.columns:
    raise ValueError("forecast.xlsx must have columns: 'date', 'forecast_demand'.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Params
prod = float(cfg["productivity_units_per_hour"])                     # units per person-hour
hc_max = int(cfg["headcount_max_per_shift"])
reg_hours = float(cfg["max_hours_per_shift"])
max_ot = float(cfg["max_ot_hours_per_person"])
holidays = set(pd.to_datetime(cfg.get("holidays", [])).date)
first_backlog_units = float(cfg["first_day_backlog"])

backlog_threshold_days = float(cfg["backlog_threshold_days"])
max_days_over_threshold = int(cfg.get("max_days_over_threshold", 0))
min_backlog_days = float(cfg["min_backlog_days"])  # can be 0.8

regular_rate = float(cfg["regular_rate"])
ot_mult = float(cfg["ot_multiplier"])
weekend_premium = float(cfg["weekend_premium"])
weekend_activation_cost = float(cfg.get("weekend_activation_cost", 0.0))

max_output_per_shift = float(cfg.get("max_output_per_shift", float("inf")))  # cap per shift if given

# Optional weekday 2-shift windows (peak season, etc.)
shift_overrides = expand_shift_calendar(cfg.get("shift_calendar", []))

# ---------------------------------------------------
# Core: pick the cheapest feasible plan for the day
# ---------------------------------------------------
def capacity_units(shifts: int, hours_per_person: float) -> float:
    # Per-shift cap (regular)
    per_shift_reg_cap = min(hc_max * prod * hours_per_person, max_output_per_shift)
    return shifts * per_shift_reg_cap

def ot_capacity_units(shifts: int, ot_hours: float) -> float:
    # Remaining headroom against per-shift cap is already respected in capacity_units.
    if ot_hours <= 0:
        return 0.0
    per_shift_ot_cap = hc_max * prod * ot_hours
    # optionally cap OT by per-shift max_output as well:
    if np.isfinite(max_output_per_shift):
        # If regular already used reg_hours, any remaining cap for OT per shift is:
        remaining = max(0.0, max_output_per_shift - (hc_max * prod * reg_hours))
        per_shift_ot_cap = min(per_shift_ot_cap, remaining)
    return shifts * per_shift_ot_cap

def cost_for_units(units_reg: float, units_ot: float, is_wkend: bool) -> float:
    # cost per unit (regular, OT)
    cpu_reg = regular_rate / prod
    cpu_ot = (regular_rate * ot_mult) / prod
    c = units_reg * cpu_reg + units_ot * cpu_ot
    if is_wkend:
        c *= weekend_premium
        if (units_reg + units_ot) > 0:
            c += weekend_activation_cost
    return c

def choose_plan_for_day(
    i: int,
    day: pd.Timestamp,
    backlog_start_units: float,
    demand_today: float,
    days_over_counter: int,
) -> dict:
    """
    Evaluate candidate actions and return the cheapest one that keeps SLA,
    using this order:
      - Weekdays: prefer 1 shift regular hours; add OT if needed; if still not enough and allowed, go to 2 shifts.
      - Weekends: default no work; activate 1 shift only if needed to avoid/limit SLA breach. Use OT before extra shift.
    """
    is_wkend = is_weekend(day)
    is_holiday = (day.date() in holidays)

    # Allowed max shifts today
    max_weekday_shifts = int(shift_overrides.get(str(day.date()), 1)) if not is_wkend else 0
    # Weekends default to 0 but can be activated with up to 1 shift:
    max_weekend_shifts = 1

    # If holiday -> force no work
    if is_holiday:
        shifts_allowed = [0]
    else:
        shifts_allowed = [0] if is_wkend else list(range(1, max_weekday_shifts + 1))
        # For weekend, consider 0 and 1 shift options
        if is_wkend:
            shifts_allowed = [0, 1]  # 0 preferred, 1 if needed

    # Compute next-day demand for backlog "days" & weekday floor logic
    nd = next_day_demand(df, i)

    # Backlog floors:
    if day.weekday() in (0, 1, 2, 3):  # Mon-Thu
        floor_units = min_backlog_days * nd
    elif day.weekday() == 4:           # Fri
        # We may take the weekend off; allow floor=0 on Fri (as requested)
        floor_units = 0.0
    else:                               # Sat/Sun
        floor_units = 0.0

    total_available_units = backlog_start_units + demand_today

    # Candidate generator:
    candidates = []

    for s in shifts_allowed:
        # Regular-only plan
        reg_units_cap = capacity_units(s, reg_hours) if s > 0 else 0.0
        # Required units to not go below floor:
        req_units_min = max(0.0, total_available_units - floor_units)
        # If SLA breach risk AND we have exceeded max_days_over_threshold already,
        # we must target backlog <= threshold today if possible.
        # Target backlog at threshold:
        target_at_threshold_units = max(0.0, total_available_units - backlog_threshold_days * nd)
        must_hit_threshold = False

        backlog_if_no_work_days = (total_available_units) / max(1.0, nd)
        if backlog_if_no_work_days > backlog_threshold_days and days_over_counter >= max_days_over_threshold:
            must_hit_threshold = True
            req_units_min = max(req_units_min, target_at_threshold_units)

        # REG first
        reg_units = min(reg_units_cap, min(req_units_min, total_available_units))
        # If REG not enough to meet req_units_min, consider OT (up to max_ot)
        need_after_reg = max(0.0, req_units_min - reg_units)
        ot_units_cap = ot_capacity_units(s, max_ot) if s > 0 else 0.0
        ot_units = min(ot_units_cap, min(need_after_reg, total_available_units - reg_units))

        # If still cannot meet req_units_min → allow more (still bounded by capacity)
        processed = reg_units + ot_units

        # For weekdays, if still above threshold and allowed 2 shifts (override),
        # try bumping to 2 shifts (only for weekdays).
        if (not is_wkend) and s < max_weekday_shifts and processed + 1e-6 < req_units_min:
            # Try with s = max_weekday_shifts
            s2 = max_weekday_shifts
            reg_cap2 = capacity_units(s2, reg_hours)
            reg2 = min(reg_cap2, min(req_units_min, total_available_units))
            need2 = max(0.0, req_units_min - reg2)
            ot_cap2 = ot_capacity_units(s2, max_ot)
            ot2 = min(ot_cap2, min(need2, total_available_units - reg2))
            processed2 = reg2 + ot2
            # Compute costs for both, pick cheaper feasible; if both infeasible, keep better one
            # Cost
            c1 = cost_for_units(reg_units, ot_units, is_wkend=False)
            c2 = cost_for_units(reg2, ot2, is_wkend=False)
            # Backlog if applied
            end_units1 = total_available_units - processed
            end_days1 = end_units1 / max(1.0, nd)
            end_units2 = total_available_units - processed2
            end_days2 = end_units2 / max(1.0, nd)

            # Choose the lower-cost that satisfies must-hit-threshold (if applicable). Tie-break by end_days then cost.
            def score(feasible, cost, end_days):
                return (0 if feasible else 1, end_days, cost)

            feas1 = (processed >= req_units_min - 1e-6)
            feas2 = (processed2 >= req_units_min - 1e-6)
            pick_second = score(feas2, c2, end_days2) < score(feas1, c1, end_days1)

            if pick_second:
                s = s2
                reg_units, ot_units, processed = reg2, ot2, processed2

        # Compute end-of-day backlog & cost
        backlog_end_units = total_available_units - processed
        backlog_end_units = max(0.0, backlog_end_units)  # never negative
        backlog_end_days = backlog_end_units / max(1.0, nd)

        # If must hit threshold but we couldn't, mark infeasible (we’ll still keep to choose the least-bad)
        feasible = True
        if must_hit_threshold and (backlog_end_days > backlog_threshold_days + 1e-9):
            feasible = False

        # Build cost
        cost = cost_for_units(reg_units, ot_units, is_wkend)
        # Hours per person per shift (REG and OT)
        if s > 0 and hc_max > 0:
            reg_hpp = reg_units / (prod * hc_max * s)
            ot_hpp = ot_units / (prod * hc_max * s)
        else:
            reg_hpp = 0.0
            ot_hpp = 0.0

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

    # Select candidate:
    # 1) Prefer feasible; 2) lowest cost; 3) tie-break by lower backlog days; 4) prefer fewer weekend shifts.
    def key_fn(c):
        penalty_weekend = 0 if not c["is_weekend"] or c["shifts"] == 0 else 0.001  # tiny nudge away from weekends
        return (0 if c["feasible"] else 1, c["total_cost"] + penalty_weekend, c["backlog_end_days"])

    best = min(candidates, key=key_fn)

    # Compute headcount (we assume using up to hc_max; cost/unit doesn’t change with headcount,
    # but for the plan we report full headcount when running a shift)
    headcount = hc_max * best["shifts"]

    return {
        "chosen": best,
        "headcount": headcount,
        "candidates": candidates
    }

# ---------------------------------------------------
# Run the plan
# ---------------------------------------------------
results = []
backlog_units = first_backlog_units
days_over = 0

for i, row in df.iterrows():
    day = row["date"]
    demand = float(row["forecast_demand"])
    if np.isnan(demand):
        demand = 0.0

    pick = choose_plan_for_day(
        i=i,
        day=day,
        backlog_start_units=backlog_units,
        demand_today=demand,
        days_over_counter=days_over
    )
    ch = pick["chosen"]

    # Update backlog with correct math
    total_avail = backlog_units + demand
    processed = min(ch["processed"], total_avail)
    backlog_units = max(0.0, total_avail - processed)

    # Backlog days uses next day’s demand (as discussed)
    nd = next_day_demand(df, i)
    backlog_days = backlog_units / max(1.0, nd)

    # Track SLA streak
    if backlog_days > backlog_threshold_days + 1e-9:
        days_over += 1
    else:
        days_over = 0

    # Cost breakdown for reporting
    is_wkend = is_weekend(day)
    reg_cost = (regular_rate / prod) * ch["reg_units"]
    ot_cost = ((regular_rate * ot_mult) / prod) * ch["ot_units"]
    if is_wkend and ch["shifts"] > 0:
        reg_cost *= weekend_premium
        ot_cost *= weekend_premium
        wknd_cost = weekend_activation_cost
    else:
        wknd_cost = 0.0

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

df_out = pd.DataFrame(results)

# ---------------------------------------------------
# Save Excel
# ---------------------------------------------------
df_out.to_excel(OUTPUT_XLSX, index=False)
print(f"✅ Plan saved to {OUTPUT_XLSX}")

# ---------------------------------------------------
# Visualization (bars + secondary Y for backlog days)
# ---------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(df_out["date"], df_out["processed_units"], label="Processed Units")
ax1.set_xlabel("Date")
ax1.set_ylabel("Processed Units")
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.plot(df_out["date"], df_out["backlog_end_days"], marker="o", label="Backlog (Days)",  color="tab:red")
ax2.axhline(y=backlog_threshold_days, linestyle="--", label=f"SLA Threshold ({backlog_threshold_days}d)")
ax2.set_ylabel("Backlog (Days)")
ax2.tick_params(axis="y")

fig.suptitle("E-commerce Fulfillment Plan — Cost-Optimized")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
