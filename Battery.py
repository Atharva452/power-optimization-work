import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Parameters
time_slots = 96  # 96 slots for a day (15 min each)
battery_capacity = 100  # kWh
max_power = 100  # kW, max charging/discharging power
initial_soc = 60  # kWh, initial state-of-charge
price_forecast = np.random.uniform(10, 20, time_slots)  # Simulated electricity price ($/MWh)

# Decision variables
charge = [LpVariable(f"charge_{t}", 0, max_power) for t in range(time_slots)]
discharge = [LpVariable(f"discharge_{t}", 0, max_power) for t in range(time_slots)]
soc = [LpVariable(f"soc_{t}", 0, battery_capacity) for t in range(time_slots)]

# Problem definition
problem = LpProblem("Battery_Storage_Optimization", LpMaximize)

# Objective: Maximize profit (or minimize cost)
problem += lpSum(
    price_forecast[t] * (discharge[t] - charge[t]) * 0.25  # Revenue from discharge, cost for charge
    for t in range(time_slots)
)

# Constraints
for t in range(time_slots):
    # SOC balance without efficiency
    if t == 0:
        problem += soc[t] == initial_soc + charge[t] - discharge[t]
    else:
        problem += soc[t] == soc[t - 1] + charge[t] - discharge[t]

    # SOC must remain within limits
    problem += soc[t] >= 0
    problem += soc[t] <= battery_capacity

# Solve the problem
problem.solve()

# Results
print("Status:", problem.status)
print("Optimized Schedule:")
for t in range(time_slots):
    print(
        f"Slot {t+1}: Charge = {charge[t].varValue:.2f} kW, "
        f"Discharge = {discharge[t].varValue:.2f} kW, "
        f"SOC = {soc[t].varValue:.2f} kWh"
    )

