import numpy as np
from scipy.optimize import minimize

# Define parameters
battery_capacity = 100.0  # kWh, maximum capacity of the battery
battery_min_soc = 10.0    # kWh, minimum state of charge
battery_max_soc = 100.0   # kWh, maximum state of charge
charge_efficiency = 0.95  # Efficiency for charging
discharge_efficiency = 0.95  # Efficiency for discharging
max_charge_power = 50.0   # kW, maximum charging rate
max_discharge_power = 50.0  # kW, maximum discharging rate
time_steps = 24           # Optimization period (e.g., 24 hours)
time_interval = 1         # Duration of each time step (hours)
initial_soc = 50.0        # kWh, initial state of charge

# Simulated demand and renewable generation (kW for each hour)
power_demand = np.array([40, 50, 60, 55, 45, 50, 70, 80, 65, 50, 40, 30, 35, 45, 55, 65, 75, 85, 70, 60, 50, 40, 30, 20])
renewable_generation = np.array([30, 40, 50, 60, 55, 50, 60, 70, 60, 50, 40, 30, 25, 30, 40, 50, 60, 70, 65, 55, 45, 35, 25, 15])

# Define the cost function
def cost_function(power_flows):
    charge_power = power_flows[:time_steps]
    discharge_power = power_flows[time_steps:]
    grid_import = power_demand - (renewable_generation + discharge_power - charge_power)
    return np.sum(np.maximum(grid_import, 0))  # Minimize grid import cost

# Define constraints
def soc_constraints(power_flows):
    charge_power = power_flows[:time_steps]
    discharge_power = power_flows[time_steps:]
    soc = np.zeros(time_steps)
    soc[0] = initial_soc + (charge_power[0] * charge_efficiency - discharge_power[0] / discharge_efficiency) * time_interval
    for t in range(1, time_steps):
        soc[t] = soc[t-1] + (charge_power[t] * charge_efficiency - discharge_power[t] / discharge_efficiency) * time_interval

    # Constraints: SOC limits, charge/discharge limits
    return np.concatenate((
        soc - battery_min_soc,             # SOC should not go below minimum
        battery_max_soc - soc,             # SOC should not exceed maximum
        max_charge_power - charge_power,   # Charging power limit
        charge_power,                      # Charging power >= 0
        max_discharge_power - discharge_power,  # Discharging power limit
        discharge_power                    # Discharging power >= 0
    ))

# Bounds for charging and discharging power
bounds = [(0, max_charge_power) for _ in range(time_steps)] + [(0, max_discharge_power) for _ in range(time_steps)]

# Initial guess (no charging or discharging)
initial_guess = np.zeros(2 * time_steps)

# Solve the optimization problem
result = minimize(cost_function, initial_guess, bounds=bounds, constraints={'type': 'ineq', 'fun': soc_constraints})

# Results
if result.success:
    optimal_power_flows = result.x
    charge_power = optimal_power_flows[:time_steps]
    discharge_power = optimal_power_flows[time_steps:]
    soc = np.zeros(time_steps)
    soc[0] = initial_soc + (charge_power[0] * charge_efficiency - discharge_power[0] / discharge_efficiency) * time_interval
    for t in range(1, time_steps):
        soc[t] = soc[t-1] + (charge_power[t] * charge_efficiency - discharge_power[t] / discharge_efficiency) * time_interval

    print("Optimization successful!")
    print(f"Charge Power (kW): {charge_power}")
    print(f"Discharge Power (kW): {discharge_power}")
    print(f"State of Charge (kWh): {soc}")
else:
    print("Optimization failed:", result.message)
