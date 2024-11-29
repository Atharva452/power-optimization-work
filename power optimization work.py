import pandas as pd
import pulp as pl
import numpy as np

# Read and prepare data
plants_data = pd.DataFrame([
    ['P1', 10.4, 18.8, 7.00],
    ['P2', 17.0, 30.8, 7.00],
    ['P3', 21.4, 39.0, 7.00],
    ['P4', 15.2, 27.6, 1.38],
    ['P5', 35.2, 63.9, 1.35],
    ['P6', 28.7, 52.2, 1.36],
    ['P7', 40.5, 73.7, 1.34],
    ['P8', 283.3, 515.1, 3.77],
    ['P9', 5.3, 9.6, 3.51],
    ['P10', 10.3, 18.8, 3.48],
    ['P11', 6.4, 11.6, 3.48],
    ['P12', 5.2, 9.5, 4.01],
    ['P13', 11.2, 20.4, 2.08],
    ['P14', 35.8, 65.1, 2.08],
    ['P15', 5.0, 9.1, 3.00],
    ['P16', 34.2, 62.1, 1.26],
    ['P17', 65.9, 119.9, 3.00],
    ['P18', 22.0, 40.0, 3.02],
    ['P19', 312.1, 416.2, 6.14],
    ['P20', -500.0, 500.0, 10.00]
], columns=['plant', 'min', 'max', 'price'])

def solve_max_load_min_cost(weight_load=0.7, weight_cost=0.3):
    """
    Solve the multi-objective optimization problem:
    - Maximize load transmission
    - Minimize cost
    Using weighted sum method
    """
    # Create optimization model
    model = pl.LpProblem("Power_Plant_Max_Load_Min_Cost", pl.LpMaximize)
    
    # Decision variables: power output for each plant
    plant_vars = pl.LpVariable.dicts("Plant",
                                    plants_data['plant'],
                                    lowBound=0)
    
    # Calculate maximum possible total capacity
    total_max_capacity = sum(plants_data['max'])
    
    # Calculate maximum possible total cost
    max_possible_cost = sum(plants_data['max'] * plants_data['price'])
    
    # Multi-objective function:
    # Maximize load (normalized) - Minimize cost (normalized)
    model += (
        weight_load * pl.lpSum([plant_vars[plant] for plant in plants_data['plant']]) / total_max_capacity -
        weight_cost * pl.lpSum([plant_vars[plant] * price 
                              for plant, price in zip(plants_data['plant'], 
                                                    plants_data['price'])]) / max_possible_cost
    )
    
    # Constraints
    for index, row in plants_data.iterrows():
        plant = row['plant']
        # Plant capacity constraints
        model += plant_vars[plant] >= row['min']  # Minimum capacity
        model += plant_vars[plant] <= row['max']  # Maximum capacity
    
    # Solve the model
    model.solve()
    
    return model, plant_vars

# Solve the optimization problem
model, plant_vars = solve_max_load_min_cost()

# Process results
results = []
total_cost = 0
total_power = 0

for plant in plants_data['plant']:
    power_output = plant_vars[plant].value()
    if power_output is not None:
        plant_cost = power_output * float(plants_data[plants_data['plant'] == plant]['price'].iloc[0])
        total_cost += plant_cost
        total_power += power_output
        results.append({
            'Plant': plant,
            'Power_Output_MW': round(power_output, 2),
            'Cost_per_MW': float(plants_data[plants_data['plant'] == plant]['price'].iloc[0]),
            'Total_Cost': round(plant_cost, 2),
            'Utilization_%': round(power_output / float(plants_data[plants_data['plant'] == plant]['max'].iloc[0]) * 100, 1)
        })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Power_Output_MW', ascending=False)

# Print summary
print("\nOptimization Results:")
print(f"Status: {pl.LpStatus[model.status]}")
print(f"\nTotal Power Output: {round(total_power, 2)} MW")
print(f"Total Cost: ${round(total_cost, 2)}")
print(f"Average Cost per MW: ${round(total_cost/total_power, 2)}")

print("\nPlant Utilization Details:")
print(results_df.to_string(index=False))

# Calculate capacity utilization statistics
total_capacity = plants_data['max'].sum()
capacity_utilization = (total_power / total_capacity) * 100

print("\nSystem Statistics:")
print(f"Total System Capacity: {round(total_capacity, 2)} MW")
print(f"Capacity Utilization: {round(capacity_utilization,2)}%")
