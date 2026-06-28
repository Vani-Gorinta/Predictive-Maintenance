import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed_data.csv")

# Convert Production Demand into numeric values
demand_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Production_Demand_Value"] = df["Production_Demand"].map(demand_map)

# Context Fusion Features
df["Torque_Load"] = df["Torque [Nm]"] * df["Energy_Load_Index"]

df["Wear_Demand"] = (
    df["Tool wear [min]"] *
    df["Production_Demand_Value"]
)

df["Temperature_Humidity"] = (
    df["Process temperature [K]"] *
    df["Ambient_Humidity"]
)

# Save
df.to_csv("data/processed_data.csv", index=False)

print("Contextual Data Fusion Completed Successfully!")