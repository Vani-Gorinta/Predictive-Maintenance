import pandas as pd

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Create new features
df["Temperature_Difference"] = (
    df["Process temperature [K]"] -
    df["Air temperature [K]"]
)

df["Wear_Torque_Index"] = (
    df["Tool wear [min]"] *
    df["Torque [Nm]"]
)

print(df[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Temperature_Difference",
    "Tool wear [min]",
    "Torque [Nm]",
    "Wear_Torque_Index"
]].head())

# Save new dataset
df.to_csv("data/processed_data.csv", index=False)

print("\nProcessed dataset saved successfully!")