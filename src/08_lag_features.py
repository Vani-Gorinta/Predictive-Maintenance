import pandas as pd

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Lag Features
df["Torque_Lag1"] = df["Torque [Nm]"].shift(1)
df["ToolWear_Lag1"] = df["Tool wear [min]"].shift(1)

# Fill first missing value
df.fillna(0, inplace=True)

# Save
df.to_csv("data/processed_data.csv", index=False)

print("Lag Features Added Successfully!")