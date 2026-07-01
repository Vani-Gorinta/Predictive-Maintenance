import pandas as pd

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Change Features
df["AirTemp_Change"] = df["Air temperature [K]"].diff()
df["ProcessTemp_Change"] = df["Process temperature [K]"].diff()
df["Torque_Change"] = df["Torque [Nm]"].diff()
df["ToolWear_Change"] = df["Tool wear [min]"].diff()

# Replace NaN
df.fillna(0, inplace=True)

# Save
df.to_csv("data/processed_data.csv", index=False)

print("Change Features Added Successfully!")