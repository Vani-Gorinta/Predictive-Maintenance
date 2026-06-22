import pandas as pd

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Rolling Mean Features
df["AirTemp_RollingMean"] = df["Air temperature [K]"].rolling(window=10, min_periods=1).mean()

df["ProcessTemp_RollingMean"] = df["Process temperature [K]"].rolling(window=10, min_periods=1).mean()

df["Torque_RollingMean"] = df["Torque [Nm]"].rolling(window=10, min_periods=1).mean()

df["ToolWear_RollingMean"] = df["Tool wear [min]"].rolling(window=10, min_periods=1).mean()

# Save dataset
df.to_csv("data/processed_data.csv", index=False)

print("Rolling Mean Features Added Successfully!")