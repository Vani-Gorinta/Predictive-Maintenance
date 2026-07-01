import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed_data.csv")

# Rolling Standard Deviation Features (window = 10)
df["AirTemp_RollingStd"] = df["Air temperature [K]"].rolling(window=10).std()
df["ProcessTemp_RollingStd"] = df["Process temperature [K]"].rolling(window=10).std()
df["Torque_RollingStd"] = df["Torque [Nm]"].rolling(window=10).std()
df["ToolWear_RollingStd"] = df["Tool wear [min]"].rolling(window=10).std()

# Fill NaN values created by rolling
df.fillna(0, inplace=True)

# Save updated dataset
df.to_csv("data/processed_data.csv", index=False)

print("Rolling Standard Deviation Features Added Successfully!")