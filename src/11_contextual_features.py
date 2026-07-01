import pandas as pd
import numpy as np

# Load processed dataset
df = pd.read_csv("data/processed_data.csv")

# For reproducibility
np.random.seed(42)

# Simulated contextual features
df["Ambient_Humidity"] = np.random.randint(30, 91, size=len(df))
df["Energy_Load_Index"] = np.round(np.random.uniform(0.4, 1.0, size=len(df)), 2)
df["Production_Demand"] = np.random.choice(
    ["Low", "Medium", "High"],
    size=len(df),
    p=[0.3, 0.5, 0.2]
)
df["Shift"] = np.random.choice(["Day", "Night"], size=len(df))
df["Days_Since_Maintenance"] = np.random.randint(1, 181, size=len(df))

# Save updated dataset
df.to_csv("data/processed_data.csv", index=False)

print("Contextual Features Added Successfully!")