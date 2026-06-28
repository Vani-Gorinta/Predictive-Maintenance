import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Encode categorical columns
le_type = LabelEncoder()
df["Type"] = le_type.fit_transform(df["Type"])

le_shift = LabelEncoder()
df["Shift"] = le_shift.fit_transform(df["Shift"])

le_demand = LabelEncoder()
df["Production_Demand"] = le_demand.fit_transform(df["Production_Demand"])

# Save processed dataset
df.to_csv("data/processed_data.csv", index=False)

print("Data preprocessing completed successfully!")