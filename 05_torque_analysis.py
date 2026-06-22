import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

plt.figure(figsize=(8,5))
plt.hist(df["Torque [Nm]"], bins=30)

plt.title("Torque Distribution")
plt.xlabel("Torque (Nm)")
plt.ylabel("Frequency")

plt.show()