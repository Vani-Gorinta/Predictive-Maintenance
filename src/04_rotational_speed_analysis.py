import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

plt.figure(figsize=(8,5))
plt.hist(df["Rotational speed [rpm]"], bins=30)

plt.title("Rotational Speed Distribution")
plt.xlabel("Rotational Speed (RPM)")
plt.ylabel("Frequency")

plt.savefig("results/rotational_speed_distribution.png")
plt.show()