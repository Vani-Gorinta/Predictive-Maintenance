import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

plt.figure(figsize=(8,5))
plt.hist(df["Air temperature [K]"], bins=30)

plt.title("Air Temperature Distribution")
plt.xlabel("Air Temperature (K)")
plt.ylabel("Frequency")

plt.savefig("results/air_temperature_distribution.png")
plt.show()