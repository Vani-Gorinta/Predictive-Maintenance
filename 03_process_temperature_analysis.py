import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

plt.figure(figsize=(8,5))
plt.hist(df["Process temperature [K]"], bins=30)

plt.title("Process Temperature Distribution")
plt.xlabel("Process Temperature (K)")
plt.ylabel("Frequency")

plt.show()