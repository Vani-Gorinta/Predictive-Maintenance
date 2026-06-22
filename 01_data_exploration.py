import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

failure_counts = df["Machine failure"].value_counts()

plt.figure(figsize=(6,4))
failure_counts.plot(kind="bar")

plt.title("Machine Failure Distribution")
plt.xlabel("Failure Status")
plt.ylabel("Count")

plt.show()