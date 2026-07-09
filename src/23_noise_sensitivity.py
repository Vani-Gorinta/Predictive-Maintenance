import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed_data.csv")

# Encode categorical columns
for col in ["Type", "Shift", "Production_Demand"]:
    if col in df.columns and df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop(columns=["Machine failure", "UDI", "Product ID"], errors="ignore")
y = df["Machine failure"]

# Clean feature names
X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("models/lightgbm_model.pkl")

noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
f1_scores = []

print("\nNoise Sensitivity Analysis\n")

for noise in noise_levels:

    X_noise = X_test.copy()

    numeric_cols = X_noise.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        std = X_noise[col].std()
        X_noise[col] += np.random.normal(0, noise * std, len(X_noise))

    y_pred = model.predict(X_noise)

    score = f1_score(y_test, y_pred)

    f1_scores.append(score)

    print(f"Noise Level: {noise:.2f}   F1 Score: {score:.3f}")

# Save Results
results = pd.DataFrame({
    "Noise Level": noise_levels,
    "F1 Score": f1_scores
})

results.to_csv("results/noise_sensitivity_results.csv", index=False)

# Plot
plt.figure(figsize=(6,4))
plt.plot(noise_levels, f1_scores, marker="o")

plt.title("Noise Sensitivity Analysis")
plt.xlabel("Noise Level")
plt.ylabel("F1 Score")
plt.grid(True)

plt.savefig("results/noise_sensitivity.png")
plt.show()

print("\nNoise Sensitivity Analysis Completed Successfully!")