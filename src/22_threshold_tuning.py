import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed_data.csv")

for col in ["Type", "Shift", "Production_Demand"]:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# -----------------------------
# Target
# -----------------------------
y = df["Machine failure"]

# -----------------------------
# Features
# -----------------------------
X = df.drop(columns=["Machine failure", "UDI", "Product ID"], errors="ignore")

# Clean feature names
X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/lightgbm_model.pkl")

# -----------------------------
# Predict Probabilities
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Threshold Tuning
# -----------------------------
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

results = []

print("\nThreshold Tuning Results\n")

for t in thresholds:

    y_pred = (y_prob >= t).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold : {t}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print()

    results.append([t, precision, recall, f1])

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=["Threshold", "Precision", "Recall", "F1 Score"]
)

results_df.to_csv(
    "results/threshold_tuning_results.csv",
    index=False
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6,4))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning")
plt.legend()
plt.grid(True)

plt.savefig("results/threshold_tuning.png")

plt.show()

print("Threshold tuning completed successfully!")