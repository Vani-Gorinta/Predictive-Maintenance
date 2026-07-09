import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score
)

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/processed_data.csv")

# Remove non-numeric columns if present
df = df.drop(columns=["UDI", "Product ID", "Type"], errors="ignore")

# ==========================
# Features and Target
# ==========================
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Clean feature names for LightGBM
X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Apply SMOTE
# ==========================
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# ==========================
# Train LightGBM Model
# ==========================
model = LGBMClassifier(random_state=42)

model.fit(X_train_smote, y_train_smote)

# ==========================
# Prediction Probabilities
# ==========================
y_scores = model.predict_proba(X_test)[:, 1]

# ==========================
# Precision-Recall Curve
# ==========================
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_scores
)

# Average Precision Score
ap_score = average_precision_score(y_test, y_scores)

# Plot Curve
display = PrecisionRecallDisplay(
    precision=precision,
    recall=recall
)

display.plot()

plt.title(f"Precision-Recall Curve (AP = {ap_score:.3f})")
plt.grid(True)

# Save Figure
plt.savefig("results/precision_recall_curve.png", dpi=300)

plt.show()

print("\nAverage Precision Score :", round(ap_score, 3))
print("Precision-Recall Curve saved successfully!")