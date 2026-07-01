import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Encode categorical columns if needed
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

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train LightGBM Model
model = LGBMClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = model.predict(X_test)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()

print("Confusion Matrix saved successfully!")