import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Encode categorical columns
encoders = {}

for col in ["Type", "Shift", "Production_Demand"]:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Features and target
X = df.drop(columns=["Machine failure", "UDI", "Product ID"], errors="ignore")
y = df["Machine failure"]

# Clean feature names
X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)

# Train-test split
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

# Train model
model = LGBMClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Save model
joblib.dump(model, "models/lightgbm_model.pkl")

# Save encoders
joblib.dump(encoders, "models/label_encoders.pkl")

print("✅ Model saved successfully!")
print("✅ Encoders saved successfully!")