import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Target column
y = df["Machine failure"]

# Features
X = df.drop(columns=["Machine failure"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed Successfully!")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")