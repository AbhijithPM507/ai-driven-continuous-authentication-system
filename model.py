from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("final_ml_dataset.csv")

# Keep only genuine user data
genuine_df = df[df["label"] == 0].copy()

# Drop label
X = genuine_df.drop(columns=["label"])

# Replace infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values
X.fillna(0, inplace=True)

# OPTIONAL (better): remove rows that still contain NaN
# X = X.dropna()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train One-Class SVM
model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

model.fit(X_scaled)

# Save
joblib.dump(model, "behavior_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("One-Class model trained successfully.")