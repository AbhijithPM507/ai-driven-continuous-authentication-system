from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

df = pd.read_csv("final_ml_dataset.csv")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save
joblib.dump(model, "behavior_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved successfully.")
