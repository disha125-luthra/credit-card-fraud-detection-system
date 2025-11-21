# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv("dataset/credit.csv")  # must contain 'Class' column (0=legit,1=fraud)
X = df.drop("Class", axis=1)
y = df["Class"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale numeric features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train model ---
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Save model + scaler + features list ---
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X.columns), "model/features.pkl")

print("✅ Model trained and saved successfully.")
