# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
#paths
DATA_PATH = "dataset/credit.csv"
OUTPUT_PATH = "dataset/credit_preprocessed.csv"

# -------------------------
# Load dataset
# -------------------------
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit(1)

# -------------------------
# Separate features and target
# -------------------------
if 'Class' not in df.columns:
    print("Error: Dataset must have a 'Class' column as target")
    exit(1)

X = df.drop('Class', axis=1)
y = df['Class']

# -------------------------
# Encode categorical variables (if any)
# -------------------------
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------
# Combine with target and save
# -------------------------
processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
processed_df.to_csv(OUTPUT_PATH, index=False)

print(f"Preprocessing complete. Processed file saved as '{OUTPUT_PATH}'")
