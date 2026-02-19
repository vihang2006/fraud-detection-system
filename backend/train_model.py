import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../data/creditcard.csv")

print("Dataset loaded successfully")
print("Fraud cases:", df["Class"].sum())

# Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount column
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Model 1 — Logistic Regression
# ----------------------------
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train, y_train)

log_probs = log_model.predict_proba(X_test)[:, 1]
log_auc = roc_auc_score(y_test, log_probs)

print("\nLogistic Regression ROC-AUC:", log_auc)

# ----------------------------
# Model 2 — Random Forest
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=30,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print("Random Forest ROC-AUC:", rf_auc)

# ----------------------------
# Compare & Select Best
# ----------------------------
if rf_auc > log_auc:
    best_model = rf_model
    best_name = "Random Forest"
    best_auc = rf_auc
else:
    best_model = log_model
    best_name = "Logistic Regression"
    best_auc = log_auc

print("\nBest Model Selected:", best_name)
print("Best ROC-AUC:", best_auc)

# Save best model
joblib.dump(best_model, "model.pkl")

# Save metrics separately
metrics = {
    "logistic_auc": float(log_auc),
    "random_forest_auc": float(rf_auc),
    "best_model": best_name,
    "best_auc": float(best_auc)
}

joblib.dump(metrics, "metrics.pkl")

print("\nModel and metrics saved successfully.")

