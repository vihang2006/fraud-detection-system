from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import SessionLocal, TransactionRecord
from datetime import datetime
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import os

# ---------------------------------------------------
# Create FastAPI App
# ---------------------------------------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Load ML Model
# ---------------------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# ---------------------------------------------------
# Load Metrics
# ---------------------------------------------------
metrics_path = os.path.join(os.path.dirname(__file__), "metrics.pkl")

try:
    metrics = joblib.load(metrics_path)
except:
    metrics = None

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class Transaction(BaseModel):
    features: conlist(float, min_length=30, max_length=30)

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction, threshold: float = 0.7):

    data = np.array(transaction.features).reshape(1, -1)
    probability = model.predict_proba(data)[0][1]

    risk_level = "Low"
    if probability > threshold:
        risk_level = "High"
    elif probability > threshold / 2:
        risk_level = "Medium"

    db = SessionLocal()
    record = TransactionRecord(
        fraud_probability=float(probability),
        risk_level=risk_level,
        timestamp=datetime.utcnow()
    )
    db.add(record)
    db.commit()
    db.close()

    return {
        "fraud_probability": float(probability),
        "risk_level": risk_level
    }

@app.get("/transactions")
def get_transactions():
    db = SessionLocal()
    records = db.query(TransactionRecord).all()
    db.close()

    return [
        {
            "id": r.id,
            "fraud_probability": r.fraud_probability,
            "risk_level": r.risk_level,
            "timestamp": r.timestamp
        }
        for r in records
    ]

@app.get("/analytics")
def fraud_analytics():
    db = SessionLocal()
    records = db.query(TransactionRecord).all()
    db.close()

    total = len(records)
    high_risk = sum(1 for r in records if r.risk_level == "High")
    medium_risk = sum(1 for r in records if r.risk_level == "Medium")
    low_risk = sum(1 for r in records if r.risk_level == "Low")

    avg_probability = (
        sum(r.fraud_probability for r in records) / total
        if total > 0 else 0
    )

    return {
        "total_transactions": total,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "average_fraud_probability": avg_probability
    }

@app.get("/model-metrics")
def get_model_metrics():
    if metrics:
        return metrics
    return {"message": "Metrics not available"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

