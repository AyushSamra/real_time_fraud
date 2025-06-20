from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("fraud_model.pkl")
app = FastAPI()

class Transaction(BaseModel):
    amount: float
    timestamp: str

@app.post("/predict")
def predict(txn: Transaction):
    txn_hour = pd.to_datetime(txn.timestamp).hour
    input_data = pd.DataFrame([[txn.amount, txn_hour]], columns=["amount", "txn_hour"])
    score = model.predict_proba(input_data)[0][1]
    return {
        "fraud_score": round(score, 3),
        "fraud_flag": int(score > 0.7)
    }
