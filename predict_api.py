from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI(title="Credit Card Fraud Detection API")

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float 
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float 
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float 
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    

@app.post("/predict")
def predict(transaction: Transaction):
    input_data = np.array([[value for value in transaction.dict().values()]])
    data_scaled = scaler.transform(input_data)
    prediction = model.predict(data_scaled)
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    logging.info(f"Input: {transaction.dict()} | Prediction: {result}")
    return {"prediction": result}
