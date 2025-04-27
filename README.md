# credit-card-fraud-mlops
End-to-end ML pipeline using FastAPI, SMOTE, and Random Forest for fraud detection

This project implements a complete MLOps pipeline to detect fraudulent credit card transactions using a machine learning model (Random Forest). The pipeline includes data ingestion, preprocessing (with SMOTE for balancing), model training, deployment via FastAPI, and basic model monitoring via logging.

Install dependencies using:

```bash
pip install -r requirements.txt

# Download the dataset

Due to file size limits, the dataset is not included

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Save the file in the root directory as:
`creditcard.csv`


# Train the model
python train_model.py

# Visualise feature importance
python plot_importance.py

# Start the API
uvicorn predict_api:app --reload

# Use the API

Go to http://127.0.0.1:8000/docs to access the Swagger UI

Enter transaction data manually and get predictions

Example output:

{
  "Time": 1.0,
  "V1": 0.5,
  "V2": 1.2,
  "V3": -0.3,
  "V4": 0.0,
  "V5": 0.6,
  "V6": -1.1,
  "V7": 0.3,
  "V8": -0.4,
  "V9": 0.9,
  "V10": 0.0,
  "V11": 1.0,
  "V12": 0.2,
  "V13": -0.1,
  "V14": 0.4,
  "V15": 0.8,
  "V16": 0.3,
  "V17": -0.5,
  "V18": 0.1,
  "V19": -0.2,
  "V20": 0.5,
  "V21": 0.6,
  "V22": -0.7,
  "V23": 0.0,
  "V24": 0.1,
  "V25": 0.2,
  "V26": -0.3,
  "V27": 0.7,
  "V28": 0.1,
  "Amount": 100.0
{
"prediction": "not fraud"
}
