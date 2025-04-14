# credit-card-fraud-mlops
End-to-end ML pipeline using FastAPI, SMOTE, and Random Forest for fraud detection

This project implements a complete MLOps pipeline to detect fraudulent credit card transactions using a machine learning model (Random Forest). The pipeline includes data ingestion, preprocessing (with SMOTE for balancing), model training, deployment via FastAPI, and basic model monitoring via logging.

Install dependencies using:

```bash
pip install -r requirements.txt

# Train the model
python train_model.py

# (Optional) Visualise feature importance
python plot_importance.py

# Start the API
uvicorn predict_api:app --reload


