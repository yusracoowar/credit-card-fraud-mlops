import joblib
import pandas as pd 
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')
df = pd.read_sql('SELECT * FROM transactions', con=engine)

df = df.drop_duplicates()
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved")
