import pandas as pd 
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE 

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')

df = pd.read_sql('SELECT * FROM transactions', con=engine)
print("Loaded data from the database")

df = df.drop_duplicates()
print("Cleaned duplicate rows")

X = df.drop('Class', axis=1)
y = df['Class']
print("Separated features and labels")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled the features")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Applied SMOTE to balance the dataset")

print("Final Class counts after SMOTE:")
print(pd.Series(y_resampled).value_counts())
