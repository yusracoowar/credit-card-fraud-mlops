import pandas as pd 
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')
df = pd.read_sql('SELECT * FROM transactions', con=engine)
print("Step 1: Data Loaded")

df = df.drop_duplicates()
print(" Step 2: Removed duplicates")

X = df.drop('Class', axis=1)
y = df['Class']
print("Step 3: Split into features and labels")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Step 4: Scaled the features")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Step 5: Applied SMOTE")

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Step 6: Split data into train and test")

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Step 7: Model trained")

y_pred = model.predict(X_test)
print("Step 8: Model evaluated")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
