import pandas as pd 
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')
df = pd.read_sql('SELECT * FROM transactions', con=engine)
print("Loaded data")

df = df.drop_duplicates()
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

importance = model.feature_importances_
columns = list(X.columns)

plt.figure(figsize=(10,6))
plt.barh(columns, importance)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
