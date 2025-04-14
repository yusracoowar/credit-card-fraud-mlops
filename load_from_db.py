import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')

df = pd.read_sql('SELECT * FROM transactions', con=engine)

print("Data pulled from MariaDB successfully.")
print(df.head())



