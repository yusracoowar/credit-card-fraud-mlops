import pandas as pd
from sqlalchemy import create_engine
import pymysql

csv_path = '/home/vboxuser/Documents/creditcard.csv'
chunk_size = 1000

engine = create_engine('mysql+pymysql://mluser:mlpass@localhost/fraud_detection')

for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    chunk.columns = [col.strip() for col in chunk.columns]
    chunk.to_sql(name='transactions', con=engine, if_exists='append', index=False) 
    print(f"Inserted {len(chunk)} rows")









