from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'yusra',
    'start_date': datetime(2024, 12, 1),
    'retries': 0
}

with DAG(
    dag_id='credit_pipeline_dag',
    default_args=default_args,
    description='My MLOps pipeline for credit card fraud detection',
    schedule_interval=None,
    catchup=False
) as dag:

    load_data = BashOperator(
        task_id='load_data',
        bash_command='python3 /home/vboxuser/Documents/load_data.py'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python3 /home/vboxuser/Documents/balance_data.py'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /home/vboxuser/Documents/train_model.py'
    )

    monitor_predictions = BashOperator(
        task_id='monitor_predictions',
        bash_command='python3 /home/vboxuser/Documents/predict_api.py'
    )

    load_data >> preprocess_data >> train_model >> monitor_predictions

