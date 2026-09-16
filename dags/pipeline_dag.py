from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess_data
from src.train_model import train
from src.evaluate_model import evaluate

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    preprocess = PythonOperator(
        task_id='data_preprocessing',
        python_callable=preprocess_data
    )

    train_model = PythonOperator(
        task_id='model_training',
        python_callable=train
    )

    evaluate_model = PythonOperator(
        task_id='model_evaluation',
        python_callable=evaluate
    )

    preprocess >> train_model >> evaluate_model