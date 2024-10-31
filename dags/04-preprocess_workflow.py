# import pandas as pd

# if __name__ == "__main__":
#     print(f"pandas version: {pd.__version__}")
    
# from airflow import DAG
# from airflow.operators.python import PythonOperator

# with DAG(
    
# ) as dag:
#     def read_logs(**kwargs):
#         pass
    
#     def preprocess_logs(**kwargs):
#         pass
    
#     def insert_logs_to_db(**kwagrs):
#         pass
    
#     read_logs_task = PythonOperator()
#     preprocess_logs_task = PythonOperator()
#     insert_logs_to_db_task = PythonOperator()
    

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

with DAG(
    dag_id="04-preprocess_workflow",
    schedule="0 9 * * *",
    start_date=pendulum.datetime(2024, 10, 31, tz="Asia/Seoul"),
    catchup=False
) as dag:
    def read_logs(**kwargs):
        pass
    
    def preprocess_logs(**kwargs):
        pass
    
    def insert_logs_to_db(**kwagrs):
        pass
    
    read_logs_task = PythonOperator(
        task_id="read_logs_task",
        python_callable=read_logs,
    )
    preprocess_logs_task = PythonOperator(
        task_id="preprocess_logs_task",
        python_callable=preprocess_logs,
    )
    insert_logs_to_db_task = PythonOperator(
        task_id="insert_logs_to_db_task",
        python_callable=insert_logs_to_db,
    )