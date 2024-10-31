# import pandas as pd

# if __name__ == "__main__":
#     logger.info(f"pandas version: {pd.__version__}")
    
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with DAG(
    dag_id="04-preprocess_workflow",
    schedule="0 9 * * *",
    start_date=pendulum.datetime(2024, 10, 31, tz="Asia/Seoul"),
    catchup=False
) as dag:
    def read_logs(**kwargs):
        import pandas as pd
        # ⭐️ 특정 패키지가 Task 내에서만 쓰일 경우, Task 내에 import하는 것이 좋다.(권장)⭐️ 
        logger.info(f"pandas version: \n{pd.__version__}\n\n")
        
        # pandas로 데이터 불러오기
        df = pd.read_csv(filepath_or_buffer="data/logs/log-241001.txt", delimiter=" ")
        logger.info(f"df: \n{df}\n\n")
        
        # 불러온 데이터의 모습 살펴보기
        desctiption_df = df.describe()
        logger.info(f"description about df: \n{desctiption_df}\n\n")
        
        # 불러온 데이터의 column 확인하기
        columns_df = df.columns
        logger.info(f"columns: \n{columns_df}\n\n")
        # 컬럼의 이름이 지정되지 않은 모습이 보인다.
        
        # 컬럼의 이름을 지정하기 전에, 컬럼의 개수 확인하기
        length_column_of_df = df.shape[1]
        logger.info(f"length_column_of_df: \n{length_column_of_df}\n\n")
        
        # 컬럼에 이름 부여하고, 확인하기
        df.columns = ['IP', "drop", "date_and_time", "drop2", "http_methods_and_endpoint", "drop3", "status_code"]
        logger.info(f"df: \n{df}\n\n")
        columns_df = df.columns
        logger.info(f"columns: \n{columns_df}\n\n")
        
        # 불필요한 컬럼 제거하기
        truncated_df = df.drop(labels=["drop", "drop2", "drop3"], axis=1, inplace=False)

        # http_method와 endpoint를 추출
        logger.info(f"truncated_df.http_methods_and_endpoint: \n{truncated_df.http_methods_and_endpoint}\n\n")
        http_method_and_endpoint = truncated_df.http_methods_and_endpoint
        http_method = http_method_and_endpoint.map(lambda x: x.split(" ")[0])
        endpoint = http_method_and_endpoint.map(lambda x: x.split(" ")[1])
        logger.info(f"http_method: \n{http_method}\n\n")
        logger.info(f"endpoint: \n{endpoint}\n\n")
        
        # 기존 컬럼 제거
        truncated_df.drop(labels="http_methods_and_endpoint", axis=1, inplace=True)
        logger.info(f"truncated_df: \n{truncated_df}\n\n")
        
        # 새로운 컬럼 추가
        merged_http_method_and_endpoint = pd.concat(objs=[http_method, endpoint], axis=1, ignore_index=True)
        logger.info(merged_http_method_and_endpoint)
        merged_http_method_and_endpoint.columns = ["http_method", "endpoint"]
        logger.info(merged_http_method_and_endpoint)
        
        merge_df = pd.concat(objs=[truncated_df, merged_http_method_and_endpoint], axis=1, ignore_index=True)
        logger.info(f"merge_df: \n{merge_df}\n\n")
        merge_df.columns = ["IP", "timestamp", "status_code", "http_method", "endpoint"]
        logger.info(f"merge_df: \n{merge_df}\n\n")
        
        # timestamp에서 괄호 제거
        merge_df['timestamp'] = merge_df['timestamp'].str.strip(to_strip='[]')
        logger.info(f"merge_df: \n{merge_df}\n\n")
        
        # 타입 변경
        merge_df['timestamp'] = pd.to_datetime(arg=merge_df['timestamp'], format='%d/%b/%Y:%H:%M:%S')
        logger.info(f"merge_df: \n{merge_df}\n\n")
        
        # 나머지 열을 문자열 타입으로 변환
        merge_df = merge_df.astype({
            'IP': 'string',
            'http_method': 'string',
            'endpoint': 'string',
            'status_code': 'string'
        })
        logger.info(f"merge_df.dtypes: {merge_df.dtypes}")
    
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