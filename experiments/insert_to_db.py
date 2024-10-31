TABLE_NAME = "logs_table_created_by_dohyung"
        
# poetry add python-dotenv
# poetry add sqlalchemy==1.4.52
# poetry add psycopg2-binary

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env')
    postgresql_connection_info = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    postgresql_connection_info += '/postgres'


    engine = create_engine(
        url=postgresql_connection_info,
        echo=True,
        # isolation_level=
    )

    Base = declarative_base()

    class LogsTable(Base):
        '''
            - IP: string
            - date_and_time: timestamp
            - http_methods: string
            - endpoint: string
            - status_code: string
        '''
        
        __tablename__ = TABLE_NAME
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        ip = Column(String)
        time = Column(DateTime)
        http_methods = Column(String)
        endpoint = Column(String)
        status_code = Column(String)
        
    # 테이블이 존재하지 않으면 생성하는 코드
    Base.metadata.create_all(engine)
    
    # DBeaver에서 확인