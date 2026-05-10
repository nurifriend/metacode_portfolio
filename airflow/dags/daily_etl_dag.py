from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pendulum

kst = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 10, tzinfo=kst),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_s3_to_dw_otto_events',
    default_args=default_args,
    description='[Containerized] S3 to Fact -> Dim Update -> Biz Mart ETL',
    schedule_interval='0 7 * * *', 
    catchup=False,
    tags=['spark', 'etl', 'bi'],
) as dag:

    # --- [Task 1] S3 Raw 데이터를 DW Fact 테이블로 추가 적재 ---
    t1 = BashOperator(
        task_id='append_fact_events',
        bash_command="python /opt/airflow/scripts/spark/s3_to_postgres.py --date {{ ds }}",
        env={
            "AWS_ACCESS_KEY_ID": "{{ conn.aws_s3_conn.login }}",
            "AWS_SECRET_ACCESS_KEY": "{{ conn.aws_s3_conn.password }}",
            "DB_HOST": "{{ conn.postgres_dw_conn.host }}",
            "DB_PORT": "{{ conn.postgres_dw_conn.port }}",
            "DB_USER": "{{ conn.postgres_dw_conn.login }}",
            "DB_PASSWORD": "{{ conn.postgres_dw_conn.password }}",
        },
        append_env=True, 
    )

    # --- [Task 2] 신규 발견된 상품(aid)에 대한 메타데이터 생성 및 추가 ---
    t2 = BashOperator(
        task_id='update_dim_product',
        bash_command="python /opt/airflow/scripts/spark/update_dim_product.py",
        env={
            "DB_HOST": "{{ conn.postgres_dw_conn.host }}",
            "DB_PORT": "{{ conn.postgres_dw_conn.port }}",
            "DB_USER": "{{ conn.postgres_dw_conn.login }}",
            "DB_PASSWORD": "{{ conn.postgres_dw_conn.password }}",
        },
        append_env=True,
    )

    # --- [Task 3] Fact + Dim 데이터를 결합하여 비즈니스 지표 산출 ---
    t3 = BashOperator(
        task_id='aggregate_biz_metrics',
        bash_command="python /opt/airflow/scripts/spark/biz_aggregation.py --date {{ ds }}",
        env={
            "DB_HOST": "{{ conn.postgres_dw_conn.host }}",
            "DB_PORT": "{{ conn.postgres_dw_conn.port }}",
            "DB_USER": "{{ conn.postgres_dw_conn.login }}",
            "DB_PASSWORD": "{{ conn.postgres_dw_conn.password }}",
        },
        append_env=True,
    )

    # 팩트 체크: 데이터 흐름에 따른 엄격한 순차 실행
    t1 >> t2 >> t3