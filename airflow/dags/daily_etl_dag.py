from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime, timedelta
import pendulum

kst = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 5, tzinfo=kst),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_events_etl_clean',
    default_args=default_args,
    description='KST 아침 7시 외부 스크립트 호출형 ETL',
    schedule_interval='0 7 * * *', 
    catchup=False,
    tags=['spark', 'etl'],
) as dag:

    run_spark_job = SSHOperator(
        task_id='trigger_spark_script',
        ssh_conn_id='vm_ssh_conn',
        command=(
            "/home/ubuntu/spark/bin/spark-submit "
            "--packages org.apache.hadoop:hadoop-aws:3.3.4,org.postgresql:postgresql:42.6.0 "
            "/home/ubuntu/metacode_portfolio/airflow/scripts/spark/s3_to_postgres.py "
            "--date {{ ds }}"
        ),
        cmd_timeout=3600
    )
    
    run_spark_job