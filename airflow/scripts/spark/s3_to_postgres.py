import sys
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

# ==========================================
# 1. 파라미터 및 시간 여행(Time Offset) 세팅
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=True, help='Airflow logical date in YYYY-MM-DD')
args = parser.parse_args()
AIRFLOW_DATE = args.date

# 🔸 OFFSET 수정: 2026-05-06 실행 시 2022-07-30 데이터를 가져오기 위해 1375 적용
OFFSET_DAYS = 1375
airflow_date_obj = datetime.strptime(AIRFLOW_DATE, "%Y-%m-%d")
real_target_date_obj = airflow_date_obj - timedelta(days=OFFSET_DAYS)
REAL_TARGET_DATE = real_target_date_obj.strftime("%Y-%m-%d")

print(f"🚀 Airflow 스케줄 날짜: {AIRFLOW_DATE} | 🕰️ 실제 추출할 데이터 날짜: {REAL_TARGET_DATE}")

# 🔸 필터링을 위한 타겟 날짜의 시작/끝 밀리초(ms) 직접 계산
# .timestamp()는 시스템 타임존에 의존하므로, 명확하게 처리하기 위해 타임존 고려 필요 없음
target_ts_start = int(real_target_date_obj.timestamp() * 1000)
target_ts_end = target_ts_start + (24 * 60 * 60 * 1000) - 1

# ==========================================
# 2. 환경 변수 및 DB 세팅
# ==========================================
env_path = "/home/ubuntu/metacode_portfolio/airflow/scripts/spark/.env"
load_dotenv(env_path)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY:
    print(f"❌ 에러: {env_path} 경로에서 .env 파일을 로드하지 못했습니다.")
    sys.exit(1)

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_URL = "jdbc:postgresql://localhost:5432/otto_dw"
DB_PROPERTIES = {"user": DB_USER, "password": DB_PASSWORD, "driver": "org.postgresql.Driver"}

# ==========================================
# 3. Spark 세션 초기화
# ==========================================
spark = SparkSession.builder \
    .appName(f"Daily_ETL_{REAL_TARGET_DATE}") \
    .config("spark.sql.session.timeZone", "UTC") \
    .getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", AWS_ACCESS_KEY) 
hadoop_conf.set("fs.s3a.secret.key", AWS_SECRET_KEY) 
hadoop_conf.set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# ==========================================
# 4. 데이터 로드 및 타겟 날짜 필터링 (최적화 로직)
# ==========================================
S3_PATH = "s3a://otto-data-786802935144-ap-northeast-2-an/topics/otto-train-data/partition=0/"
df_raw = spark.read.parquet(S3_PATH)

# 🔸 [변경 포인트] 변환 과정에서 발생하는 타임존 오차를 없애기 위해 원본 'ts'로 직접 필터링
print(f"🎯 필터링 범위(ms): {target_ts_start} ~ {target_ts_end}")
daily_events = df_raw.filter((F.col("ts") >= target_ts_start) & (F.col("ts") <= target_ts_end))

# 적재를 위한 컬럼 가공 (필터링이 끝난 후에 수행)
daily_events = daily_events.withColumn("ts_datetime", (F.col("ts") / 1000).cast(TimestampType())) \
                           .withColumn("date_id", F.lit(REAL_TARGET_DATE).cast("date"))

# 🔸 실제 데이터 건수 확인 로그
actual_count = daily_events.count()
print(f"📊 [DEBUG] 필터링된 실제 데이터 개수: {actual_count}")

if actual_count == 0:
    print("⚠️ 경고: 필터링 결과 데이터가 0건입니다. OFFSET을 다시 확인하세요.")
    # S3 전체 데이터 범위를 다시 출력하여 힌트를 얻습니다.
    df_raw.withColumn("dt", F.to_date((F.col("ts")/1000).cast(TimestampType()))) \
          .select(F.min("dt"), F.max("dt")).show()
    spark.stop()
    sys.exit(0) # 혹은 1로 설정하여 Airflow를 실패처리 가능

# 연산 효율을 위해 캐싱
daily_events.cache()

# ==========================================
# 5. [Phase 1] fact_events DB 적재
# ==========================================
daily_events.write.jdbc(url=DB_URL, table="fact_events", mode="append", properties=DB_PROPERTIES)
print(f"✅ fact_events 적재 완료")

# ==========================================
# 6. [Phase 2] fact_sessions 가공 및 DB 적재
# ==========================================
daily_sessions = daily_events.groupBy("session_id").agg(
    F.min("ts_datetime").alias("session_start_time"),
    F.max("ts_datetime").alias("session_end_time"),
    F.count("aid").alias("total_events"),
    F.countDistinct("aid").alias("unique_items"),
).withColumn("date_id", F.lit(REAL_TARGET_DATE).cast("date"))

daily_sessions.write.jdbc(url=DB_URL, table="fact_sessions", mode="append", properties=DB_PROPERTIES)
print(f"✅ fact_sessions 적재 완료")

# 종료
daily_events.unpersist()
spark.stop()