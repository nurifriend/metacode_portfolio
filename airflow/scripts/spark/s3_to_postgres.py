import sys
import os
import argparse
from datetime import datetime, timedelta
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

# 🔸 OFFSET 수정: 2026-05- 실행 시 2022-07-30 데이터를 가져오기 위해 1379 적용
OFFSET_DAYS = 1379
airflow_date_obj = datetime.strptime(AIRFLOW_DATE, "%Y-%m-%d")
real_target_date_obj = airflow_date_obj - timedelta(days=OFFSET_DAYS)
REAL_TARGET_DATE = real_target_date_obj.strftime("%Y-%m-%d")

print(f"🚀 Airflow 스케줄 날짜: {AIRFLOW_DATE} | 🕰️ 실제 추출할 데이터 날짜: {REAL_TARGET_DATE}")

target_ts_start = int(real_target_date_obj.timestamp() * 1000)
target_ts_end = target_ts_start + (24 * 60 * 60 * 1000) - 1

# ==========================================
# 2. 환경 변수 및 DB 세팅 (Airflow가 주입해 줄 예정)
# ==========================================
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY:
    print("❌ 에러: AWS 환경변수가 주입되지 않았습니다. Airflow 설정을 확인하세요.")
    sys.exit(1)

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")

DB_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/otto_dw"
DB_PROPERTIES = {"user": DB_USER, "password": DB_PASSWORD, "driver": "org.postgresql.Driver"}

# ==========================================
# 3. Spark 세션 초기화 (패키지 내재화)
# ==========================================
spark = SparkSession.builder \
    .appName(f"Daily_ETL_{REAL_TARGET_DATE}") \
    .config("spark.sql.session.timeZone", "UTC") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,org.postgresql:postgresql:42.6.0") \
    .getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", AWS_ACCESS_KEY) 
hadoop_conf.set("fs.s3a.secret.key", AWS_SECRET_KEY) 
hadoop_conf.set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# ==========================================
# 4. 데이터 로드 및 멱등성 보장 (Deduplication)
# ==========================================
# 🔸 수정 포인트: Airflow 날짜를 기반으로 S3 파티션 경로를 동적으로 생성합니다.
target_year = AIRFLOW_DATE[:4]
target_month = AIRFLOW_DATE[5:7]
target_day = AIRFLOW_DATE[8:10]

S3_PATH = f"s3a://otto-data-786802935144-ap-northeast-2-an/topics/otto-train-data/year={target_year}/month={target_month}/day={target_day}/"
print(f"📂 스파크가 탐색할 S3 파티션 경로: {S3_PATH}")

df_raw = spark.read.parquet(S3_PATH)

print(f"🎯 필터링 범위(ms): {target_ts_start} ~ {target_ts_end}")
daily_events = df_raw.filter((F.col("ts") >= target_ts_start) & (F.col("ts") <= target_ts_end))

# 시간 변환 및 date_id 추가
daily_events = daily_events.withColumn("ts_datetime", (F.col("ts") / 1000).cast(TimestampType())) \
                           .withColumn("date_id", F.lit(REAL_TARGET_DATE).cast("date"))

# [팩트 체크] 중복 제거 전 카운트
raw_count = daily_events.count()

# 핵심 로직: session_id, ts, aid, event_type이 모두 같으면 완벽한 중복으로 간주하고 날립니다.
daily_events = daily_events.dropDuplicates(['session_id', 'ts', 'aid', 'event_type'])

# [팩트 체크] 중복 제거 후 카운트
actual_count = daily_events.count()
dropped_count = raw_count - actual_count

print(f"📊 [DEBUG] 필터링된 원본 데이터: {raw_count}건")
print(f"🗑️ [DEBUG] 카프카 중복으로 제거된 데이터: {dropped_count}건")
print(f"✨ [DEBUG] DW에 적재될 최종 고유 데이터: {actual_count}건")

if actual_count == 0:
    print("⚠️ 경고: 필터링 결과 데이터가 0건입니다. OFFSET을 다시 확인하세요.")
    spark.stop()
    sys.exit(1)

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