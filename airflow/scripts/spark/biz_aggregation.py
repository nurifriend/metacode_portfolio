import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from argparse import ArgumentParser
from datetime import datetime, timedelta # ◀ 이 줄이 빠져 있었습니다. (FACT)

def main():
    parser = ArgumentParser()
    parser.add_argument("--date", help="target date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # 팩트 체크: Airflow에서 온 날짜를 1379일 전으로 동일하게 돌려줍니다.
    execution_date = datetime.strptime(args.date, "%Y-%m-%d")
    target_date = (execution_date - timedelta(days=1379)).strftime("%Y-%m-%d")

    spark = SparkSession.builder \
        .appName(f"BIZ_Aggregation_{target_date}") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .getOrCreate()

    # DB 접속 정보 (환경변수)
    db_url = f"jdbc:postgresql://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/otto_dw"
    db_properties = {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }

    # 1. 소스 데이터 로드
    # 팩트 체크: aid 컬럼명이 중복되므로 dim_product 로드 시 이름을 미리 변경해둡니다.
    fact_events = spark.read.jdbc(url=db_url, table="fact_events", properties=db_properties) \
        .filter(F.col("date_id") == target_date)
    
    dim_product = spark.read.jdbc(url=db_url, table="dim_product", properties=db_properties) \
        .withColumnRenamed("aid", "dim_aid")

    # --- [A] biz_daily_prdt 집계 ---
    # 팩트 체크: 확인된 컬럼명인 'category'와 'aid'를 사용합니다.
    print(f"🚀 {target_date} 제품별 성과 집계 시작...")
    biz_daily_prdt = fact_events.join(dim_product, fact_events.aid == dim_product.dim_aid, "left") \
        .groupBy("date_id", "category") \
        .agg(
            F.count(F.when(F.col("event_type") == "clicks", 1)).alias("click_cnt"),
            F.count(F.when(F.col("event_type") == "carts", 1)).alias("cart_cnt"),
            F.count(F.when(F.col("event_type") == "orders", 1)).alias("order_cnt"),
            F.sum(F.when(F.col("event_type") == "orders", F.col("price")).otherwise(0)).alias("daily_revenue")
        )

    # --- [B] biz_daily_funnel 집계 ---
    print(f"🚀 {target_date} 퍼널 지표 집계 시작...")
    biz_daily_funnel = fact_events.groupBy("date_id") \
        .agg(
            F.countDistinct("session_id").alias("total_sessions"),
            F.countDistinct(F.when(F.col("event_type") == "clicks", F.col("session_id"))).alias("click_sessions"),
            F.countDistinct(F.when(F.col("event_type") == "carts", F.col("session_id"))).alias("cart_sessions"),
            F.countDistinct(F.when(F.col("event_type") == "orders", F.col("session_id"))).alias("order_sessions")
        )

    # 2. 결과 적재
    biz_daily_prdt.write.jdbc(url=db_url, table="biz_daily_prdt", mode="append", properties=db_properties)
    biz_daily_funnel.write.jdbc(url=db_url, table="biz_daily_funnel", mode="append", properties=db_properties)

    print("✅ BIZ 테이블 적재 완료")
    spark.stop()

if __name__ == "__main__":
    main()