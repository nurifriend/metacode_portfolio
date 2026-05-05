import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType, TimestampType

# ==========================================
# 환경 변수 로드
# ==========================================
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# ==========================================
# 1. Spark 세션 생성
# ==========================================
spark = SparkSession.builder \
    .appName("S3_to_Postgres_Pipeline") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0,org.apache.hadoop:hadoop-aws:3.3.4") \
    .getOrCreate()

# ==========================================
# 2. S3 인증 정보 설정
# ==========================================
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", AWS_ACCESS_KEY) 
hadoop_conf.set("fs.s3a.secret.key", AWS_SECRET_KEY) 
hadoop_conf.set("fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# ==========================================
# 3. 데이터베이스 및 S3 경로 설정
# ==========================================
DB_URL = "jdbc:postgresql://localhost:5432/otto_dw"
DB_PROPERTIES = {"user": DB_USER, "password": DB_PASSWORD, "driver": "org.postgresql.Driver"}
S3_PATH = "s3a://otto-data-786802935144-ap-northeast-2-an/topics/otto-train-data/partition=0/"

# ==========================================
# 4. Phase 1: fact_events 가공 및 적재
# ==========================================
df_raw = spark.read.parquet(S3_PATH)
fact_events = df_raw.withColumn("ts_datetime", (F.col("ts") / 1000).cast(TimestampType())) \
                    .withColumn("date_id", F.to_date("ts_datetime"))

fact_events.write.jdbc(url=DB_URL, table="fact_events", mode="overwrite", properties=DB_PROPERTIES)
print("✅ fact_events 적재 완료")

# ==========================================
# 5. Phase 2: dim_product 가공 (현실적인 상품명 & 1:1 보장)
# ==========================================
aid_freq = fact_events.groupBy("aid").count()
windowSpec = Window.orderBy(F.desc("count"))
aid_ranked = aid_freq.withColumn("percentile", F.percent_rank().over(windowSpec))

def generate_realistic_attributes(aid, percentile):
    import random
    from faker import Faker
    
    random.seed(aid)
    fake = Faker('ko_KR')
    fake.seed_instance(aid)
    
    items = {
        "신선식품": ["당도선별 제주 감귤", "무항생제 한우 등심 세트", "유기농 양상추", "프리미엄 샤인머스캣"],
        "가공식품": ["단백질 닭가슴살", "저염 스팸 세트", "직화 라면", "통밀 파스타면"],
        "생필품": ["3겹 천연펄프 화장지", "고농축 액체세제", "저자극 칫솔 세트", "쑥 샴푸"],
        "의류": ["헤비 코튼 맨투맨", "스트레치 데님", "캐시미어 니트", "방수 바람막이"],
        "뷰티": ["히알루론산 크림", "비타민 세럼", "무기자차 선크림", "인텐시브 립밤"],
        "소형가전": ["스마트 무선 청소기", "핸디 다리미", "2구 토스터기", "초고속 블렌더"],
        "명품": ["카프스킨 토트백", "오토매틱 워치", "시그니처 스카프", "가죽 반지갑"],
        "대형가전": ["인버터 세탁기", "4K 65인치 TV", "노크온 냉장고", "공기청정 시스템"],
        "가구": ["메시 사무용 의자", "원목 식탁 세트", "패브릭 소파", "침대 프레임"]
    }

    if percentile <= 0.10:
        category = random.choice(["신선식품", "가공식품", "생필품"])
        price = random.randint(1, 8) * 1000
    elif percentile <= 0.70:
        category = random.choice(["의류", "뷰티", "소형가전"])
        price = random.randint(3, 30) * 10000
    else:
        category = random.choice(["명품", "대형가전", "가구"])
        price = random.randint(50, 450) * 10000

    brand = fake.company() if random.random() > 0.3 else "국산"
    
    # [핵심] 상품명 뒤에 고유 번호(aid)를 추가하여 1:1 관계를 물리적으로 강제함
    # 예: [삼성전자] 스마트 무선 청소기 (SKU-12345)
    product_name = f"[{brand}] {random.choice(items[category])} (SKU-{aid})"
    
    return f"{category}|{product_name}|{price}"

# UDF 등록 시 aid 파라미터 추가
generate_udf = F.udf(generate_realistic_attributes, StringType())

# aid를 UDF에 전달하여 시드로 활용
dim_product_raw = aid_ranked.withColumn("fake_data", generate_udf(F.col("aid"), F.col("percentile")))

dim_product = dim_product_raw.select(
    F.col("aid"),
    F.split(F.col("fake_data"), "\|")[0].alias("category"),
    F.split(F.col("fake_data"), "\|")[1].alias("product_name"),
    F.split(F.col("fake_data"), "\|")[2].cast(IntegerType()).alias("price")
)

dim_product.write.jdbc(url=DB_URL, table="dim_product", mode="overwrite", properties=DB_PROPERTIES)
print("✅ dim_product 적재 완료 (1:1 관계 보장)")

spark.stop()