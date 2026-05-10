import os
import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from faker import Faker

def generate_realistic_attributes(aid, percentile):
    """
    준규님이 작성한 Faker 기반 상품 속성 생성 로직 (1:1 보장)
    """
    # [팩트 체크] 파티션 내부에서 실행되므로 시드 고정 필수
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
    product_name = f"[{brand}] {random.choice(items[category])} (SKU-{aid})"
    
    return f"{category}|{product_name}|{price}"

def main():
    # 1. Spark 세션 초기화
    spark = SparkSession.builder \
        .appName("Update_Dim_Product_Master") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        .getOrCreate()

    # 2. DB 접속 정보 (환경변수 주입)
    DB_URL = f"jdbc:postgresql://{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/otto_dw"
    DB_PROPERTIES = {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }

    print("🔍 신규 상품 분석 시작...")

    # 3. 데이터 로드 및 신규 aid 추출
    # [팩트 체크] fact_events에 있지만 dim_product에는 없는 aid만 찾아냅니다 (Left Anti Join)
    df_fact_aids = spark.read.jdbc(url=DB_URL, table="fact_events", properties=DB_PROPERTIES) \
                        .select("aid").distinct()
    
    # 만약 dim_product 테이블이 아예 없으면 에러가 날 수 있으므로 예외 처리가 이성적입니다.
    try:
        df_dim_existing = spark.read.jdbc(url=DB_URL, table="dim_product", properties=DB_PROPERTIES) \
                               .select("aid")
        df_new_aids = df_fact_aids.join(df_dim_existing, on="aid", how="left_anti")
    except:
        print("⚠️ dim_product 테이블이 존재하지 않습니다. 전체 생성을 시도합니다.")
        df_new_aids = df_fact_aids

    new_count = df_new_aids.count()

    # 4. 신규 상품이 있을 경우에만 메타데이터 생성 및 적재
    if new_count > 0:
        print(f"✨ 신규 상품 {new_count}건 발견! 메타데이터 생성 중...")
        
        # UDF 등록
        generate_udf = F.udf(generate_realistic_attributes, StringType())

        # 데이터 생성 (percentile은 임의 부여)
        df_new_dim = df_new_aids.withColumn("percentile", F.rand()) \
                                .withColumn("fake_data", generate_udf(F.col("aid"), F.col("percentile")))

        # 컬럼 분리 및 최종 데이터셋 구성
        df_final = df_new_dim.select(
            F.col("aid"),
            F.split(F.col("fake_data"), "\|")[0].alias("category"),
            F.split(F.col("fake_data"), "\|")[1].alias("product_name"),
            F.split(F.col("fake_data"), "\|")[2].cast(IntegerType()).alias("price")
        )

        # 신규 데이터만 'append' 모드로 적재
        df_final.write.jdbc(url=DB_URL, table="dim_product", mode="append", properties=DB_PROPERTIES)
        print(f"✅ {new_count}건의 신규 상품이 dim_product에 성공적으로 추가되었습니다.")
    else:
        print("☕ 모든 상품 정보가 이미 존재합니다. 작업을 종료합니다.")

    spark.stop()

if __name__ == "__main__":
    main()