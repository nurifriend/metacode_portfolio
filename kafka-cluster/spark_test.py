from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, regexp_extract, length, desc
from datasets import load_dataset

# 1. Spark 세션 시작
spark = SparkSession.builder \
    .appName("NemotronPersonaInsight") \
    .getOrCreate()

# 2. 데이터 로드 (실습 환경을 위해 10만 건만 샘플링)
print("🚀 데이터셋 로딩 중... (약 1M 행 중 10만 행 샘플링)")
raw_ds = load_dataset("nvidia/Nemotron-Personas-Korea", split='train')
df = spark.createDataFrame(raw_ds.to_pandas().head(100000))

# --- [과제 2번: 정규표현식으로 지역 정보 추출] ---
print("📍 1. 지역 정보 추출 및 집계 중...")
# persona 컬럼에서 '광주', '서울', '경기도' 등 광역 지자체 이름을 추출합니다.
df_with_region = df.withColumn("region", regexp_extract(col("persona"), r"([가-힣]+(?:시|도|광역시|특별자치시))", 1))

# 지역별 분포 집계 (정제: 지역명이 없는 경우는 제외)
region_counts = df_with_region.filter(length(col("region")) > 0) \
    .groupBy("region").count() \
    .orderBy(desc("count"))

# --- [과제 1번: 전문성 키워드 빈도 분석] ---
print("🔍 2. 전문성 키워드 분석 중 (Explode 활용)...")
# professional_persona의 문장을 공백으로 쪼개고, 각 단어를 행으로 펼칩니다(explode).
# '씨는', '하고', '있는' 같은 불필요한 단어를 걸러내기 위해 2글자 이상만 추출합니다.
word_counts = df.withColumn("word", explode(split(col("professional_persona"), " "))) \
    .filter(length(col("word")) >= 2) \
    .groupBy("word").count() \
    .orderBy(desc("count"))

# 3. 결과 출력 (상위 10개)
print("\n[결과 1: 지역별 분포]")
region_counts.show(10)

print("[결과 2: 주요 전문성 키워드]")
word_counts.show(10)

# 4. 결과 저장 (재확인 가능한 파일 형태)
# 과제 요건에 따라 로컬 파일(Parquet)로 저장합니다.
print("💾 결과를 저장하는 중...")
region_counts.write.mode("overwrite").csv("./output/region_distribution.csv", header=True)
word_counts.write.mode("overwrite").parquet("./output/professional_keywords.parquet")

print("✅ 모든 작업이 완료되었습니다!")
spark.stop()