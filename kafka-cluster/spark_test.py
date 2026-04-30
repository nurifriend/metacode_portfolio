from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, regexp_extract, length, desc, regexp_replace, floor, concat, lit
from datasets import load_dataset

# ==========================================
# 1. Spark 세션 초기화 및 설정
# ==========================================
spark = SparkSession.builder \
    .appName("NemotronPersonaAnalysis_Master") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ==========================================
# 2. 데이터 로드 (빠른 검증을 위한 1만 건 샘플링)
# ==========================================
print("🚀 [1/4] 데이터셋 로딩 중 (1만 건 샘플링)...")
raw_ds = load_dataset("nvidia/Nemotron-Personas-Korea", split='train[:10000]')
df = spark.createDataFrame(raw_ds.to_pandas())

# 공통 파생변수 생성 (연령대)
# 나이를 10 단위로 내림하여 'O0대' 형태로 변환 (예: 74 -> 70대)
df_base = df.withColumn("age_group", concat((floor(col("age") / 10) * 10).cast("string"), lit("대")))


# ==========================================
# 3. 데이터 파이프라인 및 집계 로직 실행
# ==========================================
print("⚙️ [2/4] 7가지 주제별 데이터 정제 및 분석 파이프라인 가동 중...")

# --- [주제 1] 행정구역 추출 및 집계 ---
korea_regions = r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도)"
region_counts = df_base.withColumn("region", regexp_extract(col("persona"), korea_regions, 1)) \
    .filter(length(col("region")) > 0) \
    .groupBy("region").count() \
    .orderBy(desc("count"))

# --- [주제 2] 전문성 키워드 추출 (숫자/불용어 완벽 제거) ---
clean_text = regexp_replace(col("professional_persona"), r"[^\w\s]", "")
words = explode(split(clean_text, r"\s+"))
word_counts = df_base.withColumn("word", words) \
    .filter(length(col("word")) >= 2) \
    .filter(~col("word").rlike(r"^[0-9]+$")) \
    .filter(~col("word").rlike(r"^[0-9]+[a-zA-Z가-힣]+$")) \
    .filter(~col("word").rlike(r"(습니다|합니다|입니다|씨는|에게|에서|부터|까지|위해|대한|통해|있는|하는|자신의)$")) \
    .groupBy("word").count() \
    .orderBy(desc("count"))

# --- [주제 3] 연령대별 주거형태 분석 ---
age_housing_counts = df_base.groupBy("age_group", "housing_type").count() \
    .orderBy("age_group", desc("count"))

# --- [주제 4] 취미/관심사 리스트 평탄화 분석 ---
clean_hobbies_str = regexp_replace(col("hobbies_and_interests_list"), r"\[|\]|'", "")
hobby_words = explode(split(clean_hobbies_str, r",\s*"))
hobby_counts = df_base.withColumn("hobby", hobby_words) \
    .filter(length(col("hobby")) > 1) \
    .groupBy("hobby").count() \
    .orderBy(desc("count"))

# --- [주제 5] 핵심 타겟: 1인 가구 거주 형태 ---
single_household_counts = df_base.filter(col("family_type") == "혼자 거주") \
    .groupBy("age_group", "housing_type").count() \
    .orderBy(desc("count"))

# --- [주제 6] 전공 분야와 직업의 상관관계 매핑 ---
major_job_counts = df_base.filter(col("bachelors_field") != "해당없음") \
    .groupBy("bachelors_field", "occupation").count() \
    .orderBy("bachelors_field", desc("count"))

# --- [주제 7] 지역별 핵심 경제활동(직업) 인구 파악 ---
regional_job_counts = df_base.filter(col("occupation") != "무직") \
    .groupBy("province", "occupation").count() \
    .orderBy("province", desc("count"))


# ==========================================
# 4. 결과 터미널 출력
# ==========================================
print("\n📊 [3/4] 분석 결과 요약 (Top 10)")
print("-" * 50)

print("\n[1. 지역별 분포]")
region_counts.show(5)

print("[2. 전문성 키워드]")
word_counts.show(5)

print("[3. 1인 가구 연령 및 주거형태]")
single_household_counts.show(5, truncate=False)

print("[4. 가장 인기 있는 취미/관심사]")
hobby_counts.show(5, truncate=False)

print("[5. 전공별 주요 직업]")
major_job_counts.show(5, truncate=False)


# ==========================================
# 5. 결과 파일 저장 (.csv & .parquet)
# ==========================================
print("💾 [4/4] 로컬 파일(CSV, Parquet)로 집계 결과 저장 중...")

# 출력물을 단일 파일(coalesce(1))로 깔끔하게 저장합니다.
region_counts.coalesce(1).write.mode("overwrite").csv("./output/01_region_distribution", header=True)
word_counts.coalesce(1).write.mode("overwrite").parquet("./output/02_professional_keywords")
age_housing_counts.coalesce(1).write.mode("overwrite").csv("./output/03_age_housing_counts", header=True)
hobby_counts.coalesce(1).write.mode("overwrite").csv("./output/04_hobbies_ranking", header=True)
major_job_counts.coalesce(1).write.mode("overwrite").csv("./output/05_major_job_mapping", header=True)

print("✅ 모든 데이터 파이프라인 작업 및 저장이 완벽하게 완료되었습니다!")
spark.stop()