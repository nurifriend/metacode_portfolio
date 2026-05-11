# 1. 베이스 이미지: 공식 Airflow 이미지 사용
# (만약 기존 docker-compose.yaml에 적혀있던 버전이 있다면 그 버전으로 맞춰주세요)
FROM apache/airflow:2.7.3

# 2. ROOT 권한: 리눅스 시스템 패키지(Java) 설치를 위해 관리자 권한 획득
USER root

# Spark 구동의 핵심인 Java(JRE) 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       default-jre-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME 환경변수 등록
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# 3. AIRFLOW 계정: 다시 일반 유저로 돌아와서 파이썬 패키지 설치
USER airflow

# 팩트 체크: faker 패키지를 추가하여 이미지를 빌드합니다.
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    faker \
    apache-airflow-providers-apache-spark \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.8.txt"