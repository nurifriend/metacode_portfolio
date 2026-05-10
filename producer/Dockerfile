# 1. 가벼운 파이썬 베이스 이미지 사용
FROM python:3.12-slim

# 2. confluent-kafka 빌드에 필요한 OS 필수 패키지 설치
# (C 기반 라이브러리라 gcc와 librdkafka가 없으면 설치 중 에러가 날 수 있습니다)
RUN apt-get update && apt-get install -y \
    gcc \
    librdkafka-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Avro를 지원하는 confluent-kafka 라이브러리 설치
RUN pip install --no-cache-dir "confluent-kafka[avro]"

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 프로듀서 코드 복사
COPY replay_producer.py .

# 6. 컨테이너 실행 시 프로듀서 가동
CMD ["python", "replay_producer.py"]