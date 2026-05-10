FROM python:3.12-slim

# unzip(캐글 압축해제용), curl, coreutils(리눅스 정렬용) 필수 설치
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    coreutils \
    && apt-get clean

# Kaggle API 및 데이터 처리 라이브러리 설치
RUN pip install --no-cache-dir kaggle pandas datasets pyarrow

WORKDIR /app
COPY . .

# 쉘 스크립트에 실행 권한 부여
RUN chmod +x run_prep.sh

# 컨테이너가 켜지면 무조건 이 스크립트를 실행해라
CMD ["./run_prep.sh"]