#!/bin/bash
# 에러 발생 시 즉시 실행을 중단하는 안전장치
set -e 

echo "▶ 1단계: Kaggle 데이터 다운로드 및 압축 해제"
if [ ! -f "/app/data/train.jsonl" ]; then
    echo "데이터가 없습니다. Kaggle에서 다운로드를 시작합니다..."
    kaggle competitions download -c otto-recommender-system -p /app/data
    
    echo "압축 해제 중..."
    unzip -q /app/data/*.zip -d /app/data/
    
    # 디스크 용량 절약을 위해 원본 zip 파일 삭제
    rm /app/data/*.zip 
else
    echo "✅ [SKIP] /app/data/train.jsonl 이미 존재. 1단계를 건너뜁니다."
fi


echo "▶ 2단계: 평탄화 작업 (Python)"
if [ ! -f "/app/data/flat_events.jsonl" ]; then
    # 1단계에서 train.jsonl이 정상적으로 확보되었는지 2차 검증
    if [ ! -f "/app/data/train.jsonl" ]; then
        echo "❌ 에러: 원본 파일(train.jsonl)이 없어 평탄화를 진행할 수 없습니다."
        exit 1
    fi
    echo "평탄화를 시작합니다..."
    python flatten.py
else
    echo "✅ [SKIP] /app/data/flat_events.jsonl 이미 존재. 2단계를 건너뜁니다."
fi


echo "▶ 3단계: 리눅스 고속 정렬 (Sort)"
if [ ! -f "/app/data/sorted_events.jsonl" ]; then
    # 2단계의 flat_events.jsonl이 정상적으로 존재하는지 검증
    if [ ! -f "/app/data/flat_events.jsonl" ]; then
        echo "❌ 에러: 평탄화된 파일(flat_events.jsonl)이 없어 정렬을 진행할 수 없습니다."
        exit 1
    fi
    echo "정렬을 시작합니다. (메모리 제한: 8G)"
    sort -n -k1,1 -S 8G /app/data/flat_events.jsonl -o /app/data/sorted_events.jsonl
else
    echo "✅ [SKIP] /app/data/sorted_events.jsonl 이미 존재. 3단계를 건너뜁니다."
fi

echo "✅ 모든 데이터 준비 과정이 완벽하게 종료되었습니다."