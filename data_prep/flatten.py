import json
from pathlib import Path

# 현재 스크립트(flatten.py)의 위치를 기준으로 data 폴더의 절대 경로를 동적으로 찾습니다.
# 현재 위치: metacode_portfolio/data_prep
# 부모 위치: metacode_portfolio
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# 입력/출력 파일 경로 설정
# 처음 테스트 시에는 'test_data.jsonl'로 변경해서 돌려보세요.
INPUT_FILE = DATA_DIR / 'train.jsonl' 
OUTPUT_FILE = DATA_DIR / 'flat_events.jsonl'

print(f"📦 평탄화 작업 시작: {INPUT_FILE} -> {OUTPUT_FILE}")

# 파일이 존재하는지 사전 검증 (냉정한 에러 방지)
if not INPUT_FILE.exists():
    print(f"❌ 에러: 원본 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {INPUT_FILE}")
    exit(1)

with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for line in fin:
        try:
            data = json.loads(line)
            s_id = data["session"]
            for ev in data["events"]:
                # 핵심: ts(타임스탬프)를 맨 앞으로 빼서 나중에 리눅스 sort가 읽기 편하게 만듭니다.
                # 컬럼 구분자는 탭(\t)을 사용합니다.
                fout.write(f"{ev['ts']}\t{s_id}\t{ev['aid']}\t{ev['type']}\n")
        except Exception as e:
            print(f"⚠️ 데이터 파싱 에러 발생 (해당 라인 스킵): {e}")
            continue

print("✅ 평탄화 완료!")

