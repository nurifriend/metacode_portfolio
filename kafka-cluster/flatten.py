import json
from pathlib import Path

# 실제 파일명에 맞게 수정하세요. 
# 처음에는 3.9MB짜리 test_data.jsonl로 먼저 테스트해보는 것을 추천합니다.
INPUT_FILE = 'test_data.jsonl' 
OUTPUT_FILE = 'flat_events.jsonl'

print(f"📦 평탄화 작업 시작: {INPUT_FILE} -> {OUTPUT_FILE}")

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
            print(f"Error parsing line: {e}")
            continue

print("✅ 평탄화 완료!")


# flatten 이후 터미널에서 정렬
# sort -n -k1,1 flat_events.jsonl -o sorted_events.jsonl