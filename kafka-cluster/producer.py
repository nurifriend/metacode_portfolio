import json
import time
from pathlib import Path
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

# ── 1. 설정 및 경로 ──────────────────────────────────────────
TARGET_EPS = 10  # 초당 10건 전송
INTERVAL = 1.0 / TARGET_EPS

TOPIC_NAME = 'otto-train-data'
TEST_DATA_PATH = Path('test_data.jsonl')

# ── 2. Avro 스키마 (원본 데이터 구조에 맞춤) ──────────────────
VALUE_SCHEMA_STR = """
{
   "namespace": "otto.commerce",
   "name": "Event",
   "type": "record",
   "fields" : [
     {"name" : "session_id", "type" : "long"},
     {"name" : "aid", "type" : "long"},
     {"name" : "ts", "type" : "long"},
     {"name" : "event_type", "type" : "string"}
   ]
}
"""
value_schema = avro.loads(VALUE_SCHEMA_STR)

# ── 3. 콜백 함수 ────────────────────────────────────────────
def delivery_report(err, msg):
    if err is not None:
        print(f"❌ 전송 실패: {err}")

# ── 4. 메인 전송 로직 ───────────────────────────────────────
def main():
    # AvroProducer 초기화
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'schema.registry.url': 'http://localhost:8081',
        'client.id': 'avro-producer-eps-10'
    }
    producer = AvroProducer(conf, default_value_schema=value_schema)

    print(f"🚀 [Avro Mode] EPS {TARGET_EPS}로 원본 데이터 전송을 시작합니다...")
    
    total_events = 0

    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                session_data = json.loads(line)
                session_id = session_data["session"]

                # 한 세션 안의 여러 이벤트를 개별 레코드로 분리
                for raw_ev in session_data["events"]:
                    loop_start = time.time()
                    
                    record = {
                        "session_id": session_id,
                        "aid": raw_ev["aid"],
                        "ts": raw_ev["ts"],
                        "event_type": raw_ev["type"]
                    }

                    producer.produce(topic=TOPIC_NAME, value=record, callback=delivery_report)
                    producer.poll(0)
                    
                    total_events += 1
                    
                    if total_events % 10 == 0:
                        print(f"✅ {total_events:>4}개 이벤트 전송 중...")

                    # 쓰로틀링 유지
                    elapsed = time.time() - loop_start
                    if elapsed < INTERVAL:
                        time.sleep(INTERVAL - elapsed)

    except FileNotFoundError:
        print(f"❌ {TEST_DATA_PATH} 파일을 찾을 수 없습니다.")
    except KeyboardInterrupt:
        print("\n🛑 중단되었습니다.")

    producer.flush()
    print(f"🎉 총 {total_events}개의 원본 Avro 데이터 전송 완료!")

if __name__ == "__main__":
    main()