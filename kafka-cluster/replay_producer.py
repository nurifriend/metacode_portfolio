import time
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer

# ── 1. 설정 ────────────────────────────────────────────────
TOPIC_NAME = 'otto-train-data'
# 반드시 'sorted_events.jsonl'을 사용하세요!
INPUT_FILE = 'sorted_events.jsonl' 
SPEED_UP_FACTOR = 1.0  # 1000배속 (상황에 따라 조절)

# ── 2. Avro 스키마 (기존과 동일) ──────────────────────────────
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

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ 전송 실패: {err}")

# ── 3. 메인 로직 ───────────────────────────────────────────
def main():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'schema.registry.url': 'http://localhost:8081',
        'client.id': 'avro-producer-final'
    }
    producer = AvroProducer(conf, default_value_schema=value_schema)

    first_event_ts = None
    start_real_time = None
    total_events = 0

    print(f"🚀 리플레이 시작: {INPUT_FILE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # 탭(\t)으로 분리된 데이터를 읽습니다. (중요!)
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    continue
                
                ts = int(parts[0])
                s_id = int(parts[1])
                aid = int(parts[2])
                ev_type = parts[3]

                # 기준 시간 설정
                if first_event_ts is None:
                    first_event_ts = ts / 1000.0
                    start_real_time = time.time()

                # 시간 동기화 (배속 적용)
                current_event_ts = ts / 1000.0
                relative_event_time = (current_event_ts - first_event_ts) / SPEED_UP_FACTOR
                
                while (time.time() - start_real_time) < relative_event_time:
                    time.sleep(0.001)

                # Avro 레코드 생성 및 전송
                record = {
                    "session_id": s_id,
                    "aid": aid,
                    "ts": ts,
                    "event_type": ev_type
                }
                producer.produce(topic=TOPIC_NAME, value=record, callback=delivery_report)
                producer.poll(0)
                
                total_events += 1
                if total_events % 100 == 0:
                    print(f"✅ {total_events}개 전송 중... (ts: {ts})")

    except KeyboardInterrupt:
        print("\n🛑 중단되었습니다.")

    producer.flush()
    print(f"🎉 총 {total_events}개 데이터 전송 완료!")

if __name__ == "__main__":
    main()