from faker import Faker
from kafka import KafkaProducer
import json
import time

# 1. Faker와 카프카 프로듀서 설정
fake = Faker('ko-KR') # 한글 데이터 설정
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8') # 딕셔너리를 JSON 문자로 변환
)

topic_name = 'otto-events' # 아까 만든 토픽 이름

print(f"🚀 {topic_name} 토픽으로 테스트 데이터를 전송합니다... (중단: Ctrl+C)")

try:
    while True:
        # 점프 투 파이썬 예제 항목들을 조합하여 데이터 생성
        user_data = {
            "name": fake.name(),
            "address": fake.address(),
            "job": fake.job(),
            "email": fake.email(),
            "ip": fake.ipv4_private(),
            "ts": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 카프카로 전송
        producer.send(topic_name, value=user_data)
        
        print(f"✅ 전송 성공: {user_data['name']} ({user_data['job']})")
        
        # 너무 빠르면 보기 힘드니 1초에 한 번씩 생성
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 중단되었습니다.")
finally:
    producer.flush() # 남은 데이터 모두 전송 후 종료