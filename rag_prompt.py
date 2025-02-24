import redis
from fastapi import HTTPException

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
from rag import chain

# Redis 연결
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

question = "24년도 4분기 카본케미칼 매출이 전분기 대비 몇프로 감소했어?"
response = chain.invoke(question)
print(response)

# Redis에 질문과 응답 저장
try:
    redis_client.set(question, response)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Redis에 저장 실패: {str(e)}")
print({"question": question, "response": response})

