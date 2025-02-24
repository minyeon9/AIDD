from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

# 프롬프트를 받을 데이터 모델 정의
class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate/")
async def generate_response(request: PromptRequest):
    # Llama 모델 호출 (올라마 API 사용)
    response = ollama.chat(model="llama-3.1", messages=[{"role": "user", "content": request.prompt}])

    # 받은 응답 반환
    return {"response": response['text']}