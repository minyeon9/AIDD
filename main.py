from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import logging
from openai import OpenAI
from pymongo import MongoClient
from datetime import datetime

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 크로마DB 연결 설정
chroma_client = chromadb.HttpClient(host='localhost', port=8001)
embeddings = OpenAIEmbeddings(openai_api_key="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")

# OpenAI 설정
openai_client = OpenAI(api_key="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")
OPENAI_MODEL = "gpt-3.5-turbo"

# MongoDB 연결 설정
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["chat_db"]
collection = db["chat_logs"]

def query_chroma(query: str, collection_name: str = "uploaded_documents", k: int = 2):
    vector_db = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    results = vector_db.similarity_search_with_score(query, k=k)

    # 로그에 청크 개수 및 인덱스 출력
    logger.info(f"유사도 높은 청크 개수: {len(results)}")
    for idx, (doc, score) in enumerate(results, start=1):
        logger.info(f"[청크 {idx}] 유사도 점수: {score:.4f}\n내용:\n{doc.page_content}\n")

    return [doc.page_content for doc, _ in results]

def generate_answer_from_chunks(query: str, contexts: list):
    context_text = "\n\n---\n\n".join(contexts)

    messages = [
        {"role": "system", "content": "문서를 참고하여 사용자의 질문에 친절하고 정확하게 답변하는 AI 도우미입니다."},
        {"role": "user", "content": f"다음 문서를 참고하여 질문에 답변해줘.\n\n[문서]\n{context_text}\n\n[질문]\n{query}\n\n[답변]"}
    ]

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    return answer

@app.get("/api/chats/")
async def search(query: str):
    try:
        contexts = query_chroma(query)
        if not contexts:
            return {"message": "해당 질문에 맞는 문서를 찾지 못했습니다."}

        answer = generate_answer_from_chunks(query, contexts)

        # MongoDB에 질문과 답변 저장
        log_entry = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now()
        }
        collection.insert_one(log_entry)

        logger.info(f"MongoDB 저장 완료: {log_entry}")

        return {"answer": answer}

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
