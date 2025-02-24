from fastapi import FastAPI, File, UploadFile
import os
import PyPDF2
import pandas as pd
import docx
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

# Chroma 클라이언트 초기화
chroma_client = chromadb.HttpClient(host='localhost', port=8001)

# Embedding 모델 초기화
embeddings = OpenAIEmbeddings(openai_api_key="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")

def split_text_to_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def save_to_chroma(chunks: list, filename: str, collection_name: str = "uploaded_documents"):
    vector_db = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    vector_db.add_texts(chunks, metadatas=[{"filename": filename}] * len(chunks))

def extract_text_from_file(filepath: str):
    ext = filepath.split('.')[-1].lower()

    if ext == "pdf":
        with open(filepath, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text_data = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext in ["xlsx", "xls"]:
        df = pd.read_excel(filepath)
        text_data = df.to_string()

    elif ext == "docx":
        doc = docx.Document(filepath)
        text_data = "\n".join(para.text for para in doc.paragraphs if para.text)

    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")

    return text_data

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = ("pdf", "xlsx", "xls", "docx")
    ext = file.filename.split('.')[-1].lower()

    if ext not in allowed_extensions:
        return {"error": f"지원하지 않는 파일 형식입니다. 허용: {allowed_extensions}"}

    contents = await file.read()
    temp_filepath = f"temp_{file.filename}"

    with open(temp_filepath, "wb") as f:
        f.write(contents)

    try:
        # 텍스트 추출
        text_data = extract_text_from_file(temp_filepath)

        # 청크 분할
        chunks = split_text_to_chunks(text_data)

        # 청크를 벡터DB에 저장
        save_to_chroma(chunks, filename=file.filename)

        return {"message": f"{file.filename} 업로드 및 Chroma DB 저장 완료", "chunk_count": len(chunks)}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
