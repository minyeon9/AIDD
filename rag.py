from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("D:/documents/24Q4_IR PPT.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings(openai_api_key="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")

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
llm = ChatOpenAI(
    openai_api_key="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",  # 여기에 OpenAI API 키를 명시합니다.
    model_name="gpt-4o",             # 모델 이름 (예: "gpt-4" 또는 "gpt-4o")
    temperature=0
)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)