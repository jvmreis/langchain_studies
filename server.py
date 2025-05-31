from dotenv import load_dotenv
import os

from fastapi import FastAPI
from langserve import add_routes

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

# Load environment variables
load_dotenv()

print("🔧 Starting server setup...")

app = FastAPI()

# Load models
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.featherless.ai/v1",
    model="THUDM/GLM-4-32B-0414",
)
print("✅ LLM loaded")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("✅ Embedding model loaded")

try:
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name="pdf_file",
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        embedding=embeddings
    )
    print("✅ Vectorstore connected")
except Exception as e:
    print("❌ Failed to connect to vectorstore:", e)
    raise RuntimeError("❌ Collection 'pdf_file' not found. Run 'indexer.py' first to create it.") from e

retriever = vectorstore.as_retriever()
print("✅ Retriever ready")

custom_prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the following context extracted from technical documentation to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)
print("✅ QA chain ready")

# Util function to extract text from different input formats
def extract_query(input_data):
    if isinstance(input_data, list):
        return input_data[-1].content
    elif hasattr(input_data, 'content'):
        return input_data.content
    return input_data

# Final chain using LCEL and support for chat history
chain_final = (
    RunnableParallel({
        "input": itemgetter("input") | RunnableLambda(extract_query),
        "history": itemgetter("history")
    })
    | RunnableLambda(lambda x: {"query": x["input"]})
    | qa_chain
)

# Use LangChain's default ChatMessageHistory to fix stream_log compatibility
def get_session_history(session_id: str):
    return ChatMessageHistory()

runnable_with_history = RunnableWithMessageHistory(
    chain_final,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

try:
    add_routes(app, runnable_with_history, path="/chat")
    print("✅ Routes added successfully")
except Exception as e:
    print("❌ Failed to add routes:", e)
    raise

@app.get("/")
def health_check():
    return {"status": "ok"}
