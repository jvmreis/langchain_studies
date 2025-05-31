from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env vars
load_dotenv()

# Load and split PDF
loader = PyPDFLoader('pact-methodology-v3.0.pdf')

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(separators=[""], chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Index to Qdrant
QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    api_key=os.getenv("QDRANT_API_KEY"),
    url=os.getenv("QDRANT_URL"),
    prefer_grpc=True,
    collection_name="pdf_file"
)

print("✅ PDF indexed successfully.")
