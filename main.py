from dotenv import load_dotenv
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# -----------------------
# Initial configuration
# -----------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
debug = os.getenv("DEBUG")

# print(f"API Key: {api_key}")
# print(f"Debug mode: {debug}")

# -------------------------------
# Featherless LLM (used for generation, NOT for embeddings)
# -------------------------------
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.featherless.ai/v1",
    model="THUDM/GLM-4-32B-0414",
)

# -------------------------------
# Local embeddings using HuggingFace
# -------------------------------
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Splits the document into chunks
def split_document_into_chunks(doc_list):
    # print(">>> Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(separators=[""], chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(doc_list)

    # for i, chunk in enumerate(chunks):
    #     print("--" * 30)
    #     print(f"Chunk: {i}")
    #     print(chunk)
    #     print("--" * 30)

    return chunks

# Creates vector DB and indexes document chunks
def create_and_index_vector_store(docs):
    # print(">>> Indexing chunks in vector database...")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings_model,
        api_key=os.environ.get("QDRANT_API_KEY"),
        url=os.environ.get("QDRANT_URL"),
        prefer_grpc=True,
        collection_name="pdf_file"
    )

# Loads a sample PDF and returns its content as documents
def load_pdf_as_documents():
    # print(">>> Loading sample PDF file...")
    docs = PyPDFLoader('pact-methodology-v3.0.pdf').load()
    # print("PDF content loaded and converted to Document format.")
    return docs

# Connects to an existing vector database
def connect_to_existing_vector_store():
    return QdrantVectorStore.from_existing_collection(
        collection_name="pdf_file",
        url=os.environ.get("QDRANT_URL"),
        embedding=embeddings_model,
        api_key=os.environ.get("QDRANT_API_KEY")
    )

# Main menu and state machine
def menu():
    while True:
        print("\nOptions:")
        print("q -> Quit")
        print("1 -> Index documents into the vector store")
        print("2 -> Connect to existing vector store and ask a question")
        choice = input("Choose an option: ").strip().lower()

        if choice == 'q':
            print("Exiting program...")
            break

        elif choice == '1':
            full_doc = load_pdf_as_documents()
            split_chunks = split_document_into_chunks(full_doc)
            create_and_index_vector_store(split_chunks)
            print("Indexing completed.")

        elif choice == '2':
            # print("Connecting to existing vector store...")
            db = connect_to_existing_vector_store()
            # print("Successfully connected!")

            class LineListOutputParser(BaseOutputParser[List[str]]):
                def parse(self, text: str) -> List[str]:
                    lines = text.strip().split("\n")
                    return list(filter(None, lines))

            output_parser = LineListOutputParser()

            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""Generate five different reformulations of the following user question \
to retrieve relevant documents from a vector database. Use synonyms, paraphrasing, and different perspectives.

Question: {question}"""
            )

            db_retriever = db.as_retriever()
            llm_chain = QUERY_PROMPT | llm | output_parser
            retriever = MultiQueryRetriever(retriever=db_retriever, llm_chain=llm_chain, parser_key="lines")

            query = "What are the existing standards protocols?"
            retrieved_docs = retriever.invoke(query)

            # print(f"Total retrieved chunks: {len(retrieved_docs)}")

            # Generate final answer
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = f"""Based on the following extracted content from a technical document, answer the question clearly and concisely:

Question: {query}

Content:
{context}

Answer:"""

            final_answer = llm.invoke(final_prompt)

            print("\n=== FINAL ANSWER ===")
            print(final_answer.content)

        else:
            print("Invalid option. Please try again.")

# Run the menu
menu()
