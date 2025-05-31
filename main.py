from dotenv import load_dotenv
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever


# -----------------------
# Configurações iniciais
# -----------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY");  # Ainda usado para LLM se quiser
debug = os.getenv("DEBUG");

print(f"API Key: {api_key}")
print(f"Debug mode: {debug}")

# -------------------------------
# LLM Featherless (apenas para geração de texto, NÃO para embeddings)
# -------------------------------
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.featherless.ai/v1",
    model="THUDM/GLM-4-32B-0414",
)

# -------------------------------
# Embeddings locais com HuggingFace
# -------------------------------
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2");


# Função: Divide o documento em partes menores (chunks)
def divide_texto(lista_documento_entrada):
    print(f">>> REALIZANDO A DIVISAO DO TEXTO ORIGINAL EM CHUNKS")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0);
    documents = text_splitter.split_documents(lista_documento_entrada);
    for i, pedaco in enumerate(documents):
        print("--" * 30)
        print(f"Chunk: {i}")
        print(pedaco)
        print("--" * 30)
    return documents;


# Cria o banco de dados vetorial, gerando os embeddings dos documentos
def cria_banco_vetorial_e_indexa_documentos(documentos):
    print(f">>> REALIZANDO INDEXAÇÃO DOS CHUNKS NO BANCO VETORIAL")
    QdrantVectorStore.from_documents(
        documents=documentos,
        embedding=embeddings_model,
        api_key=os.environ.get("QDRANT_API_KEY"),
        url=os.environ.get("QDRANT_URL"),
        prefer_grpc=True,
        collection_name="pdf_file"
    )


# Função para carregar um PDF e retornar como lista de documentos
def ler_txt_e_retorna_texto_em_document():
    print(f">>> REALIZANDO A LEITURA DO PDF EXEMPLO")
    lista_documentos = PyPDFLoader('pact-methodology-v3.0.pdf').load()
    print("Texto lido e convertido em Document")
    print(lista_documentos)
    print("-----------------------------------")
    return lista_documentos;


# Conecta-se ao banco vetorial já existente
def conecta_banco_vetorial_pre_criado():
    server = QdrantVectorStore.from_existing_collection(
        collection_name="pdf_file",
        url=os.environ.get("QDRANT_URL"),
        embedding=embeddings_model,
        api_key=os.environ.get("QDRANT_API_KEY")
    )
    return server


# Menu de opções
# ... [imports e configurações anteriores] ...

# Menu de opções
def menu():
    while True:
        print("\nOpções:")
        print("q -> Sair")
        print("1 -> Indexar informações no banco")
        print("2 -> Apenas conectar ao banco existente")
        opcao = input("Escolha uma opção: ").strip().lower();

        if opcao == 'q':
            print("Saindo do programa...")
            break

        elif opcao == '1':
            texto_completo_lido = ler_txt_e_retorna_texto_em_document();
            divide_texto_resultado = divide_texto(texto_completo_lido);
            cria_banco_vetorial_e_indexa_documentos(divide_texto_resultado);
            print("Indexação concluída!")

        elif opcao == '2':
            print("Conectando ao banco vetorial existente...")
            db = conecta_banco_vetorial_pre_criado();
            print("Conexão estabelecida com sucesso!")

            # Cria o banco vetorial como recuperador:
            db_retriever = db.as_retriever();

            # A chain intermediária por padrão sem personalização vai criar 3 frases semelhantes à entrada passada antes de
            # fazer o retrivever:
            retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db_retriever, llm=llm);

            # Exemplo de consulta
            query = "what is the topic 2.1";
            pedacos_retornados = retriever_from_llm.invoke(query);

            print(f"Total de pedaços retornados (documents): {len(pedacos_retornados)}\n")

            for i, pedaco in enumerate(pedacos_retornados):
                print(f"------ (documents) chunk {i} -------")
                print(pedaco.page_content)
                print("-------------------------------------")

        else:
            print("Opção inválida. Tente novamente.")



# Executa o menu
menu()
