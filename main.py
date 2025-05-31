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
from langchain_core.documents import Document

import re
import fitz
from PIL import Image
import io
import base64
# -----------------------
# Configurações iniciais
# -----------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")  # Ainda usado para LLM se quiser
debug = os.getenv("DEBUG")

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
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Função: Divide o documento em partes menores (chunks)
def divide_texto(lista_documento_entrada):
    print(f">>> REALIZANDO A DIVISAO DO TEXTO ORIGINAL EM CHUNKS")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(lista_documento_entrada)
    for i, pedaco in enumerate(documents):
        print("--" * 30)
        print(f"Chunk: {i}")
        print(pedaco)
        print("--" * 30)
    return documents


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
    lista_documentos = PyPDFLoader('FAQ_BOOKING_COM.pdf').load()
    print("Texto lido e convertido em Document")
    print(lista_documentos)
    print("-----------------------------------")
    return lista_documentos

def identificar_imagem():
        """Identifica a página que tem uma imagem para chamar a função de transcrição depois."""
        paginas_com_imagem = []

        doc_aberto = fitz.open(caminho_documento)

        for page_num, page in enumerate(doc_aberto):
            page_text = page.get_text("text")
            # Identifica páginas que contêm figuras baseando-se na presença de legenda: "Figura X - ..."
            if re.search(r'Figura \d+', page_text):
                if page_num not in paginas_com_imagem:
                    paginas_com_imagem.append(page_num)
                    print(f"** Página {page_num+1} contém uma imagem!")

        return paginas_com_imagem

def gera_transcricao_imagem(pagina_com_imagem):
        """ Função que gera a transcrição da imagem usando modelo multimodal e retorna um Document com a descrição
        da imagem."""
        doc_aberto = fitz.open(caminho_documento)
        page = doc_aberto.load_page(pagina_com_imagem)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        # Se quiser ver a imagem:
        # img.show()

        # Criar uma instância do modelo Multimodal:
        llm_multimodal = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        prompt_de_transcricao = """Você é um especialista em criar transcrição fiel da imagem. Comente todos os detalhes. \
        Descreva valores e cores quando tiver. Inclua o que a imagem quer apresentar.
        Ignore o texto da página, foque na região que contem a imagem. Comentando todos os detalhes como se tivesse explicando \
        para uma pessoa cega. Indique também a numeração da imagem que está na legenda."""

        # Cria uma mensagem para ser enviada ao LLM. No caso de imagens é um pouco diferente!
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_de_transcricao},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"},
                },
            ],
        )
        response = llm_multimodal.invoke([message])

        return Document(page_content=response.content, metadata={"source": "./DENGUE.pdf", "page": pagina_com_imagem})
        
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
def menu():
    while True:
        print("\nOpções:")
        print("q -> Sair")
        print("1 -> Indexar informações no banco")
        print("2 -> Apenas conectar ao banco existente")
        opcao = input("Escolha uma opção: ").strip().lower()

        if opcao == 'q':
            print("Saindo do programa...")
            break

        elif opcao == '1':
            texto_completo_lido = ler_txt_e_retorna_texto_em_document()
            divide_texto_resultado = divide_texto(texto_completo_lido)

            doc_descr_imagem = []

            lista_pg_imagem = identificar_imagem()
            for pagina in lista_pg_imagem:
                doc_descr_imagem.append(gera_transcricao_imagem(pagina))

            todos_documentos = doc_descr_imagem + divide_texto_resultado

            cria_banco_vetorial_e_indexa_documentos(todos_documentos)
            print("Indexação concluída!")

        elif opcao == '2':
            print("Conectando ao banco vetorial existente...")
            db = conecta_banco_vetorial_pre_criado()
            print("Conexão estabelecida com sucesso!")

            class LineListOutputParser(BaseOutputParser[List[str]]):
                """Output parser for a list of lines."""
                def parse(self, text: str) -> List[str]:
                    lines = text.strip().split("\n")
                    return list(filter(None, lines))  # Remove empty lines

            output_parser = LineListOutputParser()

            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco \
            diferentes versões da pergunta do usuário para recuperar documentos relevantes de um vetor \
            banco de dados. Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar\
            o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
            Forneça essas perguntas alternativas separadas por novas linhas.
            Pergunta original: {question}"""
            )

            db_retriever = db.as_retriever()
            llm_chain = QUERY_PROMPT | llm | output_parser
            retriever = MultiQueryRetriever(retriever=db_retriever, llm_chain=llm_chain, parser_key="lines")

            query = "Quando eu chego na hospedagem preciso pagar algo?"
            pedacos_retornados = retriever.invoke(query)

            print(f"Total de pedaços retornados (documents): {len(pedacos_retornados)}\n")

            for i, pedaco in enumerate(pedacos_retornados):
                print(f"------ (documents) chunk {i} -------")
                print(pedaco.page_content)
                print("-------------------------------------")

        else:
            print("Opção inválida. Tente novamente.")


# Executa o menu
menu()
