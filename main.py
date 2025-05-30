from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()  # carrega as variáveis do .env

api_key = os.getenv("OPENAI_API_KEY")
debug = os.getenv("DEBUG")

print(f"API Key: {api_key}")
print(f"Debug mode: {debug}")

llm = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://api.featherless.ai/v1",
    model="THUDM/GLM-4-32B-0414",
)

messages = [
    SystemMessage(content="Você é um assistente útil que responde ao usuário com detalhes e exemplos.")

]
while True:
    entrada = input("Entrada Usuário (digite 'q' para parar.): ")
    if entrada.lower() == "q":
        break

    messages.append(HumanMessage(content=entrada))

    resultado = llm.invoke(messages)
    resposta = resultado.content
    messages.append(AIMessage(content=resposta))

    print(f"Resposta IA: {resposta}")

print("---- Histórico Completo ----")
print(messages)
