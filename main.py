from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

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
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    (
        "human",
        "I love programming."
    ),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
