from dotenv import load_dotenv
import os

load_dotenv()  # carrega as variáveis do .env

api_key = os.getenv("API_KEY")
debug = os.getenv("DEBUG")

print(f"API Key: {api_key}")
print(f"Debug mode: {debug}")
