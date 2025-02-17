import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG конфигурация
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODEL_NAME = 'gpt-4o-mini' 