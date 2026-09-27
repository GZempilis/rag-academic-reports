import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT=Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "raw_data"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
TEST_PATH = PROJECT_ROOT / "test"


HF_TOKEN=os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("There is not Huggingface token")

if not RAW_DATA_PATH.exists() or not any(RAW_DATA_PATH.glob("*.pdf")):
    raise FileNotFoundError(f"No pdfs found in {RAW_DATA_PATH}.")


CHROMA_DB_PATH.mkdir(exist_ok=True)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

TOP_K = 4 ## chunks (tune if needed) 
COLLECTION_NAME = "rag_docs"

