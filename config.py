import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model configurations
DEFAULT_LLM_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Agent configurations
MAX_ITERATIONS = 5
VERBOSE = True

# Path configurations
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)