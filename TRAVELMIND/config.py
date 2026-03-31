import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================
# API Keys
# ============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

# ============================
# Project Root
# ============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================
# OpenRouter Model Configuration
# ============================
LLM_MODEL = "gemini-2.5-flash"  # main LLM for TravelMind AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
# ============================
# Embeddings / RAG Configuration
# ============================
EMBED_MODEL = "all-MiniLM-L6-v2"  # for vector embeddings
CHROMA_PATH = os.path.join(PROJECT_ROOT, "vectorstore")      # folder to store vector DB
DATA_PATH = os.path.join(PROJECT_ROOT, "data")               # folder for raw documents

# ============================
# Application Parameters
# ============================
TOP_K_RESULTS = 2            # number of docs to retrieve for RAG
TOP_K_DESTINATIONS = 3       # number of destinations to suggest
MAX_TOKENS = 2048         # max tokens per API call