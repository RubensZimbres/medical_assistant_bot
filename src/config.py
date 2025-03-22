import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true") == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MEDICAL_ASSISTANT_BOT")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-base")
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")

# Training Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "500"))

# Dataset Configuration
DATASET_PATH = DATA_DIR / "medical_qa.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "medical_qa_processed.csv"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Gemini Configuration
GEMINI_MODEL = "gemini-2.0-flash"
