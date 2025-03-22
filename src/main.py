"""
Main application entry point for the Medical Assistant Bot.
"""
import os
import argparse
import logging
from dotenv import load_dotenv
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models.deepseek_model import DeepseekModel
from .data.data_processor import DataProcessor
from .data.data_loader import DataLoader
from src.config import PROCESSED_DATASET_PATH, MODEL_DIR

from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define API models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# Initialize FastAPI app
app = FastAPI(
    title="Medical Assistant Bot",
    description="A medical question-answering system",
    version="1.0.0"
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model
    try:
        model_path = Path(MODEL_DIR) / "medical_assistant_model"
        model = DeepseekModel()

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model.load(str(model_path))
        else:
            logger.warning(f"Model not found at {model_path}. Using base model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.post("/api/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """
    API endpoint to answer medical questions.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        answer = model.predict(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )

def run_cli():
    """Run the command-line interface."""
    try:
        # Initialize the model
        model_path = Path(MODEL_DIR) / "medical_assistant_model"
        model = DeepseekModel()

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model.load(str(model_path))
        else:
            logger.warning(f"Model not found at {model_path}. Using base model.")

        print("Medical Assistant Bot")
        print("Type 'exit' to quit")

        while True:
            question = input("\nYour question: ")

            if question.lower() in ["exit", "quit", "q"]:
                break

            answer = model.predict(question)
            print(f"\nAnswer: {answer}")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error in CLI: {e}")
        print(f"An error occurred: {e}")

def train_model(dataset_path):
    """Train the model on the specified dataset."""
    try:
        # Load and process the dataset
        processor = DataProcessor()
        df = processor.load_and_process(dataset_path)

        # Split the dataset
        train_df, val_df, test_df = processor.split_dataset(df)

        # Convert to HuggingFace datasets
        data_loader = DataLoader()
        dataset_dict = data_loader.convert_to_huggingface_datasets(train_df, val_df, test_df)

        # Initialize and train the model
        model = DeepseekModel()
        model.train(dataset_dict['train'], dataset_dict['validation'])

        # Save the model
        model_path = Path(MODEL_DIR) / "medical_assistant_model"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

        return True
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Medical Assistant Bot")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # CLI mode
    cli_parser = subparsers.add_parser("cli", help="Run in CLI mode")

    # API mode
    api_parser = subparsers.add_parser("api", help="Run as API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=8080, help="Port to bind")

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")

    args = parser.parse_args()

    if args.command == "cli":
        run_cli()
    elif args.command == "api":
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
    elif args.command == "train":
        success = train_model(args.dataset)
        if success:
            print("Model training completed successfully.")
        else:
            print("Model training failed.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
