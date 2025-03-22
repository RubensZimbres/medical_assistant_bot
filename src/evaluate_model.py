"""
Script to evaluate the trained Medical Assistant Bot model.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.models.deepseek_model import DeepseekModel
from src.data.data_loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.config import MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Run model evaluation."""
    try:
        # Load the model
        model_path = Path(MODEL_DIR) / "medical_assistant_model"
        model = DeepseekModel()

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model.load(str(model_path))
        else:
            logger.error(f"Model not found at {model_path}")
            return False

        # Load test data
        # You can use a subset of your original dataset or a separate test set
        data_loader = DataLoader()
        test_df = data_loader.load_csv("mle_screening_dataset.csv")

        # Use a small subset for testing if the dataset is large
        test_df = test_df.sample(min(100, len(test_df)))

        # Initialize the evaluator
        # Make sure LANGCHAIN_API_KEY is set in your .env file
        evaluator = ModelEvaluator()

        # Create a dataset name
        dataset_name = "medical_qa_evaluation"

        # Create the evaluation dataset
        dataset_id = evaluator.create_evaluation_dataset(
            test_df=test_df,
            dataset_name=dataset_name
        )

        # Create a model callable for evaluation
        def model_callable(example):
            question = example["question"]
            return {"generations": [[{"text": model.predict(question)}]]}

        # Run the evaluation
        results = evaluator.evaluate_model(
            model_callable=model_callable,
            dataset_name=dataset_name,
            evaluation_project_name="medical_assistant_eval"
        )

        # Generate a report
        report = evaluator.generate_evaluation_report(results)

        # Print the results
        print("\n===== Evaluation Results =====")
        for metric, value in report.items():
            print(f"{metric}: {value}")

        return True

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Model evaluation completed successfully.")
    else:
        print("Model evaluation failed.")