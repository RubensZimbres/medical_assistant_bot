import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import the project modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import required modules
from src.models.deepseek_model import DeepseekModel
from src.config import MODEL_DIR

def run_inference():
    """Run inference with the trained medical model."""
    try:
        # Initialize the model
        print("Initializing the DeepseekModel...")
        model = DeepseekModel()

        # Load the trained model
        model_path = Path(MODEL_DIR) / "medical_assistant_model"
        if not model_path.exists():
            print(f"Error: Trained model not found at {model_path}")
            return

        print(f"Loading trained model from {model_path}...")
        model.load(str(model_path))

        # Questions for inference
        questions = [
            "Tell me in one phrase: What are effective ways to prevent alcohol abuse?",
            "Tell me in one phrase: How can high blood pressure be prevented and managed?",
            "Tell me in one phrase: What structures in the eye are affected by glaucoma?"
        ]

        # Run inference for each question
        print("\n=== Running Inference ===")
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 50)

            # Generate response
            response = model.predict(question)

            print(f"Response {i}:")
            print(response)
            print("=" * 80)

        print("\nInference completed successfully.")

    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    run_inference()