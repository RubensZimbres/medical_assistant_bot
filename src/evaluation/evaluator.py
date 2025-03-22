import os
import logging
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.smith import RunEvalConfig
from langchain.smith import arun_on_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluator using LangChain and OpenAI models."""

    def __init__(self):
        """Initialize the evaluator with Gemini 2.0 Flash."""
        try:
            # Import Google's Generative AI library
            import google.generativeai as genai

            # Configure the Gemini API
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

            # Initialize Gemini 2.0 Flash model
            self.eval_llm = genai.GenerativeModel('gemini-2.0-flash')

            # Create LangChain wrapper for Gemini
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.eval_llm_chain = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )

            logger.info("Successfully initialized Gemini 2.0 Flash model for evaluation")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise

    def create_evaluation_dataset(self, test_df, dataset_name):
        """Create a dataset for evaluation."""
        try:
            # Add timestamp to make dataset name unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_dataset_name = f"{dataset_name}_{timestamp}"

            logger.info(f"Creating evaluation dataset: {unique_dataset_name}")

            # Save the dataset locally for evaluation
            dataset_path = Path(f"evaluation_datasets/{unique_dataset_name}")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare the dataset for evaluation
            eval_dataset = []
            for _, row in test_df.iterrows():
                eval_dataset.append({
                    "question": row["question"],
                    "reference": row.get("reference_answer", ""),  # Use reference answer if available
                    "id": str(uuid.uuid4())
                })

            # Save as a dataframe
            eval_df = pd.DataFrame(eval_dataset)
            eval_df.to_csv(dataset_path.with_suffix('.csv'), index=False)

            logger.info(f"Created evaluation dataset with {len(eval_df)} examples")
            return unique_dataset_name

        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {e}")
            raise

    def evaluate_model(self, model_callable, dataset_name, evaluation_project_name):
        """Evaluate the model using direct evaluation with OpenAI."""
        try:
            logger.info(f"Starting evaluation of model on dataset: {dataset_name}")

            # Find the dataset file
            dataset_files = list(Path("evaluation_datasets").glob(f"{dataset_name}*.csv"))
            if not dataset_files:
                raise FileNotFoundError(f"No dataset found with name: {dataset_name}")

            # Use the most recent dataset file
            dataset_file = sorted(dataset_files)[-1]

            # Load the dataset
            eval_df = pd.read_csv(dataset_file)

            # Define evaluation criteria
            evaluation_criteria = {
                "relevance": "The response directly addresses the medical question posed.",
                "completeness": "The response provides a thorough answer that covers all aspects of the question.",
                "factuality": "The response contains accurate medical information without factual errors.",
                "clarity": "The response is clear, well-structured, and easy to understand."
            }

            # Create evaluation chains with our specified LLM
            evaluators = {}
            for criterion_name, criterion_description in evaluation_criteria.items():
                evaluators[criterion_name] = LabeledCriteriaEvalChain.from_llm(
                    llm=self.eval_llm_chain,  # Use the LangChain wrapper for Gemini
                    criteria={criterion_name: criterion_description},
                    normalize_by="reference"
                )

            # Run evaluation for each example manually
            results = {
                "total_questions": len(eval_df),
                "completed_evaluations": 0,
                "scores": {criterion: [] for criterion in evaluation_criteria.keys()},
                "overall_scores": []
            }

            # Monkeypatch the device issue before evaluation (this is a temporary fix)
            import types
            import torch
            from src.models.deepseek_model import DeepseekModel

            # Store original method
            original_predict = DeepseekModel.predict

            # Define patched method
            def patched_predict(self, question):
                try:
                    # Determine target device - use CUDA if available
                    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

                    # Move model to target device
                    if hasattr(self, 'model'):
                        self.model = self.model.to(target_device)

                    # If the model has a tokenizer, use it and ensure inputs are on the correct device
                    if hasattr(self, 'tokenizer'):
                        # Tokenize input
                        inputs = self.tokenizer(question, return_tensors="pt")

                        # CRUCIAL: Move ALL input tensors to the same device as the model
                        inputs = {k: v.to(target_device) for k, v in inputs.items()}

                        # Generate response with tensors on the correct device
                        with torch.no_grad():
                            output = self.model.generate(
                                **inputs,
                                max_length=512
                            )

                        # Decode the output
                        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        return response
                    else:
                        # If there's no tokenizer, fall back to the original method
                        return original_predict(self, question)

                except Exception as e:
                    logger.error(f"Error generating prediction: {e}")
                    return "I apologize, but I couldn't process this medical question due to a technical issue."



            # Apply the patch
            DeepseekModel.predict = patched_predict

            # Run evaluation
            for idx, example in eval_df.iterrows():
                try:
                    # Get model response
                    response = model_callable({"question": example["question"]})
                    model_answer = response["generations"][0][0]["text"]

                    # Run evaluation for each criterion
                    example_scores = {}
                    for criterion, evaluator in evaluators.items():
                        eval_result = evaluator.evaluate_strings(
                            prediction=model_answer,
                            reference=example["reference"],
                            input=example["question"]
                        )
                        score = eval_result.get("score", 0)
                        example_scores[criterion] = score
                        results["scores"][criterion].append(score)

                    # Calculate overall score for this example
                    overall_score = sum(example_scores.values()) / len(example_scores)
                    results["overall_scores"].append(overall_score)
                    results["completed_evaluations"] += 1

                    # Log progress
                    if idx % 10 == 0 or idx == len(eval_df) - 1:
                        logger.info(f"Evaluated {idx+1}/{len(eval_df)} examples")

                except Exception as e:
                    logger.warning(f"Error evaluating example {idx}: {e}")
                    continue

            # Restore the original method
            DeepseekModel.predict = original_predict

            # Calculate average scores
            final_results = {
                "total_examples": results["total_questions"],
                "evaluated_examples": results["completed_evaluations"],
                "completion_rate": results["completed_evaluations"] / max(1, results["total_questions"])
            }

            # Add average scores for each criterion
            for criterion in evaluation_criteria.keys():
                criterion_scores = results["scores"][criterion]
                if criterion_scores:
                    avg_score = sum(criterion_scores) / len(criterion_scores)
                    final_results[f"avg_{criterion}"] = avg_score

            # Add overall average score
            if results["overall_scores"]:
                final_results["avg_overall_score"] = sum(results["overall_scores"]) / len(results["overall_scores"])
            else:
                final_results["avg_overall_score"] = 0

            logger.info(f"Evaluation completed with overall score: {final_results.get('avg_overall_score', 0):.2f}")
            return final_results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def generate_evaluation_report(self, results):
        """Generate a detailed evaluation report."""
        # Format the results for reporting
        report = {
            "Total Examples": results["total_examples"],
            "Evaluated Examples": results["evaluated_examples"],
            "Completion Rate": f"{results['completion_rate']:.2%}",
            "Overall Score": f"{results.get('avg_overall_score', 0):.2f}/1.00"
        }

        # Add individual criteria scores
        for key, value in results.items():
            if key.startswith("avg_") and key != "avg_overall_score":
                criterion = key[4:].capitalize()  # Remove 'avg_' prefix and capitalize
                report[f"{criterion} Score"] = f"{value:.2f}/1.00"

        return report