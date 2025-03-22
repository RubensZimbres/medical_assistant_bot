"""
Data preprocessing module for the Medical Assistant Bot.
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from tqdm import tqdm
import logging

from src.config import GEMINI_API_KEY, GEMINI_MODEL, PROCESSED_DATASET_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class for preprocessing medical QA data using Gemini for content verification.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the DataProcessor.

        Args:
            api_key: Gemini API key (defaults to config)
            model_name: Gemini model name (defaults to config)
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"Initialized DataProcessor with model: {self.model_name}")

    def _create_prompt(self, qa_pairs: List[Dict[str, str]]) -> str:
        """
        Create a prompt for evaluating QA pairs.

        Args:
            qa_pairs: List of dictionaries containing question-answer pairs

        Returns:
            Formatted prompt string
        """
        prompt = "Given these QAs, Tell me IN ONE WORD if the pair is RIGHT or WRONG:\n\n"

        for pair in qa_pairs:
            prompt += f"{pair['question']},\"{pair['answer']}\"\n\n"

        return prompt

    def evaluate_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[str]:
        """
        Use Gemini to evaluate whether QA pairs are correct.

        Args:
            qa_pairs: List of dictionaries containing question-answer pairs

        Returns:
            List of evaluation results ("RIGHT" or "WRONG")
        """
        try:
            prompt = self._create_prompt(qa_pairs)
            response = self.model.generate_content(prompt)
            # Parse the response to extract RIGHT/WRONG evaluations
            result_text = response.text
            print(result_text)
            results = []

            # Simple parsing - look for RIGHT or WRONG in each pair section
            for line in result_text.split('\n'):
                if "RIGHT" in line:
                    results.append("RIGHT")
                elif "WRONG" in line:
                    results.append("WRONG")

            # If we didn't get the expected number of results, log a warning
            if len(results) != len(qa_pairs):
                logger.warning(f"Expected {len(qa_pairs)} results, but got {len(results)}")
                # Fill in missing results as WRONG to be safe
                while len(results) < len(qa_pairs):
                    results.append("WRONG")

            return results

        except Exception as e:
            logger.error(f"Error evaluating QA pairs: {e}")
            # Return all as WRONG in case of error
            return ["WRONG"] * len(qa_pairs)

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 1) -> pd.DataFrame:
        """
        Process a dataset by evaluating QA pairs and filtering for correct pairs.

        Args:
            df: DataFrame containing question and answer columns
            batch_size: Number of QA pairs to evaluate in each batch

        Returns:
            Processed DataFrame with evaluated and filtered QA pairs
        """
        logger.info(f"Processing dataset with {len(df)} rows")

        # Initialize a column for evaluation results
        df['evaluation'] = 'UNKNOWN'

        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Evaluating QA pairs"):
            batch = df.iloc[i:i+batch_size]

            # Create QA pairs for the batch
            qa_pairs = [
                {"question": row["question"], "answer": row["answer"]}
                for _, row in batch.iterrows()
            ]
            print("OK")
            # Evaluate the batch
            evaluations = self.evaluate_qa_pairs(qa_pairs)

            # Update the evaluation column for the batch
            df.loc[batch.index, 'evaluation'] = evaluations

        # Filter for correct QA pairs
        processed_df = df[df['evaluation'] == 'RIGHT'].copy()

        logger.info(f"Processed dataset: {len(processed_df)} correct QA pairs out of {len(df)} total")

        return processed_df

    def load_and_process(self, file_path: str, save_processed: bool = True) -> pd.DataFrame:
        """
        Load a dataset from a file, process it, and optionally save the processed dataset.

        Args:
            file_path: Path to the dataset file
            save_processed: Whether to save the processed dataset

        Returns:
            Processed DataFrame
        """
        try:
            # Load the dataset
            df = pd.read_csv(file_path)

            # Check if required columns exist
            required_columns = ["question", "answer"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")

            # Process the dataset
            processed_df = self.process_dataset(df)

            # Save the processed dataset if requested
            if save_processed:
                processed_df.to_csv(PROCESSED_DATASET_PATH, index=False)
                logger.info(f"Saved processed dataset to {PROCESSED_DATASET_PATH}")

            return processed_df

        except Exception as e:
            logger.error(f"Error loading and processing dataset: {e}")
            raise

    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        # First split: training and temp (validation + test)
        train_df, temp_df = train_test_split(df, test_size=(1-0.8), random_state=42)

        # Second split: validation and test from temp
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        logger.info(f"Dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test")

        return train_df, val_df, test_df
