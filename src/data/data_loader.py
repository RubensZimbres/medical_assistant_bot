"""
Data loading utilities for the Medical Assistant Bot.
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datasets import Dataset, DatasetDict
import logging

from src.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class for loading and preparing datasets for the Medical Assistant Bot.
    """

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame containing the data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    @staticmethod
    def split_dataframe(
        df: pd.DataFrame,
        train_ratio: float = TRAIN_SPLIT,
        val_ratio: float = VAL_SPLIT,
        test_ratio: float = TEST_SPLIT
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into training, validation, and test sets.

        Args:
            df: DataFrame to split
            train_ratio: Fraction of data to use for training
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        # Validate that ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Split ratios must sum to 1.0")

        # First split: training and temp (validation + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=42)

        # Second split: validation and test from temp
        val_ratio_adjusted = val_ratio / temp_ratio
        val_df, test_df = train_test_split(temp_df, test_size=(1-val_ratio_adjusted), random_state=42)

        logger.info(f"Dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test")

        return train_df, val_df, test_df

    @staticmethod
    def convert_to_huggingface_datasets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> DatasetDict:
        """
        Convert pandas DataFrames to a HuggingFace DatasetDict.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            DatasetDict containing train, validation, and test datasets
        """
        try:
            # Convert each DataFrame to a Dataset
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            test_dataset = Dataset.from_pandas(test_df)

            # Combine into a DatasetDict
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })

            logger.info(f"Created HuggingFace DatasetDict with {len(train_dataset)} training, "
                        f"{len(val_dataset)} validation, and {len(test_dataset)} test examples")

            return dataset_dict

        except Exception as e:
            logger.error(f"Error converting to HuggingFace datasets: {e}")
            raise

    @staticmethod
    def format_qa_example(example: Dict) -> str:
        """
        Format a single QA example for the model.

        Args:
            example: Dictionary containing question and answer

        Returns:
            Formatted string
        """
        return f"Question: {example['question']}\nAnswer: {example['answer']}"

    @staticmethod
    def formatting_func(examples: Dict[str, List]) -> List[str]:
        """
        Format a batch of examples for the SFTTrainer.

        Args:
            examples: Dictionary of lists containing the batch data

        Returns:
            List of formatted strings
        """
        output_texts = []
        for i in range(len(examples['question'])):
            text = f"Question: {examples['question'][i]}\nAnswer: {examples['answer'][i]}"
            output_texts.append(text)
        return output_texts
