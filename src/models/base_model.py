"""
Abstract base model class for the Medical Assistant Bot.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the Medical Assistant Bot.
    """
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load a model from a path.
        
        Args:
            model_path: Path to load the model from
        """
        pass
    
    @abstractmethod
    def save(self, model_path: str) -> None:
        """
        Save the model to a path.
        
        Args:
            model_path: Path to save the model to
        """
        pass
    
    @abstractmethod
    def train(self, train_dataset: Any, val_dataset: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model using the provided datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, input_text: str, **kwargs) -> str:
        """
        Generate a prediction for the given input text.
        
        Args:
            input_text: Input text to generate a prediction for
            **kwargs: Additional prediction arguments
            
        Returns:
            Predicted output text
        """
        pass
    
    @abstractmethod
    def batch_predict(self, input_texts: List[str], **kwargs) -> List[str]:
        """
        Generate predictions for a batch of input texts.
        
        Args:
            input_texts: List of input texts to generate predictions for
            **kwargs: Additional prediction arguments
            
        Returns:
            List of predicted output texts
        """
        pass
    
    @staticmethod
    def check_cuda_availability() -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Using CPU.")
        
        return cuda_available
