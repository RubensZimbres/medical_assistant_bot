"""
Deepseek model implementation for the Medical Assistant Bot.
"""
import os
import torch
from typing import Any, Dict, List, Optional, Union
import logging
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import tempfile
from pathlib import Path

from src.config import (
    MODEL_NAME,
    DEVICE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_STEPS
)
from src.models.base_model import BaseModel
from src.data.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepseekModel(BaseModel):
    """
    Deepseek model implementation for the Medical Assistant Bot.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        use_nested_quant: bool = True
    ):
        """
        Initialize the Deepseek model.

        Args:
            model_name: Name of the model to load
            device: Device to use for inference (cuda or cpu)
            use_4bit: Whether to use 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype for 4-bit models
            bnb_4bit_quant_type: Quantization type for 4-bit models
            use_nested_quant: Whether to use nested quantization
        """
        self.model_name = model_name
        self.device = device

        # Check CUDA availability if device is cuda
        if device == "cuda" and not self.check_cuda_availability():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        # Set up quantization config for 4-bit loading if requested
        self.use_4bit = use_4bit

        if use_4bit:
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

            # Log GPU capability if using half precision
            if compute_dtype == torch.float16 and torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    logger.info("GPU supports bfloat16: training will be accelerated")
        else:
            self.bnb_config = None

        # Initialize model and tokenizer
        logger.info(f"Initializing {model_name} model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Set device map based on device
            device_map = {"": 0} if device == "cuda" else None

            # Load model with quantization if enabled
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.bnb_config if use_4bit else None,
                device_map=device_map
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

        # PEFT/LoRA config for fine-tuning
        self.lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers for DeepSeek
            task_type="CAUSAL_LM",
        )

        # Training parameters
        self.batch_size = BATCH_SIZE
        self.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        self.learning_rate = LEARNING_RATE
        self.max_steps = MAX_STEPS

        # Flag indicating if model is fine-tuned
        self.is_fine_tuned = False

    def prepare_input(self, input_text):
        """Prepare input by ensuring consistent device usage."""
        if hasattr(self, 'tokenizer'):
            # Choose the target device - either consistently use GPU or CPU
            target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Move the entire model to the target device
            self.model = self.model.to(target_device)

            # Tokenize and move tokens to the same device
            tokens = self.tokenizer(input_text, return_tensors="pt")
            return {k: v.to(target_device) for k, v in tokens.items()}
        return input_text

    def load(self, model_path: str) -> None:
        """
        Load a model from a path.

        Args:
            model_path: Path to load the model from
        """
        try:
            logger.info(f"Loading model from {model_path}")

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Set device map based on device
            device_map = {"": 0} if self.device == "cuda" else None

            # Check if this is a PEFT model
            adapter_path = Path(model_path) / "adapter_config.json"
            if adapter_path.exists():
                # Load base model with quantization if enabled
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=self.bnb_config if self.use_4bit else None,
                    device_map=device_map
                )

                # Load the PEFT adapter
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.is_fine_tuned = True
            else:
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map
                )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save(self, model_path: str) -> None:
        """
        Save the model to a path.

        Args:
            model_path: Path to save the model to
        """
        try:
            logger.info(f"Saving model to {model_path}")

            # Create directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)

            # Save the model and tokenizer
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

            logger.info(f"Model saved successfully")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model using the provided datasets.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            **kwargs: Additional training arguments

        Returns:
            Dictionary containing training metrics
        """
        try:
            logger.info("Starting model training")

            # Override default parameters with any provided in kwargs
            batch_size = kwargs.get("batch_size", self.batch_size)
            gradient_accumulation_steps = kwargs.get(
                "gradient_accumulation_steps", self.gradient_accumulation_steps
            )
            learning_rate = kwargs.get("learning_rate", self.learning_rate)
            max_steps = kwargs.get("max_steps", self.max_steps)

            # Create a temporary directory for training outputs
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Set up training arguments
                training_args = TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=2,
                    max_steps=500,
                    learning_rate=learning_rate,
                    fp16=True,
                    logging_steps=10,
                    output_dir=tmp_dir,
                    optim="paged_adamw_8bit"
                )

                # Apply LoRA config to model
                peft_model = get_peft_model(self.model, self.lora_config)

                # Create trainer
                trainer = SFTTrainer(
                    model=peft_model,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    args=training_args,
                    peft_config=self.lora_config,
                    formatting_func=DataLoader.formatting_func,
                )

                # Train the model
                train_result = trainer.train()

                # Update model reference and set fine-tuned flag
                self.model = peft_model
                self.is_fine_tuned = True

                # Extract training metrics
                metrics = {
                    "train_loss": train_result.training_loss,
                    "train_runtime": train_result.metrics.get("train_runtime", 0),
                    "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                }

                # Add evaluation metrics if validation data was provided
                if val_dataset is not None:
                    eval_results = trainer.evaluate()
                    metrics.update({
                        "eval_loss": eval_results.get("eval_loss", 0),
                    })

                logger.info(f"Training completed with metrics: {metrics}")

                return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

# Inside your DeepseekModel class or as a monkey patch

    def predict(self, question):
        """Generate a prediction with proper device management."""
        try:
            # Determine target device
            target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Move model to target device
            if hasattr(self, 'model'):
                self.model = self.model.to(target_device)

            # Tokenize input
            inputs = self.tokenizer(question, return_tensors="pt")

            # CRUCIAL: Move ALL input tensors to the same device as the model
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            # Generate response with tensors on the correct device
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=128
                )

            # Decode the output
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return a fallback response
            return "I apologize, but I couldn't process this medical question due to a technical issue."

    def batch_predict(self, input_texts: List[str], **kwargs) -> List[str]:
        """
        Generate predictions for a batch of input texts.

        Args:
            input_texts: List of input texts to generate predictions for
            **kwargs: Additional prediction arguments

        Returns:
            List of predicted output texts
        """
        return [self.predict(text, **kwargs) for text in tqdm(input_texts, desc="Generating predictions")]
