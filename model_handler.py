"""
FastVLM model handler for image description generation
"""
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
)
from PIL import Image
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set custom Hugging Face cache directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(MODELS_DIR)


class FastVLMHandler:
    """Handle FastVLM model initialization and inference"""

    def __init__(self, model_name: str = "apple/FastVLM-1.5B", device: str = "cuda"):
        """
        Initialize FastVLM model handler

        Args:
            model_name: Model identifier on HuggingFace
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_processor = None
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        logger.info(f"FastVLMHandler initialized for {self.model_name} on {self.device}")

    def load_model(self) -> bool:
        """Load FastVLM model and components from HuggingFace"""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Load image processor
            logger.info("Loading image processor...")
            self.image_processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )

            # Load model with appropriate precision
            logger.info("Loading model weights...")
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )

            self.model.eval()
            self._is_loaded = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def generate_description(
        self,
        image: Image.Image,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Generate description for the given image

        Args:
            image: PIL Image object
            max_length: Maximum length of generated text
            temperature: Sampling temperature

        Returns:
            Generated description or None if inference fails
        """
        if not self._is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            # Process image
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            # Prepare text input
            prompt = "Describe this image:"
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).input_ids.to(self.device)

            # Generate description
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )

            # Decode output
            description = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            return description.strip()

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded and self.model is not None

    def cleanup(self):
        """Cleanup and free resources"""
        if self.model is not None:
            del self.model
        if self.image_processor is not None:
            del self.image_processor
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if self.device == "cuda" else None
        logger.info("Resources cleaned up")
